"""LIVE ORCHESTRATOR — ablation-informed online optimisation of the 256k brain's active mechanism set.

Runs independently of any Claude session (like collapse_watchdog.py): polls the live brain, and every CYCLE
seconds makes ONE attributable change, measures its effect on held-out-ish bits/byte over the next cycle, and
KEEPS or REVERTS it — a greedy forward-selection over the active set, live, at scale.

Why this and not "turn everything on": the §20 ablation matrix measured that the mechanisms do NOT compose
(full-stack < baseline; everything-on is the WORST config; interneurons/STP help alone but interfere stacked),
so the optimal operating point is a small measured subset, not enable-all. But that matrix is at hidden=4096;
the 256k regime (where the NLMS energy head is REQUIRED, not optional) can interact differently — so this daemon
re-runs the search online at the real scale, anchored on what transferred.

  ANCHORS (always re-asserted): head_norm=energy (scale-critical — the power head freezes at 256k), dale
    (the one mechanism load-bearing in combination), homeostasis (the long-horizon over-excitation brake).
  CANDIDATES (A/B'd one at a time, kept only if they lower bpb at scale): the stability-positive §17 tier.
  NEVER auto-enabled: learned_fb (diverged to bpb 5.68 in the ablation) — left for manual, lower-lr trials.

Run:  nohup python runs/orchestrator.py &   (writes runs/orchestrator.log + runs/orchestrator_state.json;
                                              appends decisions to runs/tuning_log.md)
"""
import json, time, urllib.request, os, datetime

BASE = "http://127.0.0.1:8199"
LOG = "/home/dander/workspace/zk/sapience/runs/orchestrator.log"
STATE = "/home/dander/workspace/zk/sapience/runs/orchestrator_state.json"
TUNE = "/home/dander/workspace/zk/sapience/runs/tuning_log.md"

CYCLE = 1200                 # 20 min per A/B step — long enough for a bpb change to show at 256k
SETTLE = 180                 # let a change settle before we start averaging its bpb
POLL = 30
# Anchors chosen from the HORIZON sweep (runs/horizon_results.json), not the 600-step matrix: dale is the
# compounding long-horizon winner (3.99, advantage grows with age); homeostasis was REMOVED — it is a monotonic
# long-horizon DESTABILISER (homeostasis-alone 4.59→5.21; it poisons dale: dale+homeostasis diverges). Runaway
# protection is the always-on over-excitation attention brake + the watchdog, not the homeostasis threshold toggle.
ANCHORS = [("cortex", {"head_norm": "energy", "head_lr_scale": 2.0}),
           ("cortex", {"dale": True})]
# metabolic FIRST — the horizon sweep's clean late-bloomer (600-step loser → long-horizon #2 at 4.09 < baseline)
CANDIDATES = ["metabolic", "glia", "dendritic", "stdp", "bounded_synapses", "diff_neuromod",
              "stochastic", "pc", "stp", "interneurons", "peptides"]
# map candidate -> (target, payload) for /api/net
CAND_APPLY = {
    "glia": ("glia", {"on": True}), "stdp": ("stdp", {"on": True}), "stp": ("stp", {"on": True}),
    "interneurons": ("interneurons", {"on": True}), "peptides": ("peptides", {"on": True}),
    "dendritic": ("cortex", {"dendritic": True}), "bounded_synapses": ("cortex", {"bounded_synapses": True}),
    "diff_neuromod": ("cortex", {"diff_neuromod": True}), "stochastic": ("cortex", {"stochastic": True}),
    "pc": ("cortex", {"learn_rule": "pc"}), "metabolic": ("cortex", {"metabolic": True}),
}
CAND_REVERT = {
    "glia": ("glia", {"on": False}), "stdp": ("stdp", {"on": False}), "stp": ("stp", {"on": False}),
    "interneurons": ("interneurons", {"on": False}), "peptides": ("peptides", {"on": False}),
    "dendritic": ("cortex", {"dendritic": False}), "bounded_synapses": ("cortex", {"bounded_synapses": False}),
    "diff_neuromod": ("cortex", {"diff_neuromod": False}), "stochastic": ("cortex", {"stochastic": False}),
    "pc": ("cortex", {"learn_rule": "eprop"}), "metabolic": ("cortex", {"metabolic": False}),
}
KEEP_MARGIN = 0.02           # candidate must lower bpb by >0.02 to be kept (else revert — Occam)
BPB_UNSAFE = 9.0             # above this, revert the pending change immediately (watchdog owns true rescue)


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=8) as r:
        return json.load(r)


def _post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f: f.write(line + "\n")
    except Exception:
        pass


def tune(msg):
    try:
        with open(TUNE, "a") as f:
            f.write(f"\n[orchestrator {datetime.datetime.now():%Y-%m-%d %H:%M}] {msg}\n")
    except Exception:
        pass


def bpb_now(samples=5):
    """Average bpb over a few polls to smooth the per-step noise."""
    vals = []
    for _ in range(samples):
        try:
            st = _get("/api/state").get("state", {})
            v = st.get("bpb")
            if isinstance(v, (int, float)) and v == v:
                vals.append(float(v))
        except Exception:
            pass
        time.sleep(POLL)
    return sum(vals) / len(vals) if vals else None


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"kept": [], "tried": [], "pending": None, "baseline_bpb": None}


def save_state(s):
    try:
        json.dump(s, open(STATE, "w"), indent=2)
    except Exception:
        pass


def assert_anchors():
    for tgt, payload in ANCHORS:
        try:
            _post("/api/net", {"target": tgt, **payload})
        except Exception as e:
            log(f"anchor {tgt} {payload} failed: {e}")


def main():
    log("orchestrator start — anchors = energy-head + dale + homeostasis; greedy A/B over the §17 tier")
    # wait for the brain to be awake
    for _ in range(60):
        try:
            if _get("/api/diag").get("status") in ("awake", "sleeping"):
                break
        except Exception:
            pass
        time.sleep(10)
    assert_anchors()
    tune("orchestrator online. Anchors set (NLMS energy head + dale + homeostasis). Greedy A/B forward-selection "
         "over the §17 tier begins; keep-if-Δbpb>0.02, revert otherwise. learned_fb excluded (ablation: diverges).")
    s = load_state()

    while True:
        time.sleep(SETTLE)
        assert_anchors()
        cur = bpb_now()
        if cur is None:
            log("no bpb reading; retrying"); time.sleep(CYCLE); continue

        # resolve a pending A/B first
        if s.get("pending"):
            cand = s["pending"]; base = s.get("baseline_bpb")
            improved = base is not None and cur < base - KEEP_MARGIN
            unsafe = cur > BPB_UNSAFE
            if improved and not unsafe:
                s["kept"].append(cand)
                log(f"KEEP {cand}: bpb {base:.3f} -> {cur:.3f} (Δ {base-cur:+.3f})")
                tune(f"KEPT {cand} — bpb {base:.3f}→{cur:.3f} at 256k scale.")
            else:
                tgt, payload = CAND_REVERT[cand]
                try: _post("/api/net", {"target": tgt, **payload})
                except Exception: pass
                why = "unsafe" if unsafe else "no gain"
                log(f"REVERT {cand} ({why}): bpb {base:.3f} -> {cur:.3f}")
                tune(f"REVERTED {cand} ({why}) — bpb {base:.3f}→{cur:.3f}.")
            s["pending"] = None; s["baseline_bpb"] = None
            save_state(s)
            continue

        # start the next A/B: pick an untried candidate
        remaining = [c for c in CANDIDATES if c not in s["tried"]]
        if not remaining:
            log(f"all candidates tried. kept set = {s['kept']}. idling (anchors held).")
            time.sleep(CYCLE); continue
        cand = remaining[0]
        s["tried"].append(cand); s["baseline_bpb"] = cur; s["pending"] = cand
        tgt, payload = CAND_APPLY[cand]
        try:
            _post("/api/net", {"target": tgt, **payload})
            log(f"TEST {cand}: baseline bpb {cur:.3f}; enabled {tgt} {payload}; measuring next cycle")
        except Exception as e:
            log(f"enable {cand} failed: {e}"); s["pending"] = None
        save_state(s)
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
