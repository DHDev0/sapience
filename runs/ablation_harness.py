"""ABLATION MATRIX harness — which learning mechanisms actually EARN THEIR KEEP on the real next-byte task.

Reproducible: fixed seed, fixed corpus (wikitext-2 train windows), fixed step budget, fixed architecture.
For each config we spin a FRESH brain, enable the toggle(s), train N e-prop steps on deterministic windows
(with the SAME per-step life wiring the live brain uses — peptide/glia gate modulation, glia sense+pgain,
plateau handle), then measure HELD-OUT bits/byte on a disjoint validation chunk (brain.bits_per_byte, a clean
no-grad forward). Δbpb = bpb(baseline) − bpb(config); POSITIVE ⇒ the mechanism helped. Multi-seed → mean±std.

Run:   python runs/ablation_harness.py single   # each toggle alone vs baseline
       python runs/ablation_harness.py combo    # full-stack + leave-one-out over the winners
Writes runs/ablation_results.json (incremental).

NOTE on scope: this measures mechanisms that shape the AWAKE next-byte weight update (the faithfulness stack
+ the plasticity §17 tier + peptide/glia gate). ripple/theta (sleep consolidation), embodiment/spatial (world/
nav tasks) operate at the life/sleep/behaviour level — their value is not a next-byte-bpb quantity and is
measured separately in the live system (see §16 endocrine A/B). They are listed but flagged, not silently mixed in.
"""
import os, sys, json, time, random, argparse
sys.path.insert(0, "/home/dander/workspace/zk/sapience")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import torch
from brain.spiking_brain import SpikingBrain
from brain.plateau import DendriticPlateau
from brain.interneurons import SpikingInterneurons
from brain.glia import SpikingGlia
from brain.laminar import LaminarMicrocircuit
from brain.neuropeptides import SpikingNeuropeptides

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32
HIDDEN, LAYERS, SEQ, BS = 4096, 2, 48, 16
STEPS = 600
SEEDS = [0, 1, 2]
RESULTS = "/home/dander/workspace/zk/sapience/runs/ablation_results.json"
RESULTS_COMBO = "/home/dander/workspace/zk/sapience/runs/ablation_combo_results.json"

# toggles that route through set_faith (the faithfulness stack)
FAITH = {"two_compartment", "bounded_synapses", "homeostasis", "btsp", "dale", "dendritic",
         "diff_neuromod", "stochastic", "metabolic"}
# two_compartment is a hard PREREQ for these (their code path is gated on twocomp)
NEEDS_TWOCOMP = {"interneurons", "plateau"}


def load_corpus():
    from datasets import load_dataset                # same source the brain births on (Salesforce/wikitext-2)
    tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    va = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    train = "\n".join(tr["text"])[:400000]
    evf = "\n".join(va["text"])
    ev = evf[len(evf)//3: len(evf)//3 + 6000]      # a disjoint held-out chunk
    return train, ev


def make_brain(seed):
    b = SpikingBrain(DEV, dtype=DT, emb=64, hidden=HIDDEN, layers=LAYERS, seq=SEQ, seed=seed,
                     sparse=True, rec_fanin=32, in_fanin=32, syn_density=0.5)
    b.to(DEV); b._ensure_feedback()
    # baseline foundation = faithful e-prop, RANDOM feedback (DFA), power head, no enhancements. At hidden=4096
    # the power head is NOT yet starved (that's the 128k+ regime) and learns better than NLMS energy here, so it
    # is the stronger FOUNDATION for measuring the other mechanisms; we take the MIN held-out bpb over checkpoints
    # so its known late-divergence doesn't swamp the signal, and also report FINAL bpb as a stability read. The
    # power↔energy head comparison is scale-dependent and done in a dedicated sweep, not this fixed-width matrix.
    b.set_faith(learn_rule="eprop", feedback_mode="random", eprop_lr_scale=2000.0, head_norm="power")
    b.interneurons = SpikingInterneurons(DEV, dtype=DT)
    b.glia = SpikingGlia(DEV, dtype=DT)
    b.laminar = LaminarMicrocircuit(DEV, dtype=DT)
    b.peptides = SpikingNeuropeptides(DEV, dtype=DT)
    b._plateau_obj = DendriticPlateau(DEV, dtype=DT)
    return b


def apply_config(b, toggles):
    tg = set(toggles)
    if tg & NEEDS_TWOCOMP:
        tg.add("two_compartment")                  # auto-add the hard prereq
    for k in tg:
        if k == "learned_fb":       b.set_faith(feedback_mode="learned")
        elif k == "head_energy":    b.set_faith(head_norm="energy", head_lr_scale=2.0)  # ablate power head → NLMS
        elif k == "pc":             b.set_faith(learn_rule="pc")
        elif k in FAITH:            b.set_faith(**{k: True})
        elif k == "stdp":           b.stdp.set_params(on=True)
        elif k == "stp":            b.stp.set_params(on=True)
        elif k == "interneurons":   b.interneurons.set_params(on=True)
        elif k == "plateau":        b._plateau_obj.set_params(on=True)
        elif k == "glia":           b.glia.set_params(on=True)
        elif k == "laminar":        b.laminar.set_params(on=True); b.laminar.rebuild(b)
        elif k == "peptides":       b.peptides.set_params(on=True)
    return tg


def train_eval(toggles, seed, data, ev, eval_every=100):
    torch.manual_seed(seed)
    b = make_brain(seed)
    tg = apply_config(b, toggles)
    rng = random.Random(seed)
    n = data.numel()
    use_pc = "pc" in tg
    best = float("inf")                              # MIN held-out bpb over training = learning capacity,
    for step in range(STEPS):                        #   robust to late divergence (which is itself a stability signal)
        idx = [rng.randint(0, n - SEQ - 2) for _ in range(BS)]
        x = torch.stack([data[i:i + SEQ] for i in idx])
        y = torch.stack([data[i + 1:i + SEQ + 1] for i in idx])
        gate = 1.0
        if "peptides" in tg: gate *= float(b.peptides.plasticity_bias())
        if "glia" in tg:
            gate *= float(b.glia.global_gain())
            b._astro_on = True
            b._astro_pgain = b.glia.pgain_per_layer([c.hid for c in b.cells])
            b._astro_metab_mult = float(b.glia.metab_mult())
        if "plateau" in tg: b._plateau = b._plateau_obj
        b._eprop_step(x, y, gate=gate, pc=use_pc)
        if "glia" in tg: b.glia.sense(getattr(b, "_spk_rate_vec", None))
        if "peptides" in tg: b.peptides.wake_tick(progress=0.5, novelty=0.5, threat=0.1)
        if (step + 1) % eval_every == 0:
            bp = float(b.bits_per_byte(ev))
            if bp == bp: best = min(best, bp)        # nan-safe
    final = float(b.bits_per_byte(ev))
    if final == final: best = min(best, final)
    return best, (final if final == final else 99.0)   # (min=capacity, final=stability)


def measure(name, toggles, data, ev):
    mins, finals = [], []
    for s in SEEDS:
        t0 = time.time()
        mn, fn = train_eval(toggles, s, data, ev)
        mins.append(mn); finals.append(fn)
        print(f"    {name:22s} seed{s} min={mn:.4f} final={fn:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    m = sum(mins) / len(mins)
    sd = (sum((x - m) ** 2 for x in mins) / len(mins)) ** 0.5
    return {"name": name, "toggles": list(toggles), "bpb_mean": round(m, 4),
            "bpb_std": round(sd, 4), "bpb_seeds": [round(v, 4) for v in mins],
            "final_mean": round(sum(finals) / len(finals), 4), "final_seeds": [round(v, 4) for v in finals]}


def save(results):
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)


SINGLES = ["learned_fb", "two_compartment", "bounded_synapses", "homeostasis", "btsp", "dale",
           "dendritic", "diff_neuromod", "stochastic", "metabolic", "head_energy",
           "stdp", "stp", "interneurons", "plateau", "glia", "laminar", "peptides", "pc"]


def run_single():
    data_text, ev = load_corpus()
    b0 = make_brain(0); data = torch.tensor(b0.to_bytes(data_text), device=DEV, dtype=torch.long); del b0
    print(f"corpus: train {data.numel()} bytes, eval {len(ev)} bytes | HIDDEN={HIDDEN} STEPS={STEPS} SEEDS={SEEDS}", flush=True)
    results = {"config": {"hidden": HIDDEN, "layers": LAYERS, "seq": SEQ, "bs": BS, "steps": STEPS,
                          "seeds": SEEDS, "device": str(DEV)}, "runs": []}
    print("[baseline]", flush=True)
    base = measure("baseline", set(), data, ev); results["runs"].append(base); save(results)
    print(f"  baseline bpb = {base['bpb_mean']} ± {base['bpb_std']}", flush=True)
    # a twocomp reference so the twocomp-dependent mechanisms get a fair marginal
    tcref = measure("two_compartment", {"two_compartment"}, data, ev); results["runs"].append(tcref); save(results)
    for t in SINGLES:
        if t == "two_compartment": continue           # already measured as the reference
        r = measure(t, {t}, data, ev)
        ref = tcref if t in NEEDS_TWOCOMP else base
        r["delta_vs_ref"] = round(ref["bpb_mean"] - r["bpb_mean"], 4)   # + = helps
        r["ref"] = ref["name"]
        results["runs"].append(r); save(results)
        print(f"  {t}: bpb {r['bpb_mean']}±{r['bpb_std']}  Δ vs {ref['name']} = {r['delta_vs_ref']:+.4f}", flush=True)
    save(results)
    print("SINGLE ABLATION DONE →", RESULTS, flush=True)


def run_combo():
    """Interaction effects: full-stack of the SINGLE winners + leave-one-out (which are load-bearing together),
    plus the 'everything on' config (all singles) the user wants to run live. Reads the single-ablation JSON."""
    with open(RESULTS) as f:
        prev = json.load(f)
    single = {r["name"]: r for r in prev["runs"]}
    base = single["baseline"]["bpb_mean"]
    # winners = helped their reference by more than a small margin (above seed noise)
    winners = sorted([r["name"] for r in prev["runs"]
                      if r["name"] not in ("baseline",) and r.get("delta_vs_ref", -9) > 0.005])
    data_text, ev = load_corpus()
    b0 = make_brain(0); data = torch.tensor(b0.to_bytes(data_text), device=DEV, dtype=torch.long); del b0
    print(f"WINNERS (Δ>0.005 vs ref): {winners}", flush=True)
    combo = {"config": prev["config"], "baseline_bpb": base, "winners": winners, "runs": []}

    full = measure("full_stack(winners)", set(winners), data, ev)
    full["delta_vs_baseline"] = round(base - full["bpb_mean"], 4)
    combo["runs"].append(full)
    print(f"  full_stack {full['bpb_mean']}±{full['bpb_std']}  Δ vs baseline = {full['delta_vs_baseline']:+.4f}", flush=True)
    _c=open(RESULTS_COMBO,"w"); json.dump({"single": prev["runs"], "combo": combo}, _c, indent=2); _c.close()

    for w in winners:                                  # leave-one-out: does removing w HURT the full stack?
        loo = measure(f"LOO-{w}", set(winners) - {w}, data, ev)
        loo["loo_removed"] = w
        loo["hurt_by_removing"] = round(loo["bpb_mean"] - full["bpb_mean"], 4)   # + = w is load-bearing
        combo["runs"].append(loo)
        _c=open(RESULTS_COMBO,"w"); json.dump({"single": prev["runs"], "combo": combo}, _c, indent=2); _c.close()
        print(f"  LOO-{w}: {loo['bpb_mean']}  removing {w} hurts by {loo['hurt_by_removing']:+.4f}", flush=True)

    everything = measure("everything(all singles)", set(SINGLES), data, ev)
    everything["delta_vs_baseline"] = round(base - everything["bpb_mean"], 4)
    combo["runs"].append(everything)
    _c=open(RESULTS_COMBO,"w"); json.dump({"single": prev["runs"], "combo": combo}, _c, indent=2); _c.close()
    print(f"  everything-on {everything['bpb_mean']}  Δ vs baseline = {everything['delta_vs_baseline']:+.4f}", flush=True)
    print("COMBO ABLATION DONE →", RESULTS, flush=True)


RESULTS_CONN = "/home/dander/workspace/zk/sapience/runs/connectome_results.json"
RESULTS_HORIZON = "/home/dander/workspace/zk/sapience/runs/horizon_results.json"


def train_curve(toggles, seed, data, ev, steps, eval_every=250, energy_head=True):
    """Train `steps` and record the held-out bpb learning CURVE. Uses the STABLE NLMS energy head as the fixed
    foundation (so late divergence of a power head can't confound a long-horizon mechanism comparison)."""
    torch.manual_seed(seed); b = make_brain(seed)
    if energy_head:
        b.set_faith(head_norm="energy", head_lr_scale=2.0, head_energy_eps=1e-2)
    tg = apply_config(b, toggles)
    rng = random.Random(seed); n = data.numel(); use_pc = "pc" in tg; curve = []
    for step in range(steps):
        idx = [rng.randint(0, n - SEQ - 2) for _ in range(BS)]
        x = torch.stack([data[i:i + SEQ] for i in idx]); y = torch.stack([data[i + 1:i + SEQ + 1] for i in idx])
        gate = 1.0
        if "peptides" in tg: gate *= float(b.peptides.plasticity_bias())
        if "glia" in tg:
            gate *= float(b.glia.global_gain()); b._astro_on = True
            b._astro_pgain = b.glia.pgain_per_layer([c.hid for c in b.cells]); b._astro_metab_mult = float(b.glia.metab_mult())
        if "plateau" in tg: b._plateau = b._plateau_obj
        b._eprop_step(x, y, gate=gate, pc=use_pc)
        if "glia" in tg: b.glia.sense(getattr(b, "_spk_rate_vec", None))
        if "peptides" in tg: b.peptides.wake_tick(progress=0.5, novelty=0.5, threat=0.1)
        if (step + 1) % eval_every == 0:
            curve.append(round(float(b.bits_per_byte(ev)), 4))
    return curve


HORIZON_CONFIGS = {
    "baseline": set(),
    "dale": {"dale"},
    "dale+homeostasis": {"dale", "homeostasis"},
    "full_stack": {"bounded_synapses", "dale", "dendritic", "diff_neuromod", "interneurons", "pc", "peptides", "stdp", "stochastic", "stp"},
    "everything": set(SINGLES) - {"head_energy"},   # everything except the head toggle (energy IS the foundation here)
    "homeostasis": {"homeostasis"},
    "metabolic": {"metabolic"},
    "learned_fb": {"learned_fb"},
}


def run_horizon(steps=3600, seeds=(0, 1)):
    """Does the 600-step ranking HOLD with age? Train the key configs 6x longer on the STABLE energy head and
    record learning curves — look for crossovers (does everything-on catch up? does homeostasis/metabolic/
    learned_fb pay off late? does dale stay ahead?)."""
    data_text, ev = load_corpus()
    b0 = make_brain(0); data = torch.tensor(b0.to_bytes(data_text), device=DEV, dtype=torch.long); del b0
    out = {"config": {"hidden": HIDDEN, "steps": steps, "eval_every": 250, "seeds": list(seeds),
                      "head": "energy(NLMS) fixed foundation"}, "curves": {}}
    print(f"HORIZON sweep: {steps} steps, energy head, configs={list(HORIZON_CONFIGS)}", flush=True)
    for name, tg in HORIZON_CONFIGS.items():
        allc = []
        for s in seeds:
            t0 = time.time(); c = train_curve(tg, s, data, ev, steps)
            allc.append(c); print(f"  {name:18s} seed{s} min={min(c):.3f} end={c[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        L = min(len(c) for c in allc)
        mean = [round(sum(c[i] for c in allc) / len(allc), 4) for i in range(L)]
        out["curves"][name] = {"mean_curve": mean, "best": round(min(min(c) for c in allc), 4),
                               "end": round(sum(c[-1] for c in allc) / len(allc), 4), "seed_curves": allc}
        with open(RESULTS_HORIZON, "w") as f: json.dump(out, f, indent=2)
    # crossover report: best-so-far ranking at each checkpoint
    xs = list(range(250, steps + 1, 250))
    print("\nstep   " + "  ".join(f"{n[:9]:>9s}" for n in HORIZON_CONFIGS), flush=True)
    for i, st in enumerate(xs):
        row = []
        for n in HORIZON_CONFIGS:
            mc = out["curves"][n]["mean_curve"]
            row.append(f"{mc[i]:9.3f}" if i < len(mc) else "    -    ")
        print(f"{st:5d}  " + "  ".join(row), flush=True)
    print("HORIZON SWEEP DONE →", RESULTS_HORIZON, flush=True)


def connectome_columns(hid, fanin, seed, lam_frac=0.05, longrange=0.12):
    """Distance-dependent (small-world) presynaptic columns on a ring of `hid` neurons: (1−longrange) of each
    row's fan-in drawn LOCALLY with an exponential distance rule (Ercsey-Ravasz 2013 cortical EDR → high
    clustering), the rest long-range random (Watts-Strogatz rewiring → short path length). Replaces the flat
    uniform-random `_seed_csr` columns; same fan-in count so the CSR structure (crow/row) is untouched."""
    g = torch.Generator().manual_seed(int(seed))
    lam = max(2.0, hid * lam_frac)
    rows = torch.arange(hid).unsqueeze(1)
    n_long = int(round(fanin * longrange)); n_loc = fanin - n_long
    off = torch.empty(hid, n_loc).exponential_(1.0 / lam, generator=g).round().long() + 1
    sign = torch.randint(0, 2, (hid, n_loc), generator=g) * 2 - 1
    loc = (rows + sign * off) % hid
    lon = torch.randint(0, hid, (hid, n_long), generator=g)
    return torch.cat([loc, lon], dim=1).reshape(-1).to(torch.int32)


def connectome_rewire(b, seed):
    """Rewire every sparse cell to connectome stats: distance-dependent columns + LOG-NORMAL synaptic weight
    magnitudes (Song 2005 / Buzsáki-Mizuseki 2014 — a few strong, many weak), mean-|w| matched to the baseline
    N(0,1/√f) init so ONLY the distribution/topology differs, not the overall scale."""
    for c in b.cells:
        if not b._is_sparse(c):
            continue
        hid, f = c.hid, c.rec_fanin
        cols = connectome_columns(hid, f, seed + hid)
        c.rec_col.data = cols.to(c.rec_col.device)
        nnz = cols.numel()
        g = torch.Generator().manual_seed(int(seed) + 7)
        mag = torch.empty(nnz).log_normal_(mean=-0.5, std=0.7, generator=g)
        mag = mag / mag.mean() * (0.7979 / (f ** 0.5))       # match mean|w| of N(0,1/√f) = √(2/π)/√f
        sgn = (torch.randint(0, 2, (nnz,), generator=g) * 2 - 1).float()
        c.rec_val.data = (mag * sgn).to(c.rec_val.device, c.rec_val.dtype)


def train_traj(rewire, seed, data, ev, eval_every=100):
    """Like train_eval but also returns the checkpoint LEARNING CURVE (for the connectome speed comparison)."""
    torch.manual_seed(seed); b = make_brain(seed)
    if rewire:
        connectome_rewire(b, seed)
    rng = random.Random(seed); n = data.numel(); traj = []
    for step in range(STEPS):
        idx = [rng.randint(0, n - SEQ - 2) for _ in range(BS)]
        x = torch.stack([data[i:i + SEQ] for i in idx]); y = torch.stack([data[i + 1:i + SEQ + 1] for i in idx])
        b._eprop_step(x, y, gate=1.0)
        if (step + 1) % eval_every == 0:
            traj.append(round(float(b.bits_per_byte(ev)), 4))
    return traj


def run_connectome():
    """The one biophysical A/B worth building: real-connectome-stats wiring vs random sparse. Same seed/data;
    compare learning curve (speed) + best bpb (quality)."""
    data_text, ev = load_corpus()
    b0 = make_brain(0); data = torch.tensor(b0.to_bytes(data_text), device=DEV, dtype=torch.long); del b0
    out = {"config": {"hidden": HIDDEN, "steps": STEPS, "seeds": SEEDS, "eval_every": 100}, "arms": {}}
    for arm, rewire in (("random_sparse", False), ("connectome", True)):
        trajs = []
        for s in SEEDS:
            t0 = time.time(); tj = train_traj(rewire, s, data, ev)
            trajs.append(tj); print(f"  {arm:14s} seed{s} curve={tj}  ({time.time()-t0:.0f}s)", flush=True)
        # mean curve + best
        L = min(len(t) for t in trajs)
        mean_curve = [round(sum(t[i] for t in trajs) / len(trajs), 4) for i in range(L)]
        best = round(min(min(t) for t in trajs), 4)
        out["arms"][arm] = {"mean_curve": mean_curve, "best_bpb": best, "seed_curves": trajs}
        with open(RESULTS_CONN, "w") as f: json.dump(out, f, indent=2)
    r, c = out["arms"]["random_sparse"], out["arms"]["connectome"]
    print(f"\nrandom_sparse best={r['best_bpb']} curve={r['mean_curve']}", flush=True)
    print(f"connectome    best={c['best_bpb']} curve={c['mean_curve']}", flush=True)
    print(f"Δbest (random−connectome) = {round(r['best_bpb']-c['best_bpb'],4):+.4f}  (+ = connectome better)", flush=True)
    print("CONNECTOME A/B DONE →", RESULTS_CONN, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", nargs="?", default="single")
    a = ap.parse_args()
    if a.mode == "single":
        run_single()
    elif a.mode == "combo":
        run_combo()
    elif a.mode == "connectome":
        run_connectome()
    elif a.mode == "horizon":
        run_horizon()
