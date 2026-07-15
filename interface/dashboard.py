#!/usr/bin/env python
"""
dashboard.py — ONE place to run, watch, and talk to the brain.

A single self-contained web control-plane that replaces `tail -f` + `tensorboard` + `tui`:
live metric charts, the stream of thought / life log / vision, a chat box to talk to it, and
controls to LAUNCH, STOP (graceful), KILL (force), and RE-LAUNCH from a checkpoint on a
different device (CPU ↔ GPU ↔ multi-GPU) with different parallelism — all from the browser or
the CLI. TensorBoard logging keeps running underneath for deep dives.

  python interface/dashboard.py                      # start the board + a fresh brain, print the URL
  python interface/dashboard.py --checkpoint DIR     # start the board resuming a saved run
  python interface/dashboard.py --no-autostart       # just the board; launch the brain from the web
  python interface/dashboard.py --port 8080
  python interface/dashboard.py --stop [DIR]         # graceful stop the running brain (checkpoints)
  python interface/dashboard.py --kill               # force-kill the dashboard process

Then open http://localhost:8080 . Device 'auto' maximises compute for the detected hardware
(every GPU if present, else all CPU cores).
"""
import os, sys, json, time, argparse, threading, signal, glob
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (interface/ is a subfolder)
sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = _ROOT
BASE = os.path.join(_ROOT, "runs")               # all run folders live under runs/
CHART_KEYS = [                                   # (state field, label, lower_is_better, y-unit)
    ("understanding", "understanding ↑", False, ""), ("bpb", "bits/byte ↓", True, ""),
    ("neurons", "neurons (fixed)", False, ""), ("synapses", "synapses (active) ↑", False, ""),
    ("parameters", "parameters ↑", False, ""),
    ("synapse_density", "synapse density", False, ""), ("tps", "thoughts/sec", False, "/s"),
    ("teacher_cps", "teacher speed", False, " c/s"), ("think_bps", "think speed", False, " B/s"),
    ("learn_bps", "learn speed", False, " B/s"), ("novelty", "novelty", False, ""),
    ("model_gb", "model size", False, " GB"), ("episodes", "episodes", False, ""),
    ("debt", "sleep debt", False, ""), ("time_sense", "time-sense ↑", False, ""),
    ("perplexity_train", "train perplexity ↓", True, ""), ("perplexity_gen", "gen perplexity", False, ""),
    ("gen_entropy", "gen entropy", False, " b"), ("spike_rate", "spike rate", False, ""),
    ("replay_mb", "replay buffer", False, " MB"), ("lived_chars", "lived chars", False, ""),
    ("ram_gb", "RAM used", False, " GB"), ("ram_pct", "RAM % of box", False, ""),
    ("vram_gb", "VRAM used", False, " GB"), ("vram_pct", "VRAM % of GPU", False, ""),
    ("storage_gb", "storage used", False, " GB"), ("disk_free_gb", "disk free", False, " GB"),
    ("cerebellum_mse", "cerebellum MSE ↓", True, ""), ("bg_spike_rate", "§2 BG spike rate", False, ""),
    ("hippo_spike_rate", "§4 hippo spike rate", False, ""), ("hippo_fidelity", "§4 recall fidelity ↑", False, ""),
    ("bg_policy_entropy", "§2 policy entropy", False, " b"), ("replay_total", "replays (cumulative)", False, ""),
    ("ripple_rate", "SWR ripple rate", False, ""), ("gated_commit_fraction", "SWR gated-commit frac", False, ""),
    ("seq_recall_acc", "§17 sequence recall ↑", False, ""), ("n_trajectories", "§17 trajectories", False, ""),
    ("world_return", "§17 world return ↑", False, ""), ("world_return_random", "§17 random baseline", False, ""),
    ("advantage", "§17 advantage ↑", False, ""), ("world_success_rate", "§17 goal reached ↑", False, ""),
    ("world_steps_to_goal", "§17 steps to goal ↓", True, ""), ("world_surprise", "§17 world surprise ↓", True, ""),
    # cortex learning-health leading indicators (nested under net.weights; flattened into history below). These
    # diagnose WHY the net does/doesn't learn: head_w_std collapsing to init + head_update_mag≈0 = starved readout;
    # fb_align_cos≈0 = feedback not aligning; mem_mag climbing = representation runaway; update_mag = recurrent Δw.
    ("head_w_std", "readout head weight std ↑", False, ""), ("head_update_mag", "head Δw/step ↑", False, ""),
    ("fb_align_cos", "feedback↔head align ↑", False, ""), ("mem_mag", "representation |v|", False, ""),
    ("update_mag", "recurrent Δw/step", False, ""), ("grad_mag", "readout grad", False, ""),
    ("attention", "attention (self-lr)", False, ""), ("eff_lr_scale", "effective lr", False, ""),
    ("loss_ema", "loss EMA ↓", True, ""), ("next_byte_acc", "next-byte acc ↑", False, ""),
]


class Controller:
    """Owns ONE BrainLife at a time; can stop it and start another (e.g. resume the same
    checkpoint on a different device) without taking the web server down."""

    def __init__(self):
        self.life = None
        self.thread = None
        self.run_dir = None
        self.config = {}
        self.compute = {}
        self.latest = {}                          # merged live state (thought, mind, metrics…)
        self.logs = deque(maxlen=400)
        self.history = {k: deque(maxlen=1200) for k, *_ in CHART_KEYS}
        self.status = "idle"                      # idle | being_born | awake | sleeping | stopping | stopped
        self.birth_status = ""
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------- #
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg):
        from brain.life import BrainLife, resolve_compute, latest_run
        if self.running():
            self.stop()
        ckpt = cfg.get("checkpoint")
        resume = bool(ckpt)
        if resume:
            self.run_dir = ckpt if os.path.isabs(ckpt) else os.path.join(BASE, ckpt)
        else:
            import datetime
            self.run_dir = os.path.join(BASE, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.run_dir, exist_ok=True)
        comp = resolve_compute(cfg.get("device", "auto"), cfg.get("threads"))
        self.compute = comp
        self.config = cfg
        self.latest = {}; self.status = "resuming" if resume else "being_born"
        self.birth_status = "resuming…" if resume else "starting…"
        for d in self.history.values():
            d.clear()
        with open(os.path.join(BASE, "CURRENT"), "w") as f:
            f.write(self.run_dir)

        def worker():
            try:
                life = BrainLife(self.run_dir, core=cfg.get("core", "spiking"),
                                 emb=int(cfg.get("emb", 128)), hidden=int(cfg.get("hidden", 512)),
                                 layers=int(cfg.get("layers", 2)),
                                 use_teacher=cfg.get("teacher", False), use_visual=cfg.get("visual", True),
                                 budget=float(cfg.get("budget", 0.10)),
                                 min_awake=float(cfg.get("min_awake", 120)), max_awake=float(cfg.get("max_awake", 600)),
                                 device=comp["device"], dtype=cfg.get("dtype", comp["dtype"]), threads=comp["threads"],
                                 resonate_k=int(cfg.get("resonate_k", 4)),
                                 syn_density=float(cfg.get("syn_density", 0.5)),
                                 sparse=cfg.get("sparse", None),
                                 sparse_hidden_threshold=int(cfg.get("sparse_hidden_threshold", 8192)),
                                 rec_fanin=int(cfg.get("rec_fanin", 64)),
                                 in_fanin=int(cfg.get("in_fanin", 64)),
                                 learn_rule=cfg.get("learn_rule", "eprop"),   # faithful default; "bptt" = opt-in fast ref
                                 eprop_lr_scale=float(cfg.get("eprop_lr_scale", 2000.0)),
                                 max_model_gb=float(cfg.get("max_model_gb", 14.0)),
                                 max_log_mb=float(cfg.get("max_log_mb", 20.0)),
                                 max_tb_mb=float(cfg.get("max_tb_mb", 60.0)),
                                 resume=resume, use_tb=True)
                self.life = life
                if not resume:                            # apply the launch form's soft/dev fields
                    self.set_params(cfg)
                life.log_cb = self._on_log
                life.birth(on_update=self._on_state)
                life.run(on_update=self._on_state)
            except Exception as e:
                self._on_log(f"[dashboard] life crashed: {e}")
            finally:
                self.status = "stopped"
                self.life = None

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        return {"ok": True, "run_dir": self.run_dir, "compute": comp}

    def stop(self, timeout=45):
        """Graceful: ask the life to checkpoint and exit, then join."""
        if self.life is not None:
            self.status = "stopping"
            self.life.request_stop()
        t = self.thread
        if t is not None:
            t.join(timeout=timeout)
        self.status = "stopped"
        return {"ok": True}

    def save(self):
        """Force a checkpoint right now (weights + full config), without stopping."""
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        self.life.save_life()
        return {"ok": True, "run_dir": self.run_dir}

    def kill(self):
        """Force: stop the brain NOW (no checkpoint) and WAIT for the worker to actually exit — the
        old version nulled the thread before waiting, so the loop + the sense thread (stuck in a
        Sonnet call) kept running while the board already showed 'stopped'."""
        self.status = "stopping"
        L, t = self.life, self.thread
        if L is not None:
            L._killed = True
            L.request_stop()                       # _running=False AND kill any in-flight teacher call
        if t is not None:
            t.join(timeout=25)                     # the worker's finally also joins the sense thread
        stopped = (t is None) or (not t.is_alive())
        self.thread = None
        self.life = None
        self.status = "stopped"
        if not stopped:
            self._on_log("[kill] worker did not exit within 25s (abandoned; no more learning occurs)")
        return {"ok": True, "stopped": stopped}

    def chat(self, text):
        if self.life is not None and text.strip():
            self.life.inject(text.strip())
            self.logs.append(f"you → brain: {text.strip()}")
            return {"ok": True}
        return {"ok": False, "err": "no live brain"}

    def set_params(self, d):
        """Change hyperparameters LIVE on the running brain — no restart, takes effect on the
        next loop iteration. Structural changes (device/core/checkpoint) need /api/start."""
        L = self.life
        if L is None:
            self.config.update(d or {})
            return {"ok": False, "err": "no live brain; staged for next launch", "staged": d}
        import torch
        applied = {}
        for k, v in (d or {}).items():
            try:
                if k in ("budget", "min_awake", "max_awake", "debt_threshold", "max_sleep",
                         "perceive_gap", "max_log_mb", "max_tb_mb", "novelty_gate"):
                    setattr(L, k, float(v)); applied[k] = float(v)
                elif k == "think_chunk":
                    L.think_chunk = max(1, int(v)); applied[k] = L.think_chunk
                elif k in ("sleep_chunks", "sleep_steps", "sleep_seq"):
                    setattr(L, k, max(1, int(v))); applied[k] = getattr(L, k)   # scale-aware sleep replay load
                elif k == "resonate_k":
                    L.resonate_k = max(1, int(v)); applied[k] = L.resonate_k
                elif k == "threads":
                    n = max(1, int(v)); L.threads = n; torch.set_num_threads(n); applied[k] = n
                elif k == "learn_steps":
                    L.learn_steps = max(1, int(v)); applied[k] = L.learn_steps
                elif k == "max_model_gb":
                    L.max_model_gb = float(v); L.brain.max_model_gb = float(v); applied[k] = float(v)
                elif k == "grow_add":
                    L.grow_add = max(0, int(v)); applied[k] = L.grow_add
                elif k in ("grow_until", "prune_until"):
                    setattr(L.brain, k, int(v)); applied[k] = int(v)
                elif k in ("freeze_growth", "freeze_sleep", "freeze_learning"):
                    setattr(L, k, bool(v)); applied[k] = bool(v)
                elif k == "sleep_mode":                          # §16: buffer vs GENERATIVE self-replay
                    L.sleep_mode = v if v in ("buffer", "generative") else L.sleep_mode; applied[k] = L.sleep_mode
                elif k in ("gr_dreams", "gr_dream_len"):
                    setattr(L, k, max(1, int(v))); applied[k] = getattr(L, k)
                elif k in ("gr_temperature", "gr_anchor_frac"):
                    setattr(L, k, max(0.0, float(v))); applied[k] = getattr(L, k)
                elif k == "neurogenesis":                        # §16 adult-DG neurogenesis toggle
                    L.neurogenesis = (v if isinstance(v, bool) else str(v).strip().lower()
                                      not in ("false", "0", "off", "no", "")); applied[k] = L.neurogenesis
                elif k in ("neurogenesis_add", "neurogenesis_every"):
                    setattr(L, k, max(1, int(float(v)))); applied[k] = getattr(L, k)
                elif k == "hard_disk_gb":
                    L.memory.hard = int(float(v) * 1e9); applied[k] = float(v)
                elif k in ("visual", "teacher"):
                    setattr(L, "use_" + k, bool(v)); applied[k] = bool(v)
                else:
                    continue                                  # structural → ignored here
                self.config[k] = applied[k]
            except Exception as e:
                applied[k] = "err:" + str(e)[:40]
        self.logs.append(f"[live-tune] {applied}")
        return {"ok": True, "applied": applied}

    def teach(self, d):
        """Teach the live brain specific content NOW (text / wiki topic / web url / local path)."""
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        return self.life.teach(text=d.get("text"), topic=d.get("topic"), url=d.get("url"),
                               path=d.get("path"), label=d.get("label"))

    def focus(self, d):
        """Redirect the learning feed live (topics / urls / mode / label)."""
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        return self.life.focus(topics=d.get("topics"), urls=d.get("urls"),
                               mode=d.get("mode"), label=d.get("label"))

    # ---- tools / other AIs (register, install, run, toggle) --------- #
    def _tools(self):
        return self.life.tools if self.life is not None else None
    def tools_list(self):
        t = self._tools(); return {"tools": t.list() if t else []}
    def tools_add(self, d):
        t = self._tools(); return t.add(d) if t else {"ok": False, "err": "no live brain"}
    def tools_remove(self, d):
        t = self._tools(); return t.remove(d.get("name")) if t else {"ok": False, "err": "no live brain"}
    def tools_toggle(self, d):
        t = self._tools(); return t.toggle(d.get("name"), d.get("enabled"), d.get("autonomous")) if t else {"ok": False}
    def tools_install(self, d):
        t = self._tools(); return t.install(d.get("name")) if t else {"ok": False, "err": "no live brain"}
    def tools_run(self, d):
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        return self.life.use_tool(d.get("name"), d.get("input"), learn=d.get("learn", True))

    # ---- per-module net tuning + sensory observation replay -------- #
    def set_net(self, d):
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        return self.life.set_net(d.get("target"), {k: v for k, v in d.items() if k != "target"})

    def edit_arch(self, d):
        """Live per-part neuron/synapse surgery: grow_neurons | grow_synapses | prune_synapses |
        refresh_synapses on cortex/cerebellum/bg/hippocampus."""
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        return self.life.edit_arch(d.get("target"), d.get("op"),
                                   amount=d.get("amount"), density=d.get("density"))

    def arch(self):
        """Fresh per-part neuron/synapse census straight from the live brain (works during birth
        too — the network exists before the living loop's first state tick)."""
        if self.life is not None:
            try: return self.life._arch_diag()
            except Exception as e: return {"err": str(e)[:80]}
        return self.latest.get("arch", {})

    def set_device(self, d):
        """Place a region (or 'all') on its own device: {target, device}."""
        if self.life is None:
            return {"ok": False, "err": "no live brain"}
        return self.life.set_part_device(d.get("target"), d.get("device"))

    def resources(self):
        """Fresh RAM / VRAM / storage usage vs limits."""
        if self.life is not None:
            try: return self.life._resources()
            except Exception as e: return {"err": str(e)[:80]}
        return self.latest.get("resources", {})

    def get_value(self, path):
        """Return ANY value from the full live snapshot by dotted path (e.g. 'state.understanding',
        'arch.parts.cortex.synapses', 'resources.ram.pct', 'resources.storage.disk_free_gb'). No
        path → the queryable top-level keys. List indices allowed ('state.x.0')."""
        snap = self.snapshot()
        if not path:
            return {"keys": sorted(snap.keys()), "hint": "GET /api/value?key=state.understanding | resources.vram.used_gb | arch.total_synapses"}
        cur = snap
        for part in str(path).split("."):
            try:
                if isinstance(cur, (list, tuple)):
                    cur = cur[int(part)]
                elif isinstance(cur, dict):
                    if part not in cur:
                        return {"key": path, "error": "not found at '%s'" % part, "available": sorted(cur.keys())[:40]}
                    cur = cur[part]
                else:
                    return {"key": path, "error": "cannot descend into %s at '%s'" % (type(cur).__name__, part)}
            except Exception as e:
                return {"key": path, "error": str(e)[:80]}
        return {"key": path, "value": cur}

    def observations(self):
        if self.life is None:
            return {"observations": []}
        return {"observations": [{k: o[k] for k in ("i", "modality", "ts", "label")}
                                 for o in self.life.last_observations]}

    def observation_media(self, i):
        """Returns (bytes, content_type) for the reconstructed audio/image, or None."""
        if self.life is None:
            return None
        return self.life.observation_media(i)

    # ---- callbacks -------------------------------------------------- #
    def _on_log(self, line):
        self.logs.append(line)

    def _on_state(self, st):
        with self._lock:
            if "status" in st and "state" not in st:
                self.birth_status = st["status"]
            self.latest.update(st)
            if "state" in st:
                self.status = "awake" if st.get("awake") else "sleeping"
                x = st.get("lived_seconds", int(time.time()))
                wt = (st.get("net") or {}).get("weights") or {}      # cortex leading-indicator diagnostics live here
                for k, *_ in CHART_KEYS:
                    val = st[k] if k in st else wt.get(k)            # top-level field, else a nested weight diagnostic
                    if isinstance(val, (int, float)):
                        self.history[k].append([x, round(float(val), 4)])

    def _hp(self):
        """The ACTUAL running hyperparameters (read from the live brain, authoritative), so
        the board always shows exactly what it's running on."""
        L = self.life
        if L is None:
            return dict(self.config or {})
        return dict(core=L.core, device=str(L.dev), dtype=str(L.dtype).replace("torch.", ""),
                    threads=L.threads, resonate_k=L.resonate_k, budget=L.budget,
                    min_awake=L.min_awake, max_awake=L.max_awake, max_model_gb=L.max_model_gb,
                    teacher=L.use_teacher, visual=L.use_visual, resumed=L.resumed,
                    perceive_gap=L.perceive_gap, think_chunk=L.think_chunk, learn_steps=L.learn_steps,
                    feed_mode=L.feed_mode, focus=L.focus_label or "(random)",
                    topics=len(L.topics), teach_queue=L._teach_q.qsize(),
                    max_log_mb=L.max_log_mb, max_tb_mb=L.max_tb_mb,
                    grow_add=L.grow_add, grow_until=getattr(L.brain, "grow_until", None),
                    prune_until=getattr(L.brain, "prune_until", None),
                    freeze_growth=L.freeze_growth, freeze_sleep=L.freeze_sleep,
                    freeze_learning=L.freeze_learning, hard_disk_gb=round(L.memory.hard / 1e9, 2),
                    layers=L.layers_n if hasattr(L, "layers_n") else getattr(L.brain, "layers_n", None),
                    run=os.path.basename(L.resdir))

    def _trend(self, key):
        """Signed recent change of a metric from its own history (last value − value ~20 samples
        back). Lets an eyeless operator see direction (learning? growing?) in one call."""
        h = self.history.get(key)
        if not h or len(h) < 2:
            return None
        pts = list(h); a = pts[max(0, len(pts) - 20)][1]; b = pts[-1][1]
        return round(b - a, 4)

    def diag(self):
        """One-call HEALTH CHECK for driving-by-API: is it alive, is it learning (trends), are all
        five systems firing, is the architecture evolving — plus explicit WARNINGS for anything off.
        Built for an operator with no eyes on the dashboard."""
        arch = self.arch()
        with self._lock:
            l = dict(self.latest); st = {k: l.get(k) for k in l}
            net = l.get("net", {})
            trends = {k: self._trend(k) for k in ("bpb", "understanding", "synapses", "neurons",
                                                  "perplexity_train", "spike_rate")}
            warn = []
            sr = st.get("spike_rate")
            if sr is not None and sr <= 0.0: warn.append("cortex spike rate is 0 — dead/silent neurons")
            if sr is not None and sr >= 0.9: warn.append("cortex spike rate ~1 — no sparsity (over-firing)")
            pt = st.get("perplexity_train")
            if pt is not None and (pt != pt): warn.append("train perplexity is NaN")
            if st.get("bg_spike_rate") == 0.0: warn.append("§2 basal-ganglia not firing")
            hf = st.get("hippo_fidelity")
            if hf is not None and hf < 0.5: warn.append(f"§4 hippocampus recall fidelity low ({hf})")
            dens = st.get("synapse_density")
            if dens is not None and dens < 0.05: warn.append(f"synapse density very low ({dens}) — over-pruned")
            if trends.get("bpb") is not None and trends["bpb"] > 0.2: warn.append("bits/byte RISING — not learning / diverging")
            systems = {
                "cortex (§3)": {"spike_rate": st.get("spike_rate"), "perplexity_train": pt, "ok": bool(sr and 0 < sr < 0.9)},
                "cerebellum (§1)": {"mse": st.get("cerebellum_mse"), "ok": st.get("cerebellum_mse") is not None},
                "basal_ganglia (§2)": {"spike_rate": st.get("bg_spike_rate"), "policy_entropy": st.get("bg_policy_entropy"),
                                       "ok": st.get("bg_spike_rate") is not None},
                "hippocampus (§4)": {"spike_rate": st.get("hippo_spike_rate"), "fidelity": hf, "episodes": st.get("episodes"),
                                     "ok": hf is None or hf >= 0.5},
                "neuromod (§5)": {"da_tone": st.get("da_tone"), "ok": True},
            }
            return {
                "status": self.status, "running": self.running(),
                "alive": self.running(), "phase": st.get("phase"), "nights": st.get("nights"),
                "age": st.get("age"), "lived_seconds": st.get("lived_seconds"),
                "learning": {"bits_per_byte": st.get("bpb"), "understanding": st.get("understanding"),
                             "perplexity_train": pt, "perplexity_gen": st.get("perplexity_gen")},
                "trends_recent": trends,
                "architecture": {"total_neurons": arch.get("total_neurons"), "total_synapses": arch.get("total_synapses"),
                                 "cortex": (arch.get("parts", {}) or {}).get("cortex", {})},
                "systems": systems,
                "replay_total": st.get("replay_total"), "sleep_debt": st.get("debt"),
                "warnings": warn or ["none — all systems nominal"],
            }

    def snapshot(self):
        with self._lock:
            l = self.latest
            hist = {k: (list(self.history[k])[::max(1, len(self.history[k]) // 240)]) for k in self.history}
            return {
                "status": self.status, "running": self.running(),
                "birth_status": self.birth_status,
                "run_dir": self.run_dir, "config": self.config, "compute": self.compute,
                "hp": self._hp(),
                "tools": (self.life.tools.list() if self.life is not None else []),
                "net": l.get("net", {}), "netparams": l.get("netparams", {}),
                "arch": self.arch(), "resources": self.resources(),
                "observations": l.get("observations", []),
                "state": {k: l.get(k) for k in (
                    "awake", "cycle", "nights", "age", "phase", "eta", "granules", "model_gb",
                    "device", "dtype", "understanding", "bpb", "time_sense", "novelty", "da_tone",
                    "teacher_cps", "teacher_name", "think_bps", "learn_bps", "tps", "episodes",
                    "perceptions", "debt", "debt_threshold", "awake_seconds", "max_awake",
                    "sleep_remaining", "clock", "lived_seconds",
                    "neurons", "synapses", "synapse_density", "parameters",
                    "perplexity_train", "perplexity_gen", "gen_entropy", "spike_rate",
                    "cerebellum_mse", "bg_spike_rate", "hippo_spike_rate", "hippo_fidelity",
                    "bg_policy_entropy", "replay_total", "replay_mb", "lived_chars",
                    "ram_gb", "ram_pct", "vram_gb", "vram_pct", "storage_gb", "disk_free_gb")},
                "thought": l.get("thought", ""), "mind": l.get("mind", ""),
                "feed": l.get("feed", []),
                "perceived": l.get("perceived", ""), "ascii": l.get("ascii", ""),
                "parallel": l.get("parallel", []),
                "logs": list(self.logs)[-120:],
                "history": hist,
            }


def list_runs():
    runs = sorted(g for g in glob.glob(os.path.join(BASE, "run_*")) if os.path.isdir(g))
    return [os.path.basename(r) for r in runs][::-1]


def ssh_forward_hint(port):
    """If we're on an SSH host, the exact `ssh -L` command to view the board from a laptop."""
    conn = os.environ.get("SSH_CONNECTION", "").split()
    if len(conn) < 3:
        return None
    host, user = conn[2], os.environ.get("USER", "user")
    return f"ssh -L {port}:localhost:{port} {user}@{host}"


CTRL = Controller()

# ---------------------------------------------------------------- HTML ---- #
PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>the living brain</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b0f14;--pan:#131a22;--ln:#1e2a36;--tx:#cfe3f2;--dim:#7690a6;--acc:#4ec9b0;--warn:#e2c08d;--go:#7bd88f;--stop:#e06c75;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.45 ui-monospace,Menlo,Consolas,monospace}
header{display:flex;gap:10px;align-items:center;padding:8px 12px;background:var(--pan);border-bottom:1px solid var(--ln);flex-wrap:wrap}
.pill{padding:2px 9px;border-radius:12px;font-weight:700}
.awake{background:#20361f;color:var(--go)}.sleeping{background:#1b2740;color:#8fb2ff}.stopped{background:#3a1f22;color:var(--stop)}.being_born,.stopping,.idle{background:#33301c;color:var(--warn)}
button{background:#1c2530;color:var(--tx);border:1px solid var(--ln);border-radius:6px;padding:5px 10px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}.b-go{border-color:var(--go);color:var(--go)}.b-stop{border-color:var(--warn);color:var(--warn)}.b-kill{border-color:var(--stop);color:var(--stop)}
small{color:var(--dim)}.grow{flex:1}
#cfg{display:none;padding:10px 12px;background:var(--pan);border-bottom:1px solid var(--ln);gap:12px;flex-wrap:wrap;align-items:end}
#cfg.show{display:flex}label{display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--dim)}
select,input{background:#0d141b;color:var(--tx);border:1px solid var(--ln);border-radius:5px;padding:4px 6px;font:inherit}
main{display:grid;grid-template-columns:340px 1fr;gap:10px;padding:10px}
.card{background:var(--pan);border:1px solid var(--ln);border-radius:8px;padding:8px 10px}
.card h3{margin:0 0 6px;font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
#charts{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
.tile{background:#0d141b;border:1px solid var(--ln);border-radius:6px;padding:3px 5px 5px;width:250px;height:150px;min-width:150px;min-height:96px;resize:both;overflow:hidden;position:relative;display:flex;flex-direction:column}
.tile.dragging{opacity:.35;border-color:var(--acc)}
.tile.drop-l{border-left:2px solid var(--acc)}
.tile .th{display:flex;justify-content:space-between;align-items:center;font-size:10px;color:var(--dim);cursor:grab}
.tile .th:active{cursor:grabbing}
.tile .tfs{cursor:pointer;padding:0 2px}.tile .tfs:hover{color:var(--acc)}
.tile canvas{flex:1;width:100%;min-height:0;display:block}
.tile.full{position:fixed;inset:24px;width:auto!important;height:auto!important;z-index:950;box-shadow:0 0 0 9999px rgba(0,0,0,.55)}
.metric{display:flex;justify-content:space-between}.metric b{color:var(--acc)}
#think{height:190px;overflow:auto}
.card{position:relative}
.fs{position:absolute;top:5px;right:9px;cursor:pointer;color:var(--dim);font-size:15px;user-select:none;z-index:3}
.fs:hover{color:var(--acc)}
.card.full{position:fixed;inset:0;margin:0;z-index:900;border-radius:0;overflow:auto}
.card.full #think,.card.full #log,.card.full #vision,.card.full #perc,.card.full #stats,.card.full #hp{height:calc(100vh - 54px)!important;max-height:none}
.card.full .tile{width:46%;height:280px}
.card.full #vision{font-size:11px;line-height:11px}
.tline{padding:2px 0;border-bottom:1px solid #0f171e;display:flex;gap:8px}
.tline .ts{color:#5c7488;flex:0 0 auto}
.tline .tx{color:#dfeaf3;word-break:break-word}
.tline.refl .tx{color:var(--warn)}
#log{height:190px;overflow:auto;white-space:pre-wrap;font-size:12px;color:var(--dim)}
#vision{white-space:pre;font-size:7px;line-height:7px;height:150px;overflow:hidden;color:var(--acc)}
#perc{height:90px;overflow:auto;color:var(--dim);font-size:12px}
#chat{display:flex;gap:6px;margin-top:6px}#chat input{flex:1}
.row{display:flex;gap:12px;flex-wrap:wrap}
</style></head><body>
<header>
 <b style="color:var(--acc)">🧠 living brain</b>
 <span id=pill class=pill>…</span>
 <span id=sub><small>—</small></span>
 <span class=grow></span>
 <button class=b-go onclick="toggleCfg()">⚙ launch / config</button>
 <button onclick="act('save').then(()=>flash('checkpointed ✓'))">💾 save</button>
 <button class=b-stop onclick="act('stop')">⏸ stop (save)</button>
 <button class=b-kill onclick="act('kill')">⛔ kill</button>
</header>
<div id=cfg>
 <label>device<select id=c_device><option>auto</option><option>cpu</option><option>cuda</option><option>cuda:0</option><option>multi</option></select></label>
 <label>threads (CPU)<input id=c_threads type=number min=1 style=width:70px placeholder=auto></label>
 <label>parallel resonance k<input id=c_resk type=number min=1 value=4 style=width:70px></label>
 <label>neurons / layer (fixed)<input id=c_hidden type=number min=32 value=512 style=width:90px></label>
 <label>init synapse density<input id=c_syn type=number step=0.05 min=0.02 max=1 value=0.5 style=width:80px></label>
 <label>sparse connectome<select id=c_sparse><option value=auto>auto (large→sparse)</option><option value=1>force on</option><option value=0>force off</option></select></label>
 <label>rec fan-in (sparse)<input id=c_fanin type=number min=4 value=64 style=width:80px></label>
 <label>layers<input id=c_layers type=number min=1 value=2 style=width:60px></label>
 <label>checkpoint<select id=c_ckpt><option value="">— fresh run —</option></select></label>
 <label>teacher<select id=c_teacher><option value=0>wikitext only</option><option value=1>Qwen + wikitext</option></select></label>
 <label>web browsing<select id=c_visual><option value=1>on</option><option value=0>off</option></select></label>
 <label>budget $/lesson<input id=c_budget type=number step=0.01 value=0.10 style=width:80px></label>
 <label>min awake s<input id=c_min type=number value=120 style=width:80px></label>
 <label>max awake s<input id=c_max type=number value=600 style=width:80px></label>
 <label>perceive gap s<input id=c_pgap type=number step=0.5 value=6 style=width:80px></label>
 <label>think chunk<input id=c_think type=number value=20 style=width:70px></label>
 <label>learn steps<input id=c_lsteps type=number value=8 style=width:70px></label>
 <label>model cap GB<input id=c_mgb type=number step=0.5 value=14 style=width:80px></label>
 <label>log cap MB<input id=c_logmb type=number step=1 value=20 style=width:80px></label>
 <label>tb cap MB<input id=c_tbmb type=number step=1 value=60 style=width:80px></label>
 <label>replay cap GB<input id=c_diskgb type=number step=0.5 value=10 style=width:80px></label>
 <label>grow +neurons/night<input id=c_growadd type=number min=0 value=64 style=width:90px></label>
 <label>grow until age<input id=c_growuntil type=number min=0 value=8 style=width:80px></label>
 <label>prune until age<input id=c_pruneuntil type=number min=0 value=16 style=width:80px></label>
 <label style="flex-direction:row;gap:3px;color:var(--dim);align-items:center"><input type=checkbox id=c_fgrow> freeze growth</label>
 <label style="flex-direction:row;gap:3px;color:var(--dim);align-items:center"><input type=checkbox id=c_fsleep> freeze sleep (stay awake)</label>
 <label style="flex-direction:row;gap:3px;color:var(--dim);align-items:center"><input type=checkbox id=c_flearn> freeze learning (observe only)</label>
 <button class=b-go onclick="applyLive()">✓ apply live (no restart)</button>
 <button class=b-go onclick="launch()">▶ relaunch (device/ckpt)</button>
 <span id=cnote><small>live params apply instantly; device/checkpoint need relaunch</small></span>
</div>
<div id=birthbar style="display:none;padding:9px 14px;background:#33301c;color:var(--warn);border-bottom:1px solid var(--ln);font-weight:700"></div>
<main>
 <div>
  <div class=card><h3>status / config</h3><div id=hp></div></div>
  <div class=card style=margin-top:10px><h3>state</h3><div id=stats></div></div>
  <div class=card style=margin-top:10px><h3>vision (what it sees)</h3><div id=vision></div></div>
  <div class=card style=margin-top:10px><h3>perception</h3><div id=perc></div>
   <div id=chat><input id=msg placeholder="talk to it…  (browse <url>, ?time)" onkeydown="if(event.key=='Enter')send()"><button class=b-go onclick=send()>send</button></div>
  </div>
  <div class=card style=margin-top:10px><h3>🎓 teach &amp; steer (live)</h3>
   <div style="display:flex;flex-direction:column;gap:5px">
    <div style=display:flex;gap:4px><input id=t_text placeholder="teach raw text…" style=flex:1 onkeydown="if(event.key=='Enter')teachText()"><button class=b-go onclick=teachText()>teach</button></div>
    <div style=display:flex;gap:4px><select id=t_kind><option>topic</option><option>url</option><option>path</option></select><input id=t_src placeholder="wiki topic / url / /path/to/code" style=flex:1><button class=b-go onclick=teachSrc()>learn</button></div>
    <div style=display:flex;gap:4px><input id=f_topics placeholder="focus topics, comma-separated" style=flex:1><select id=f_mode><option>topics</option><option>mixed</option><option>urls</option><option>random</option></select><button class=b-go onclick=focusNow()>🎯 focus</button></div>
    <div style=display:flex;gap:4px;align-items:center><button onclick=resetFocus()>reset feed → random</button><small id=fnote class=grow></small></div>
   </div>
  </div>
  <div class=card style=margin-top:10px><h3>🔧 tools &amp; other AIs (live)</h3>
   <div id=toollist></div>
   <div style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
    <input id=tl_name placeholder="name (e.g. opencode)">
    <input id=tl_cmd placeholder="command with {input}  (e.g. opencode run {input})">
    <input id=tl_install placeholder="install command (optional, e.g. npm i -g opencode-ai)">
    <div style=display:flex;gap:8px;align-items:center>
     <select id=tl_kind><option>text</option><option>audio</option><option>image</option><option>bytes</option></select>
     <label style="flex-direction:row;gap:3px;color:var(--dim)"><input type=checkbox id=tl_shell> shell</label>
     <label style="flex-direction:row;gap:3px;color:var(--dim)"><input type=checkbox id=tl_auto> autonomous</label>
     <button class=b-go onclick=toolAdd() style=margin-left:auto>＋ register</button>
    </div>
   </div>
  </div>
  <div class=card style=margin-top:10px><h3>🔬 net diagnostics</h3><div id=netdiag></div></div>
  <div class=card style=margin-top:10px><h3>🧬 architecture — neurons · synapses · parameters (per region)</h3>
   <div id=archdiag></div>
   <div style=display:flex;gap:4px;margin-top:6px;flex-wrap:wrap>
    <select id=ar_target><option>cortex</option><option>cerebellum</option><option>bg</option><option>hippocampus</option><option value=all>ALL regions</option></select>
    <select id=ar_op><option value=grow_synapses>grow synapses</option><option value=prune_synapses>prune synapses</option><option value=set_synapses>set synapses →</option><option value=grow_neurons>grow neurons</option><option value=set_neurons>set neurons →</option><option value=refresh_synapses>refresh @ density</option></select>
    <input id=ar_amt placeholder="amount / frac / density" style=width:130px>
    <button class=b-go onclick=editArch()>apply</button>
   </div>
   <div style=display:flex;gap:4px;margin-top:4px;flex-wrap:wrap;align-items:center>
    <small>synapse grow rate/night</small>
    <select id=gr_target><option>cortex</option><option>cerebellum</option><option>bg</option><option>hippocampus</option><option value=all>ALL</option></select>
    <input id=gr_val type=number step=0.01 min=0 max=1 placeholder=0.15 style=width:70px>
    <button class=b-go onclick=setGrowRate()>set rate</button>
    <small>device</small>
    <select id=dv_target><option>cortex</option><option>cerebellum</option><option>bg</option><option>hippocampus</option><option value=all>ALL</option></select>
    <select id=dv_dev><option>cpu</option><option>cuda</option><option>cuda:0</option><option>cuda:1</option></select>
    <button class=b-go onclick=setDevice()>move</button>
   </div>
   <small>set synapses/neurons → grows or prunes to a TARGET (count, or a 0–1 density for synapses). refresh re-seeds at density. each region can live on its own device (tensors convert at the boundaries).</small>
  </div>
  <div class=card style=margin-top:10px><h3>💾 resources — usage vs limit</h3><div id=resources></div></div>
  <div class=card style=margin-top:10px><h3>🎚 tune a module (live)</h3>
   <div style=display:flex;gap:4px>
    <select id=nt_target><option>cortex</option><option>hippocampus</option><option>bg</option><option>neuromod</option><option>cerebellum</option><option>endocrine</option><option>dynamics</option><option>peptides</option><option>glia</option><option>stdp</option><option>stp</option><option>plateau</option><option>interneurons</option><option>laminar</option><option>ripple</option><option>theta</option><option>embodiment</option></select>
    <input id=nt_key placeholder="param (lr, beta, da…)" style=flex:1><input id=nt_val placeholder="value" style=width:70px>
    <button class=b-go onclick=setNet()>set</button>
   </div>
   <div id=netparams style=margin-top:6px></div>
  </div>
  <div class=card style=margin-top:10px><h3>🧠 faithfulness stack (live toggles — §15.16)</h3>
   <div id=faithbtns style=display:flex;flex-wrap:wrap;gap:4px></div>
   <small>each biological constraint is independent; its capability cost is on the fidelity curve. click to flip live.</small>
  </div>
  <div class=card style=margin-top:10px><h3>👁👂 observations — replay what it sensed</h3><div id=obs></div></div>
 </div>
 <div>
  <div class=card><h3>metrics over time</h3><div id=charts></div></div>
  <div class=row style=margin-top:10px>
   <div class=card style=flex:1;min-width:280px><h3>💭 stream of thought</h3><div id=think></div></div>
   <div class=card style=flex:1;min-width:280px><h3>📜 life log</h3><div id=log></div></div>
  </div>
 </div>
</main>
<script>
let hist={};
function toggleCfg(){document.getElementById('cfg').classList.toggle('show')}
async function act(a){if(a=='kill'&&!confirm('Force-kill the brain (no checkpoint)?'))return;await fetch('/api/'+a,{method:'POST'});}
async function launch(){const chk=id=>document.getElementById(id).checked;
 const cfg={device:v('c_device'),threads:v('c_threads')||null,resonate_k:+v('c_resk'),checkpoint:v('c_ckpt'),
   hidden:+v('c_hidden'),layers:+v('c_layers'),syn_density:+v('c_syn'),
   sparse:(v('c_sparse')=='auto'?null:v('c_sparse')=='1'),rec_fanin:+v('c_fanin'),in_fanin:+v('c_fanin'),
   teacher:v('c_teacher')==1,visual:v('c_visual')==1,budget:+v('c_budget'),min_awake:+v('c_min'),max_awake:+v('c_max'),
   max_model_gb:+v('c_mgb'),max_log_mb:+v('c_logmb'),max_tb_mb:+v('c_tbmb'),hard_disk_gb:+v('c_diskgb'),
   perceive_gap:+v('c_pgap'),think_chunk:+v('c_think'),learn_steps:+v('c_lsteps'),
   grow_add:+v('c_growadd'),grow_until:+v('c_growuntil'),prune_until:+v('c_pruneuntil'),
   freeze_growth:chk('c_fgrow'),freeze_sleep:chk('c_fsleep'),freeze_learning:chk('c_flearn')};
 await fetch('/api/start',{method:'POST',body:JSON.stringify(cfg)});
 document.getElementById('cfg').classList.remove('show');
}
async function send(){const m=document.getElementById('msg');if(!m.value)return;await fetch('/api/chat',{method:'POST',body:JSON.stringify({text:m.value})});m.value='';}
async function post(u,b){try{return await(await fetch(u,{method:'POST',body:JSON.stringify(b)})).json();}catch(e){return{ok:false,err:''+e};}}
function flash(m){const n=document.getElementById('fnote');if(n){const s=(''+m).slice(0,90);n.textContent=s;setTimeout(()=>{if(n.textContent==s)n.textContent='';},5000);}}
async function teachText(){const el=document.getElementById('t_text');if(!el.value.trim())return;flash('teaching…');const r=await post('/api/teach',{text:el.value,label:'manual'});flash('teach: '+JSON.stringify(r));el.value='';}
async function teachSrc(){const s=document.getElementById('t_src').value.trim();if(!s)return;const k=v('t_kind');const b={label:s};b[k]=s;flash('learning '+k+'…');const r=await post('/api/teach',b);flash(k+': '+JSON.stringify(r));document.getElementById('t_src').value='';}
async function focusNow(){const t=v('f_topics').split(',').map(x=>x.trim()).filter(x=>x);const r=await post('/api/focus',{topics:t.length?t:null,mode:v('f_mode'),label:t[0]||v('f_mode')});flash('focus → '+(r.mode||JSON.stringify(r)));}
async function resetFocus(){await post('/api/focus',{topics:[],urls:[],mode:'random',label:''});flash('feed → random');}
function renderTools(tools){const el=document.getElementById('toollist');if(!el)return;
 el.innerHTML=(tools&&tools.length)?tools.map(t=>{const n=esc(t.name);return '<div class=metric style=gap:6px><span>'+n+' <small>['+esc(t.kind)+(t.autonomous?' ·auto':'')+(t.enabled?'':' ·off')+']</small></span><span style=white-space:nowrap>'
  +'<button onclick="toolRun(\''+n+'\')">run</button> '
  +'<button onclick="toolTog(\''+n+'\',\'enabled\','+(!t.enabled)+')">'+(t.enabled?'off':'on')+'</button> '
  +'<button onclick="toolTog(\''+n+'\',\'autonomous\','+(!t.autonomous)+')">'+(t.autonomous?'manual':'auto')+'</button> '
  +'<button onclick="toolInstall(\''+n+'\')">install</button> '
  +'<button class=b-kill onclick="toolDel(\''+n+'\')">✕</button></span></div>';}).join(''):'<small>no tools yet — register one below (opencode, a local model, TTS, …)</small>';}
async function toolAdd(){const b={name:v('tl_name'),cmd:v('tl_cmd'),kind:v('tl_kind'),install:v('tl_install'),shell:document.getElementById('tl_shell').checked,autonomous:document.getElementById('tl_auto').checked};if(!b.name||!b.cmd){flash('need name + cmd');return;}const r=await post('/api/tools/add',b);flash('tool: '+(r.ok?('registered '+r.tool.name):JSON.stringify(r)));document.getElementById('tl_name').value='';document.getElementById('tl_cmd').value='';document.getElementById('tl_install').value='';}
async function toolRun(n){const inp=prompt('input to '+n+'  (blank = the brain\'s own thought):','');if(inp===null)return;flash('running '+n+'…');const r=await post('/api/tools/run',{name:n,input:inp||undefined});flash(n+' → '+(r.output?esc(r.output).slice(0,90):JSON.stringify(r)));}
async function toolTog(n,k,val){const b={name:n};b[k]=val;await post('/api/tools/toggle',b);}
async function toolInstall(n){flash('installing '+n+'…');const r=await post('/api/tools/install',{name:n});flash('install '+n+': '+(r.ok?'ok ✓':esc(JSON.stringify(r)).slice(0,90)));}
async function toolDel(n){if(confirm('remove tool '+n+'?'))await post('/api/tools/remove',{name:n});}
async function setNet(){const t=v('nt_target'),k=v('nt_key').trim(),raw=v('nt_val').trim();if(!k)return;const b={target:t};b[k]=isNaN(+raw)?raw:+raw;const r=await post('/api/net',b);flash('set_net '+t+': '+JSON.stringify(r.applied||r));}
async function editArch(){const t=v('ar_target'),op=v('ar_op'),raw=v('ar_amt').trim();const b={target:t,op:op};if(raw!==''){const key=(op=='refresh_synapses'||((op=='set_synapses')&&+raw<=1))?'density':'amount';b[key]=+raw;}const r=await post('/api/arch',b);flash('arch '+t+' '+op+': '+JSON.stringify(r.applied||r.err||r));}
async function setGrowRate(){const t=v('gr_target'),raw=v('gr_val').trim();if(raw==='')return;const r=await post('/api/net',{target:t,grow_syn_frac:+raw});flash('grow rate '+t+': '+JSON.stringify(r.applied||r.err||r));}
async function setDevice(){const t=v('dv_target'),d=v('dv_dev');const r=await post('/api/device',{target:t,device:d});flash('device '+t+'→'+d+': '+(r.ok?'ok ✓':esc(JSON.stringify(r.err||r))));}
function fmt(n){return (n==null)?'—':(n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':(''+n));}
function bar(label,used,limit,unit){const pct=(limit&&limit>0)?Math.min(100,100*used/limit):0;const col=pct>90?'var(--stop)':pct>70?'var(--warn)':'var(--go)';
 return '<div class=metric style=flex-direction:column;align-items:stretch;gap:2px><div style=display:flex;justify-content:space-between><span>'+label+'</span><b>'+(used==null?'—':used)+(unit||'')+' / '+(limit==null?'—':limit)+(unit||'')+(limit?' ('+Math.round(pct)+'%)':'')+'</b></div>'+
 '<div style=height:5px;background:var(--ln);border-radius:3px;overflow:hidden><div style="height:100%;width:'+pct+'%;background:'+col+'"></div></div></div>';}
function renderResources(r){const el=document.getElementById('resources');if(!el)return;if(!r||(!r.ram&&!r.storage)){el.innerHTML='<small>—</small>';return;}
 let h='';const ram=r.ram||{},vr=r.vram||{},st=r.storage||{};
 if(ram.used_gb!=null)h+=bar('RAM (box)',ram.used_gb,ram.limit_gb,' GB')+'<div class=metric><span><small>this process</small></span><small>'+(ram.process_gb!=null?ram.process_gb+' GB':'—')+'</small></div>';
 h+=bar('VRAM ('+(vr.device||'—')+')',vr.used_gb,vr.limit_gb,' GB');
 if(st.checkpoint_gb!=null)h+=bar('checkpoint',st.checkpoint_gb,st.checkpoint_limit_gb,' GB');
 if(st.replay_gb!=null)h+=bar('replay buffer',st.replay_gb,st.replay_limit_gb,' GB');
 if(st.log_mb!=null)h+=bar('log',st.log_mb,st.log_limit_mb,' MB');
 if(st.tb_mb!=null)h+=bar('tensorboard',st.tb_mb,st.tb_limit_mb,' MB');
 if(st.disk_total_gb!=null)h+=bar('disk',st.disk_total_gb-st.disk_free_gb,st.disk_total_gb,' GB');
 el.innerHTML=h;}
function renderArch(a){const el=document.getElementById('archdiag');if(!el)return;if(!a||!a.parts){el.innerHTML='<small>—</small>';return;}
 // synapses = ACTIVE (wired) connections; capacity = all wire-able SLOTS (active + silent). "% wired"
 // is active/capacity of the allocated fan-in slots — NOT matrix density (a fan-in-32 cortex over H=128k
 // is ~0.02% dense). params ≈ capacity because it counts every weight slot (incl. silent) + biases.
 let tcap=(a.total_synapse_capacity!=null?' / '+fmt(a.total_synapse_capacity):'');
 let h='<div class=metric><span><b>TOTAL</b></span><b>'+fmt(a.total_neurons)+' neurons · '+fmt(a.total_synapses)+tcap+' syn (active/slots) · '+fmt(a.total_parameters)+' params</b></div>';
 for(const[k,p]of Object.entries(a.parts)){if(p.err){h+='<div class=metric><span>'+k+'</span><small>'+esc(p.err)+'</small></div>';continue;}
  let extra=((p.density!=null&&p.synapse_capacity)?' · '+Math.round(p.density*100)+'% wired':'')+(p.grow_syn_frac!=null?' · grow '+p.grow_syn_frac:'')+(p.tone_channels!=null?' · '+p.tone_channels+' tone ch':'')+(p.device?' · '+p.device:'')+(p.layer_widths?' · L'+p.layer_widths.join('/'):'')+(p.stored!=null?' · '+p.stored+' stored':'');
  let syn=fmt(p.synapses)+(p.synapse_capacity!=null?' / '+fmt(p.synapse_capacity):'')+' syn';
  h+='<div class=metric><span>'+k+'<small>'+esc(extra)+'</small></span><b>'+fmt(p.neurons)+' n · '+syn+'</b></div>';}
 el.innerHTML=h;}
function renderNet(nd,np){const el=document.getElementById('netdiag');if(el){const rows=[['train perplexity ↓',nd.perplexity_train],['gen perplexity',nd.perplexity_gen],['gen entropy (bits)',nd.gen_entropy],['cortex spike rate',nd.spike_rate],['cerebellum MSE ↓',nd.cerebellum_mse],['BG spike rate',nd.bg_spike_rate],['hippo spike rate',nd.hippo_spike_rate],['hippo recall fidelity',nd.hippo_fidelity],['BG policy entropy',nd.bg_policy_entropy]];Object.entries(nd.weights||{}).forEach(e=>rows.push([e[0],e[1]]));el.innerHTML=rows.filter(r=>r[1]!=null).map(r=>'<div class=metric><span>'+r[0]+'</span><b>'+esc(''+r[1])+'</b></div>').join('')||'<small>computing…</small>';}
 const pe=document.getElementById('netparams');if(pe){let h='';Object.entries(np||{}).forEach(m=>{h+='<div style=color:var(--acc);margin-top:3px>'+m[0]+'</div>'+Object.entries(m[1]).map(kv=>'<div class=metric><span>&nbsp;&nbsp;'+kv[0]+'</span><b>'+esc(''+kv[1])+'</b></div>').join('');});pe.innerHTML=h;}
 renderFaith(np);}
function setFaith(k,v){post('/api/net',{target:'cortex',[k]:v}).then(r=>flash('faith '+k+'='+v+': '+esc(JSON.stringify(r.applied||r.err||r))));}
function renderFaith(np){const el=document.getElementById('faithbtns');if(!el||!np||!np.cortex)return;const c=np.cortex;let h='';
 h+='<button class=b-go onclick="setFaith(\'learn_rule\',\''+(c.learn_rule=='eprop'?'bptt':'eprop')+'\')">rule: '+esc(''+c.learn_rule)+'</button>';
 h+='<button class=b-go onclick="setFaith(\'feedback_mode\',\''+(c.feedback_mode=='learned'?'random':'learned')+'\')">feedback: '+esc(''+c.feedback_mode)+'</button>';
 ['two_compartment','diff_neuromod','dale','dendritic','bounded_synapses','homeostasis','btsp','stochastic','metabolic'].forEach(k=>{const on=!!c[k];
  h+='<button class=b-go style="opacity:'+(on?1:.45)+'" onclick="setFaith(\''+k+'\','+(on?'false':'true')+')">'+k.replace('_synapses','')+': '+(on?'ON':'off')+'</button>';});
 el.innerHTML=h;}
function renderObs(obs){const el=document.getElementById('obs');if(!el)return;el.innerHTML=(obs&&obs.length)?obs.slice().reverse().map(o=>{const url='/api/observe?i='+o.i;const m=o.modality=='audio'?'<audio controls src="'+url+'" style=height:26px;max-width:150px></audio>':o.modality=='image'?'<img src="'+url+'" style=height:40px;image-rendering:pixelated;border:1px solid #234>':'<small>'+o.modality+'</small>';return '<div class=metric style=align-items:center><span>'+o.ts+' <small>['+o.modality+'] '+esc(o.label||'')+'</small></span><span>'+m+'</span></div>';}).join(''):'<small>no non-text observations yet — register an audio/image tool and it appears here to replay</small>';}
async function applyLive(){const chk=id=>document.getElementById(id).checked;
 const cfg={budget:+v('c_budget'),min_awake:+v('c_min'),max_awake:+v('c_max'),resonate_k:+v('c_resk'),perceive_gap:+v('c_pgap'),think_chunk:+v('c_think'),learn_steps:+v('c_lsteps'),max_model_gb:+v('c_mgb'),max_log_mb:+v('c_logmb'),max_tb_mb:+v('c_tbmb'),hard_disk_gb:+v('c_diskgb'),grow_add:+v('c_growadd'),grow_until:+v('c_growuntil'),prune_until:+v('c_pruneuntil'),freeze_growth:chk('c_fgrow'),freeze_sleep:chk('c_fsleep'),freeze_learning:chk('c_flearn'),visual:v('c_visual')==1};
 if(v('c_threads'))cfg.threads=+v('c_threads');const r=await post('/api/set',cfg);document.getElementById('cnote').innerHTML='<small>applied: '+esc(JSON.stringify(r.applied||r))+'</small>';}
function v(id){return document.getElementById(id).value}
function esc(s){return (s||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
function fmtNum(v){v=+v;const a=Math.abs(v);if(a>=1e9)return(v/1e9).toFixed(1)+'G';if(a>=1e6)return(v/1e6).toFixed(1)+'M';if(a>=1e3)return(v/1e3).toFixed(1)+'k';if(a>=100)return v.toFixed(0);if(a>=1)return v.toFixed(2);return v.toFixed(3);}
function fmtTime(s){s=Math.max(0,+s);if(s>=3600)return Math.floor(s/3600)+'h'+Math.floor((s%3600)/60)+'m';if(s>=60)return Math.floor(s/60)+'m'+Math.round(s%60)+'s';return Math.round(s)+'s';}
function drawChart(cv,series,opts){
 opts=opts||{};const dpr=2,W=cv.width=Math.max(60,cv.clientWidth*dpr),H=cv.height=Math.max(60,cv.clientHeight*dpr);
 const ctx=cv.getContext('2d');ctx.clearRect(0,0,W,H);ctx.font=(9.5*dpr)+'px ui-monospace,monospace';
 const pL=44*dpr,pB=15*dpr,pT=14*dpr,pR=6*dpr;const u=opts.unit||'';
 if(!series||series.length<2){ctx.fillStyle='#456';ctx.fillText('… waiting for data',pL,H/2);return;}
 const xs=series.map(p=>p[0]),ys=series.map(p=>p[1]);
 let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 if(y1-y0<1e-9){const m=Math.abs(y0)*.1+1;y0-=m;y1+=m;}else{const pad=(y1-y0)*.08;y0-=pad;y1+=pad;}
 const px=v=>pL+((v-x0)/((x1-x0)||1))*(W-pL-pR), py=v=>H-pB-((v-y0)/((y1-y0)||1))*(H-pT-pB);
 // grid + y ticks (auto-scaled units)
 ctx.fillStyle='#5c7488';ctx.strokeStyle='#141d26';ctx.lineWidth=1;
 for(let i=0;i<=3;i++){const yv=y0+(y1-y0)*i/3,Y=py(yv);ctx.beginPath();ctx.moveTo(pL,Y);ctx.lineTo(W-pR,Y);ctx.stroke();ctx.fillText((fmtNum(yv)+u).slice(0,8),2*dpr,Y+3*dpr);}
 // x ticks (time since birth)
 for(let i=0;i<=2;i++){const xv=x0+(x1-x0)*i/2,X=px(xv);ctx.fillText(fmtTime(xv),Math.min(X,W-26*dpr),H-4*dpr);}
 // axes
 ctx.strokeStyle='#2a3947';ctx.beginPath();ctx.moveTo(pL,pT-4*dpr);ctx.lineTo(pL,H-pB);ctx.lineTo(W-pR,H-pB);ctx.stroke();
 // line
 ctx.strokeStyle='#4ec9b0';ctx.lineWidth=1.6*dpr;ctx.beginPath();series.forEach((p,i)=>{const X=px(p[0]),Y=py(p[1]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
 // latest value
 ctx.fillStyle='#8fdccb';ctx.font=(11*dpr)+'px ui-monospace,monospace';ctx.fillText(fmtNum(ys[ys.length-1])+u,pL+3*dpr,pT+2*dpr);
 cv._c={series,px,py,dpr,W,H,pT,pB,pL,opts,u};
}
function hoverChart(cv,e){const c=cv._c;if(!c)return;const r=cv.getBoundingClientRect();const mx=(e.clientX-r.left)*c.dpr;
 let best=null,bd=1e18;c.series.forEach(p=>{const d=Math.abs(c.px(p[0])-mx);if(d<bd){bd=d;best=p;}});if(!best)return;
 drawChart(cv,c.series,c.opts);const ctx=cv.getContext('2d'),X=c.px(best[0]),Y=c.py(best[1]);
 ctx.strokeStyle='#e2c08d';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(X,c.pT-4*c.dpr);ctx.lineTo(X,c.H-c.pB);ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#e2c08d';ctx.beginPath();ctx.arc(X,Y,3*c.dpr,0,7);ctx.fill();
 const tip=fmtTime(best[0])+' · '+fmtNum(best[1])+c.u;ctx.font=(10*c.dpr)+'px ui-monospace,monospace';
 const tw=ctx.measureText(tip).width+8*c.dpr;let tx=X+6*c.dpr;if(tx+tw>c.W)tx=X-tw-6*c.dpr;
 ctx.fillStyle='rgba(12,20,27,.96)';ctx.fillRect(tx,c.pT-4*c.dpr,tw,15*c.dpr);ctx.fillStyle='#e2c08d';ctx.fillText(tip,tx+4*c.dpr,c.pT+7*c.dpr);}
function loadLayout(){try{return JSON.parse(localStorage.getItem('brainCharts'))||{}}catch(e){return{}}}
function saveLayout(){const L={order:[...document.querySelectorAll('#charts .tile')].map(t=>t.dataset.m),size:{}};
 document.querySelectorAll('#charts .tile').forEach(t=>{if(t.style.width)L.size[t.dataset.m]={w:t.style.width,h:t.style.height};});
 try{localStorage.setItem('brainCharts',JSON.stringify(L));}catch(e){}}
function redrawTile(t){const m=t.dataset.m,cv=t.querySelector('canvas'),c=CHARTS.find(x=>x[0]==m);if(cv&&window._last)drawChart(cv,(window._last.history||{})[m],{unit:c[3],lower:c[2]});}
function buildCharts(){const cw=document.getElementById('charts');if(cw.dataset.built)return;cw.dataset.built=1;
 const L=loadLayout();let order=(L.order&&L.order.length)?L.order.filter(m=>CHARTS.find(c=>c[0]==m)):CHARTS.map(c=>c[0]);
 CHARTS.forEach(c=>{if(order.indexOf(c[0])<0)order.push(c[0]);});
 order.forEach(m=>{const c=CHARTS.find(x=>x[0]==m);if(!c)return;
  const t=document.createElement('div');t.className='tile';t.dataset.m=m;
  const sz=(L.size||{})[m];if(sz){t.style.width=sz.w;t.style.height=sz.h;}
  t.innerHTML='<div class=th draggable=true><span>'+c[1]+'</span><span class=tfs title=fullscreen>⛶</span></div><canvas></canvas>';
  cw.appendChild(t);
  t.querySelector('.tfs').onclick=e=>{e.stopPropagation();const f=t.classList.toggle('full');e.target.textContent=f?'✕':'⛶';setTimeout(()=>redrawTile(t),30);};
  const h=t.querySelector('.th');
  h.addEventListener('dragstart',()=>{t.classList.add('dragging');window._drag=t;});
  h.addEventListener('dragend',()=>{t.classList.remove('dragging');[...cw.children].forEach(x=>x.classList.remove('drop-l'));saveLayout();});
  const cv=t.querySelector('canvas');
  cv.addEventListener('mousemove',e=>hoverChart(cv,e));
  cv.addEventListener('mouseleave',()=>redrawTile(t));
  let rt;new ResizeObserver(()=>{redrawTile(t);clearTimeout(rt);rt=setTimeout(saveLayout,400);}).observe(t);
 });
 cw.addEventListener('dragover',e=>{e.preventDefault();const d=window._drag;if(!d)return;
  const els=[...cw.querySelectorAll('.tile:not(.dragging)')];let tgt=null,best=1e18;
  els.forEach(el=>{const r=el.getBoundingClientRect();const off=Math.hypot(e.clientX-(r.left+r.width/2),e.clientY-(r.top+r.height/2));if(off<best){best=off;tgt=el;}});
  els.forEach(x=>x.classList.remove('drop-l'));
  if(tgt){const r=tgt.getBoundingClientRect();if(e.clientX<r.left+r.width/2){tgt.classList.add('drop-l');cw.insertBefore(d,tgt);}else{cw.insertBefore(d,tgt.nextSibling);}}});
 cw.addEventListener('drop',e=>{e.preventDefault();saveLayout();});}
async function tick(){
 let d;try{d=await(await fetch('/api/state')).json();}catch(e){return;}
 const p=document.getElementById('pill');p.textContent=(d.running?d.status:'stopped').toUpperCase();p.className='pill '+(d.running?d.status:'stopped');
 const s=d.state||{},c=d.compute||{},hp=d.hp||{};
 document.getElementById('sub').innerHTML='<small>'+esc(d.run_dir||'—').split('/').slice(-1)[0]+' · '+esc(c.note||'')+(d.status=='being_born'?' · '+esc(d.birth_status):'')+'</small>';
 const bb=document.getElementById('birthbar');
 if(d.status=='being_born'||d.status=='resuming'){const sp='◐◓◑◒'[Math.floor(Date.now()/200)%4];bb.style.display='block';bb.textContent=sp+' '+(d.birth_status||'being born…');}
 else bb.style.display='none';
 const b=v=>v===true?'on':v===false?'off':(v??'—');
 const hrows=[['running',d.running?'▶ YES':'■ no'],['status',d.status],['run folder',hp.run],['device',(hp.device||'')+' / '+(hp.dtype||'')],
  ['compute',c.note],['CPU threads',hp.threads],['parallel resonance k',hp.resonate_k],['core',hp.core],
  ['🎯 focus',hp.focus],['feed mode',hp.feed_mode],['focus topics',hp.topics],['teach queue',hp.teach_queue],
  ['learn steps/lesson',hp.learn_steps],['perceive gap (s)',hp.perceive_gap],['think chunk',hp.think_chunk],
  ['Qwen teacher (birth)',b(hp.teacher)],['web browsing',b(hp.visual)],['budget $/lesson',hp.budget],
  ['min awake (s)',hp.min_awake],['max awake (s)',hp.max_awake],
  ['neurons (fixed)',fmt(s.neurons)],['synapses (active)',fmt(s.synapses)],['synapse density',s.synapse_density],['layers',hp.layers],['grow until age',hp.grow_until],['prune until age',hp.prune_until],
  ['freeze growth',b(hp.freeze_growth)],['freeze sleep',b(hp.freeze_sleep)],['freeze learning',b(hp.freeze_learning)],
  ['model/ckpt cap (GB)',hp.max_model_gb],['replay cap (GB)',hp.hard_disk_gb],
  ['log cap (MB)',hp.max_log_mb],['tb cap (MB)',hp.max_tb_mb],['resumed from ckpt',b(hp.resumed)]];
 document.getElementById('hp').innerHTML=hrows.map(r=>'<div class=metric><span>'+r[0]+'</span><b>'+esc(''+(r[1]??'—'))+'</b></div>').join('');
 const rows=[['device',(s.device||'')+' / '+(s.dtype||'')],['age',s.age+' ('+(s.phase||'')+')'],['neurons (fixed)',fmt(s.neurons)],['synapses',fmt(s.synapses)],['parameters',fmt(s.parameters)],['model',(s.model_gb||0)+' GB'],
  ['nights',s.nights],['thoughts',s.cycle],['thoughts/s',s.tps],['understanding',s.understanding],['bits/byte',s.bpb],['time-sense',s.time_sense],
  ['novelty',s.novelty],['dopamine',s.da_tone],['episodes',s.episodes],['perceptions',s.perceptions],
  ['teacher',(s.teacher_name||'—')+' '+(s.teacher_cps||0)+' c/s'],['think',(s.think_bps||0)+' b/s'],['learn',(s.learn_bps||0)+' b/s'],
  ['sleep debt',(s.debt||0)+'/'+(s.debt_threshold||0)],['clock',s.clock]];
 document.getElementById('stats').innerHTML=rows.map(r=>'<div class=metric><span>'+r[0]+'</span><b>'+esc(''+(r[1]??'—'))+'</b></div>').join('');
 const th=document.getElementById('think');const atBottom=th.scrollTop+th.clientHeight>=th.scrollHeight-30;
 th.innerHTML=(d.feed||[]).map(e=>{const r=(e[1]||'').startsWith('↻');return '<div class="tline'+(r?' refl':'')+'"><span class=ts>'+e[0]+'</span><span class=tx>'+esc(e[1])+'</span></div>';}).join('');
 if(atBottom)th.scrollTop=1e6;
 document.getElementById('log').textContent=(d.logs||[]).join('\n');document.getElementById('log').scrollTop=1e6;
 document.getElementById('vision').textContent=d.ascii||'';document.getElementById('perc').textContent=d.perceived||'';
 window._last=d; buildCharts();
 document.querySelectorAll('#charts .tile').forEach(t=>{if(!t.matches(':hover'))redrawTile(t);});
 renderTools(d.tools);
 renderNet(d.net||{},d.netparams||{});renderArch(d.arch||{});renderResources(d.resources||{});
 renderObs(d.observations);
 addFsButtons();
}
function addFsButtons(){document.querySelectorAll('main .card').forEach(c=>{if(c.querySelector('.fs'))return;
 const b=document.createElement('span');b.className='fs';b.textContent='⛶';b.title='fullscreen';
 b.onclick=()=>{const f=c.classList.toggle('full');b.textContent=f?'✕':'⛶';};c.appendChild(b);});}
const CHARTS=__CHARTS__;
async function loadRuns(){try{const r=await(await fetch('/api/runs')).json();document.getElementById('c_ckpt').innerHTML='<option value="">— fresh run —</option>'+r.map(x=>'<option>'+x+'</option>').join('');}catch(e){}}
loadRuns();setInterval(tick,1500);tick();
</script></body></html>"""
PAGE = PAGE.replace("__CHARTS__", json.dumps([[k, lbl, lb, u] for k, lbl, lb, u in CHART_KEYS]))


API_HELP = {
    "note": "Everything the board does is one of these calls — drive the brain fully by API.",
    "GET /api/state": "full live state: status, config (hp), metrics, history, thought feed, logs, vision. "
                      "state.* includes per-part diagnostics: spike_rate (cortex), bg_spike_rate, hippo_spike_rate, "
                      "hippo_fidelity, bg_policy_entropy, cerebellum_mse, perplexity_train/gen, gen_entropy, replay_total",
    "GET /api/runs": "list saved checkpoint folders",
    "GET /api/logs?n=N": "full-resolution recent log lines (deque up to 400; state gives only last 120)",
    "GET /api/history?key=K": "full-resolution time-series for metric K (omit key = every series). "
                              "keys: any CHART field incl neurons, synapses, synapse_density, bg_spike_rate, "
                              "hippo_spike_rate, hippo_fidelity, cerebellum_mse",
    "GET /api/arch": "per-region NEURON + SYNAPSE + PARAMETER census (cortex/cerebellum/bg/hippocampus/neuromod) + "
                     "global totals + per-layer widths + synapse density + per-region grow rate + device. "
                     "Neurons are the fixed population; synapses are what evolve.",
    "GET /api/diag": "ONE-CALL HEALTH CHECK: alive? learning (bpb/perplexity + recent TRENDS)? all five systems "
                     "firing? architecture totals? + explicit WARNINGS for anything off. For driving by API without eyes.",
    "GET /api/resources": "RAM / VRAM (per device) / storage — current usage AND limit AND pct for each: "
                          "ram{process_gb,used_gb,limit_gb,pct}, vram{used_gb,limit_gb,pct,device}, "
                          "storage{checkpoint,replay,log,tb vs caps + disk_free/total/pct}. Time-series via /api/history.",
    "GET /api/value?key=PATH": "fetch ANY value from the live snapshot by dotted path — e.g. state.understanding, "
                               "resources.vram.used_gb, arch.parts.cortex.synapses, resources.storage.disk_free_gb. "
                               "Omit key to list the top-level keys.",
    "GET /api/help": "this",
    "POST /api/chat": "{text} — talk to it / inject non-blocking feedback",
    "POST /api/teach": "{text | topic | url | path, label?} — teach specific content NOW "
                       "(raw text, a wiki topic, a web page, or a local file/dir of code/notation/prose); "
                       "learned with priority in the live loop",
    "POST /api/focus": "{topics:[...]?, urls:[...]?, mode: random|topics|urls|mixed, label?} — "
                       "redirect the learning feed to a target area (e.g. topics=['music theory'], mode='topics')",
    "POST /api/set": "change GLOBAL hyperparameters LIVE, no restart: "
                     "budget, min_awake, max_awake, debt_threshold, perceive_gap, think_chunk, "
                     "resonate_k, threads, learn_steps, visual, teacher, "
                     "grow_add, grow_until, prune_until, freeze_growth, freeze_sleep, freeze_learning, "
                     "max_model_gb (= checkpoint cap), hard_disk_gb (replay cap), max_log_mb, max_tb_mb, "
                     "sleep_mode (buffer|generative), gr_dreams, gr_dream_len, gr_temperature, gr_anchor_frac (§16 P0 replay), "
                     "neurogenesis, neurogenesis_add, neurogenesis_every (§16 adult-DG neurogenesis). "
                     "(initial neurons/layers/device/core need /api/start relaunch.)",
    "POST /api/net": "tune a PART (or 'all') of the net live: {target: cortex|cerebellum|hippocampus|bg|neuromod|endocrine|dynamics|peptides|glia|stdp|stp|plateau|interneurons|laminar|ripple|theta|embodiment|all, ...params} — "
                     "cortex {lr,read_alpha,seq,think_temp,prune_frac,grow_syn_frac}, cerebellum {eta,sparsity,g_golgi,thr0,grow_syn_frac,prune_frac}, "
                     "hippocampus {beta,sparsity,capacity,thr,g_inh,grow_syn_frac,prune_frac}, bg {alpha_v,alpha_pi,beta,thr,grow_syn_frac,prune_frac}, "
                     "neuromod {da,ach,ne,ht}, endocrine {on,alpha_D,tau_C,C_star,C_sigma,k_thr,k_pe,k_need,drive_met,novelty_met,lam_mood,al_thr} (§16 P1), "
                     "dynamics {on,beta0,kappa,ignite_thr,f_alpha,f_gamma} (§16 P2), "
                     "peptides {on,tau_p,k_op,k_cp,k_on,k_ov,k_ov2,k_cv,k_ct,k_oc,k_os,social_gain} (§16 slow neuropeptide layer), "
                     "glia {on,tau_a,k_p,k_g,k_m,rho_clear,target_rate} (§17 astrocyte field), "
                     "stdp {on,a_plus,a_minus,tau_plus,tau_minus,mix,w_ceiling} (§15.18 spike-timing plasticity; on also via cortex{stdp:true}), "
                     "stp {on,tau_rec,tau_facil,U,modulate_input} (§17 short-term synaptic plasticity), "
                     "plateau {on,p_thr,p_gain,rho_p,dur,btsp_couple} (§17 NMDA apical plateau; needs two_compartment), "
                     "interneurons {on,n_pv,n_som,n_vip,beta_i,thr_pv,thr_som,thr_vip,...} (§17 PV/SOM/VIP spiking pools; needs two_compartment), "
                     "laminar {on,frac_L4,frac_L23,frac_L56,allow_fb,input_to_l23,apical_l4_gain,strict} (§17 canonical L4/L2-3/L5-6 microcircuit), "
                     "ripple {on,f_so,dt,p0,up_thr,refractory,press_gain,debt_gain,debt_scale,rem_suppress,seed} (§16 SWR-gated consolidation), "
                     "theta {on,L,capacity,beta,g_inh,thr,fwd_frac,ripple_k,commit_on_boundary} (§17 hippocampal sequence memory), "
                     "embodiment {on,grid,max_steps,gamma,explore_temp,epistemic_w,step_cost,cadence} (§17 active-inference world loop). "
                     "Every region has its own live synapse grow rate (grow_syn_frac).",
    "POST /api/arch": "LIVE per-region OR global neuron/synapse surgery: {target: cortex|cerebellum|bg|hippocampus|all, op, amount?, density?}. "
                      "ops: grow_neurons, set_neurons (grow to a TARGET count), grow_synapses (amount<1 = fraction), "
                      "prune_synapses (amount = fraction), set_synapses (grow/prune to a TARGET count, or density 0–1), "
                      "refresh_synapses (re-seed at density). target 'all' fans the op across every region. "
                      "Neurons are the fixed population; synapses are what grow/prune over the life.",
    "POST /api/device": "place a region on its own device, live: {target: cortex|cerebellum|bg|hippocampus|neuromod|all, "
                        "device: cpu|cuda|cuda:N}. Tensors convert automatically at the cross-region boundaries.",
    "GET /api/observations": "list recent non-text sensory frames (audio/image) the brain observed",
    "GET /api/observe?i=N": "reconstruct observation N back into playable/viewable media "
                            "(audio/wav or image/png — a lossy rendering of exactly what it sensed)",
    "POST /api/start": "launch/relaunch (needed for device/core/checkpoint/initial-size changes): "
                       "{device: auto|cpu|cuda|cuda:N|multi, hidden (initial FIXED neuron population, can be large), "
                       "syn_density (0-1, initial fraction of synapses active; childhood grows it), layers, threads, "
                       "resonate_k, checkpoint, teacher, visual, budget, min_awake, max_awake, core, "
                       "max_model_gb, max_log_mb, max_tb_mb} (+ any /api/set soft key, applied on fresh launch)",
    "POST /api/save": "force a checkpoint now (weights + full live config) without stopping",
    "POST /api/stop": "graceful stop (checkpoints, then exits the brain)",
    "POST /api/kill": "force-stop the brain (no checkpoint)",
    "GET /api/tools": "list registered tools / other AIs",
    "POST /api/tools/add": "{name, cmd (with {input}), kind: text|audio|image|bytes, install?, shell?, "
                           "autonomous?, timeout?} — register a CLI tool / other AI the brain can talk to; "
                           "its OUTPUT is folded into the byte stream by modality",
    "POST /api/tools/install": "{name} — run the tool's install command",
    "POST /api/tools/run": "{name, input?, learn?} — invoke it NOW (input defaults to the brain's thought); "
                           "returns the output and (learn=true) folds it into the learning stream",
    "POST /api/tools/toggle": "{name, enabled?, autonomous?} — enable/disable, or let the brain converse with it on its own",
    "POST /api/tools/remove": "{name}",
    "examples": [
        "curl -XPOST localhost:8181/api/focus -d '{\"topics\":[\"Python (programming language)\",\"algorithms\"],\"mode\":\"topics\",\"label\":\"coding\"}'",
        "curl -XPOST localhost:8181/api/teach -d '{\"path\":\"/path/to/sapience/brain\",\"label\":\"its own code\"}'",
        "curl -XPOST localhost:8181/api/set -d '{\"budget\":0.2,\"resonate_k\":8,\"perceive_gap\":3}'",
        "# register another AI (opencode with a free model) and let the brain talk to it:",
        "curl -XPOST localhost:8181/api/tools/add -d '{\"name\":\"opencode\",\"cmd\":\"opencode run {input}\",\"kind\":\"text\",\"install\":\"npm i -g opencode-ai\",\"autonomous\":true}'",
        "curl -XPOST localhost:8181/api/tools/run -d '{\"name\":\"opencode\",\"input\":\"explain a for loop simply\"}'",
        "# a text-to-speech tool whose audio output is encoded as cochlea bytes:",
        "curl -XPOST localhost:8181/api/tools/add -d '{\"name\":\"say\",\"cmd\":\"tts --text {input} --out /tmp/o.wav && echo /tmp/o.wav\",\"kind\":\"audio\",\"shell\":true}'",
    ],
}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path.startswith("/api/state"):
            self._send(200, json.dumps(CTRL.snapshot()))
        elif self.path.startswith("/api/runs"):
            self._send(200, json.dumps(list_runs()))
        elif self.path.startswith("/api/tools"):
            self._send(200, json.dumps(CTRL.tools_list()))
        elif self.path.startswith("/api/observations"):
            self._send(200, json.dumps(CTRL.observations()))
        elif self.path.startswith("/api/observe"):
            from urllib.parse import urlparse, parse_qs
            i = parse_qs(urlparse(self.path).query).get("i", ["0"])[0]
            media = CTRL.observation_media(i)
            if media:
                self._send(200, media[0], media[1])
            else:
                self._send(404, "{}")
        elif self.path.startswith("/api/logs"):
            from urllib.parse import urlparse, parse_qs
            n = int(parse_qs(urlparse(self.path).query).get("n", ["400"])[0])
            lg = list(CTRL.logs)[-max(1, n):]
            self._send(200, json.dumps({"logs": lg, "n": len(lg), "capacity": CTRL.logs.maxlen}))
        elif self.path.startswith("/api/history"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query); key = q.get("key", [None])[0]
            with CTRL._lock:
                if key and key in CTRL.history:
                    out = {key: list(CTRL.history[key])}
                elif key:
                    out = {"err": f"unknown key '{key}'", "keys": list(CTRL.history)}
                else:
                    out = {k: list(v) for k, v in CTRL.history.items()}   # full-resolution, every series
            self._send(200, json.dumps(out))
        elif self.path.startswith("/api/arch"):
            self._send(200, json.dumps(CTRL.arch()))
        elif self.path.startswith("/api/resources"):
            self._send(200, json.dumps(CTRL.resources()))
        elif self.path.startswith("/api/value") or self.path.startswith("/api/get"):
            from urllib.parse import urlparse, parse_qs
            key = parse_qs(urlparse(self.path).query).get("key", [None])[0]
            self._send(200, json.dumps(CTRL.get_value(key)))
        elif self.path.startswith("/api/diag"):
            self._send(200, json.dumps(CTRL.diag()))
        elif self.path.startswith("/api/help") or self.path.startswith("/api/docs"):
            self._send(200, json.dumps(API_HELP, indent=2))
        else:
            self._send(404, "{}")

    def do_POST(self):
        b = self._body()
        # longest-prefix routes first (so /api/tools/add beats /api/tools)
        route = [("/api/tools/add", lambda: CTRL.tools_add(b)), ("/api/tools/remove", lambda: CTRL.tools_remove(b)),
                 ("/api/tools/toggle", lambda: CTRL.tools_toggle(b)), ("/api/tools/install", lambda: CTRL.tools_install(b)),
                 ("/api/tools/run", lambda: CTRL.tools_run(b)),
                 ("/api/start", lambda: CTRL.start(b)), ("/api/stop", lambda: CTRL.stop()),
                 ("/api/save", lambda: CTRL.save()),
                 ("/api/kill", lambda: CTRL.kill()), ("/api/chat", lambda: CTRL.chat(b.get("text", ""))),
                 ("/api/set", lambda: CTRL.set_params(b)), ("/api/teach", lambda: CTRL.teach(b)),
                 ("/api/focus", lambda: CTRL.focus(b)), ("/api/net", lambda: CTRL.set_net(b)),
                 ("/api/arch", lambda: CTRL.edit_arch(b)), ("/api/device", lambda: CTRL.set_device(b))]
        for p, fn in route:
            if self.path.startswith(p):
                self._send(200, json.dumps(fn())); return
        self._send(404, "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address. DEFAULT 127.0.0.1 (localhost only — the safe choice; the API can run "
                         "shell tools and read local paths, so it must NOT be world-reachable). View a remote "
                         "board over the printed `ssh -L` tunnel. Pass --host 0.0.0.0 ONLY on a trusted/isolated "
                         "network to expose it directly.")
    ap.add_argument("--checkpoint", default=None, help="resume this run folder on autostart")
    ap.add_argument("--no-autostart", action="store_true", help="start the board only; launch from the web")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | multi")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--resonate-k", type=int, default=4)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--min-awake", type=float, default=120.0)
    ap.add_argument("--max-awake", type=float, default=600.0)
    ap.add_argument("--stop", nargs="?", const="__latest__", default=None,
                    help="graceful-stop a running brain (folder, or newest) and exit")
    ap.add_argument("--kill", action="store_true", help="force-kill the running dashboard process and exit")
    args = ap.parse_args()
    os.makedirs(BASE, exist_ok=True)

    # ---- pure CLI control actions (no server) ----
    if args.stop is not None:
        from brain.life import request_stop, latest_run
        d = latest_run(BASE) if args.stop == "__latest__" else (args.stop if os.path.isabs(args.stop) else os.path.join(BASE, args.stop))
        if not d:
            print("no run to stop"); return
        request_stop(d); print(f"graceful stop requested → {d} (it will checkpoint and exit)"); return
    if args.kill:
        pidf = os.path.join(BASE, "dashboard.pid")
        if os.path.exists(pidf):
            pid = int(open(pidf).read().strip())
            try:
                os.kill(pid, signal.SIGKILL); print(f"killed dashboard pid {pid}")
            except ProcessLookupError:
                print("dashboard not running")
        else:
            print("no dashboard.pid found")
        return

    with open(os.path.join(BASE, "dashboard.pid"), "w") as f:
        f.write(str(os.getpid()))
    srv = ThreadingHTTPServer((args.host, args.port), H)
    url = f"http://localhost:{args.port}"
    fwd = ssh_forward_hint(args.port)
    print("=" * 74)
    print(f"  🧠  LIVING-BRAIN DASHBOARD  →  {url}   (bound {args.host})")
    if args.host != "127.0.0.1":
        print(f"  ⚠  EXPOSED on {args.host}: the API runs shell tools + reads local paths and has NO auth — "
              f"only do this on a trusted, isolated network. The safe path is the ssh -L tunnel below.")
    if fwd:
        print(f"  on a REMOTE/SSH box — run this on your LAPTOP, then open {url} :")
        print(f"      {fwd}")
    print(f"  the board: live metrics, thoughts, log, chat, status/config, launch/stop/kill")
    print(f"  CLI graceful stop:  python interface/dashboard.py --stop      force kill: --kill")
    print("=" * 74, flush=True)

    def shutdown(*_):
        try: CTRL.stop(timeout=30)
        except Exception: pass
        srv.shutdown()
    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), sys.exit(0)))

    if not args.no_autostart:
        CTRL.start({"device": args.device, "threads": args.threads, "resonate_k": args.resonate_k,
                    "teacher": not args.no_teacher, "visual": not args.no_visual, "budget": args.budget,
                    "min_awake": args.min_awake, "max_awake": args.max_awake,
                    "checkpoint": args.checkpoint})
    srv.serve_forever()


if __name__ == "__main__":
    main()
