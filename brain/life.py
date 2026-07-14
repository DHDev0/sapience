"""
life.py — BrainLife: the brain's continuous, always-thinking life.

The brain is NEVER idle. It runs a continuous inner monologue (a stream of thought),
learning from its own thoughts AND from the world; a background sense-thread feeds it
Claude Sonnet 5's teaching and web pixels without ever pausing the thinking; your
input is non-blocking FEEDBACK woven into the same stream; stop typing and it just
keeps learning on its own. It sleeps when tired (it decides), develops (grows/prunes)
each night, tells time, and can be stopped and RESUMED from a checkpoint.

  THINK   generate the next fragment of thought → learn from it (self-rehearsal) → stream it
  SENSE   (background) babble→Claude teaches→learn + visually browse→learn   (never blocks thinking)
  FEEL    your typed feedback is attached to the mind-stream, non-blocking
  SLEEP   when sleep-debt is high (within min/max awake), replay+consolidate the buffer
  DEVELOP each night: grow (child) / prune (adolescent) / stabilize (adult), η(t) falls

Everything — sight, language, time, its own voice, your words — is one byte-code stream.
Memory is tiered: model params in VRAM/RAM (grows to a max_model_gb cap), a hot working
buffer in RAM, the whole-life episode history on SSD. Runs on CPU or GPU (auto), fp32 on
CPU / bf16 on GPU.
"""
from __future__ import annotations
import os, json, time, threading, queue
from collections import deque

import torch
from .rnn_brain import ByteRNNBrain
from .memory import EpisodicMemory
from .spiking_modules import SpikingCerebellum, SpikingBasalGanglia, SpikingHippocampus, SpikingNeuromod
from .tools import ToolRegistry
from . import senses, motor, partner
from .ascii_art import image_to_ascii

TOPICS = ["the ocean", "how stars form", "memory and the brain", "why it rains",
          "how computers work", "volcanoes", "the history of writing", "how plants grow",
          "electricity", "the moon", "music and sound", "how languages change"]
BROWSE = ["https://en.wikipedia.org/wiki/Special:Random",
          "https://en.wikipedia.org/wiki/Portal:Current_events",
          "https://news.ycombinator.com/",
          "https://en.wikipedia.org/wiki/Special:Random"]

# a fixed, simple-English yardstick the brain is never trained on directly — a stable
# generalisation probe so bits/byte trends mean "getting better", not "memorised the probe".
HELDOUT_PROBE = ("Water flows downhill and gathers into rivers that run to the sea. The sun "
                 "warms the air, wind carries the clouds, and rain falls back onto the land. "
                 "People build homes, share stories, ask questions, and remember what they learn. "
                 "Night follows day, seasons turn, and living things grow, rest, and change.")


def _usable_lesson(text):
    """A Claude reply is worth distilling only if it is real prose, not a CLI error or a
    stub. Filters budget/timeout/rate-limit errors and too-short replies so the brain never
    learns 'Error: Exceeded USD budget' as if it were language."""
    if not text:
        return False
    t = text.strip()
    if len(t) < 40:                                  # a real 4-8 sentence lesson is long
        return False
    low = t.lower()
    bad = ("error:", "exceeded", "usd budget", "rate limit", "overloaded",
           "invalid api key", "not found", "usage:", "unknown option")
    return not any(low.startswith(b) or (b in low[:80]) for b in bad)


def pick_device(pref="auto"):
    """CPU/GPU selection + precision: fp32 on CPU, bf16 on GPU."""
    avail = torch.cuda.is_available()
    if pref == "cpu" or (pref == "auto" and not avail):
        return torch.device("cpu"), torch.float32
    if avail:
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32       # requested cuda but none → fall back


def resolve_compute(device="auto", threads=None):
    """Pick the config that MAXIMISES compute for the setup, so the dashboard/CLI 'auto' just
    works. Returns {device, dtype, threads, gpus, multi, note}. 'auto' → every GPU if present
    (multi-GPU shard when >1), else one GPU (bf16), else all CPU cores."""
    ncpu = os.cpu_count() or 4
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    d = device
    if device in ("auto", "multi"):
        d = "cuda" if ngpu >= 1 else "cpu"
    if d.startswith("cuda") and ngpu == 0:
        d = "cpu"                                        # asked for GPU, none present → CPU
    multi = (device in ("auto", "multi", "cuda") and ngpu > 1)
    dtype = "bf16" if d.startswith("cuda") else "fp32"
    th = int(threads) if threads else ncpu
    # honest note: the model currently runs on ONE device; multi-GPU FSDP2 sharding is the
    # documented scale path but is not yet wired, so >1 GPU still uses cuda:0.
    note = (f"{ngpu} GPUs present · using cuda:0 · bf16 (FSDP2 shard not yet wired)" if multi else
            "1 GPU · bf16" if d.startswith("cuda") else f"CPU · {th} threads · fp32")
    return dict(device=d, dtype=dtype, threads=th, gpus=ngpu, multi=multi, note=note)


def request_stop(resdir):
    """Create the STOP file so a life running in ANOTHER process exits gracefully (it
    finishes the current step, checkpoints, and shuts down)."""
    os.makedirs(resdir, exist_ok=True)
    open(os.path.join(resdir, "STOP"), "w").close()


def latest_run(base):
    """The newest results_life/run_* folder — used by `--stop` with no explicit folder."""
    import glob
    runs = sorted(g for g in glob.glob(os.path.join(base, "run_*")) if os.path.isdir(g))
    return runs[-1] if runs else None


class BrainLife:
    def __init__(self, resdir, core="spiking", emb=128, hidden=512, layers=2, granule=4000,
                 budget=0.30, use_teacher=True, use_visual=True,
                 min_awake=90.0, max_awake=300.0, debt_threshold=6.0,
                 device="auto", dtype="auto", max_model_gb=14.0,
                 max_ram_mb=64.0, hard_disk_gb=10.0, threads=None, resonate_k=4,
                 max_log_mb=20.0, max_tb_mb=60.0, syn_density=0.5,
                 sparse=None, sparse_hidden_threshold=8192, rec_fanin=64, in_fanin=64,
                 learn_rule="bptt", eprop_lr_scale=15.0,
                 think_chunk=20, perceive_gap=6.0, resume=False, use_tb=True, seed=0):
        os.makedirs(resdir, exist_ok=True)
        self.resdir = resdir
        self.ckpt = os.path.join(resdir, "brain.pt")
        self.logpath = os.path.join(resdir, "life.log")

        self.dev, self.dtype = pick_device(device)
        if dtype in ("fp32", "float32"): self.dtype = torch.float32
        if dtype in ("bf16", "bfloat16"): self.dtype = torch.bfloat16
        # CPU defaults to fp32 (bf16 matmul can be slow without AVX512-BF16), but honour an
        # EXPLICIT bf16 request on CPU (mixed precision via autocast; weights stay fp32).
        if self.dev.type == "cpu" and dtype not in ("bf16", "bfloat16"):
            self.dtype = torch.float32
        # CPU parallelism: threads=None → use ALL cores (auto-maximise compute)
        self.threads = int(threads) if threads else (os.cpu_count() or 4)
        if self.dev.type == "cpu": torch.set_num_threads(self.threads)
        self.resonate_k = max(1, int(resonate_k))          # width of parallel thought resonance

        # the cortex core. "spiking" = faithful growable spiking §3 cortex (LIF, §3.5
        # surrogate-BPTT, the default); "rnn" = rate byte-GRU (fluent, not the paper's
        # architecture) — the capability reference.
        if core == "rnn":
            self.brain = ByteRNNBrain(self.dev, dtype=self.dtype, emb=emb, hidden=hidden,
                                      layers=layers, max_model_gb=max_model_gb, seed=seed)
        else:
            core = "spiking"
            from .spiking_brain import SpikingBrain
            self.brain = SpikingBrain(self.dev, dtype=self.dtype, emb=emb, hidden=hidden,
                                      layers=layers, max_model_gb=max_model_gb, seed=seed,
                                      syn_density=syn_density, sparse=sparse,
                                      sparse_hidden_threshold=sparse_hidden_threshold,
                                      rec_fanin=rec_fanin, in_fanin=in_fanin)
            self.brain.learn_rule = learn_rule                 # "eprop" = biologically faithful
            self.brain.eprop_lr_scale = float(eprop_lr_scale)
        self.core = core
        self.max_model_gb = max_model_gb
        # The four OTHER spiking systems (§1,§2,§4,§5), coupled to the cortex by ACTIVATION and
        # REWARD only — never gradients (§6 gradient cut). They give the living loop a real
        # 5-system architecture: §1 cerebellum runs a fast supervised next-byte forward model,
        # §2 drives curiosity, §4 indexes episodes for novelty-gated replay (CLS), §5 sets tone.
        self.modules_on = (core == "spiking")
        if self.modules_on:
            # §1 cerebellum scales WITH the cortex (biologically it holds the most neurons — a huge
            # sparse granule layer). §2/§4 are task-bound decision/index systems (kept modest). ALL
            # four are sparse (masked, fixed-neuron/growing-synapse; the cortex additionally uses a
            # CSR store because it is the only O(neurons²) system — the others are neurons×const).
            n_gran = max(1500, int(hidden))
            self.nm = SpikingNeuromod((1,), self.dev)                                  # §5 (tone only)
            self.hippo = SpikingHippocampus(256, self.dev, seed=seed, syn_density=syn_density)  # §4
            self.bg = SpikingBasalGanglia(len(TOPICS), len(TOPICS), self.dev,
                                          alpha_v=0.1, alpha_pi=0.3, seed=seed, syn_density=syn_density)  # §2
            self.cerebellum = SpikingCerebellum(256, 256, self.dev, n_granule=n_gran, seed=seed,
                                                syn_density=syn_density)               # §1
            self.cereb_mse = 0.0                                                        # its live error
            self.cereb_eta = 0.3                                                        # delta-rule rate
            self._topic_feat = torch.eye(len(TOPICS), device=self.dev)
            self._last_topic = 0
            self._novelty = 1.0
            self.novelty_gate = 0.15                    # §4: min novelty to WRITE an episode (live-tunable)
        # tiered episodic memory of experienced TEXT (RAM hot + SSD compressed + eviction)
        self.memory = EpisodicMemory(os.path.join(resdir, "memory"),
                                     hot_mb=max_ram_mb, hard_gb=hard_disk_gb)
        self.clock = senses.Clock()
        self.budget = budget
        self.use_teacher, self.use_visual = use_teacher, use_visual
        self.perceive_gap = perceive_gap
        self.think_chunk = think_chunk

        # wake/sleep (§7.6 debt, §8-9); the brain decides, within min/max windows
        self.awake = True
        self.min_awake, self.max_awake = float(min_awake), float(max_awake)
        self.debt_threshold = float(debt_threshold)
        self.max_sleep = 900.0                 # cap on a single consolidation (s), live-tunable
        self.wake_start = time.time()
        self.debt = 0.0
        self.sleep_remaining = 0
        self.slept_count = 0
        self.cycle = 0
        # development / cycle control (all live-modifiable via the API)
        self.grow_add = 64                     # neurons (+ their synapses) added per grow cycle
        self.freeze_growth = False             # pause §10 synaptogenesis (fix the neuron count)
        self.freeze_sleep = False              # pause the wake/sleep cycle (stay awake)
        self.freeze_learning = False           # pause all weight updates (observe only)

        # continuous mind: the stream of consciousness (bytes) + I/O queues
        self.mind = deque(maxlen=4096)
        self.thought = ""                      # latest decoded thought fragment (for UI)
        self.parallel_thoughts = []            # last batch of parallel resonance streams
        self.thought_log = deque(maxlen=140)   # timestamped, line-coalesced thoughts (for the UI feed)
        self._line_buf = ""                    # accumulates fragments into readable lines
        # LIVE-STEERABLE learning: where the feed points + directed lessons injected via the API
        self.topics = list(TOPICS)             # what curiosity asks Claude about (mutable)
        self.browse = list(BROWSE)             # what it browses (mutable)
        self.feed_mode = "random"              # random | topics | urls | mixed — steer the feed
        self.focus_label = ""                  # human label of the current focus (e.g. "coding")
        self.learn_steps = 8                   # BPTT steps per lesson (live-tunable intensity)
        self._teach_q = queue.Queue()          # directed lessons (API) — learned with priority
        self.tools = ToolRegistry(os.path.join(resdir, "tools.json"))   # CLI tools / other AIs
        self.last_observations = deque(maxlen=12)   # recent non-text sensory frames (replayable)
        self._obs_i = 0

        # throughput meters (EMA-smoothed) for the UI + tensorboard
        self.teacher_cps = 0.0                 # teacher gen speed: Qwen at birth / Claude in life (chars/s)
        self.teacher_name = "—"                # which teacher produced the last measured speed
        self.think_bps = 0.0                   # main model thinking speed (bytes/s)
        self.learn_bps = 0.0                   # main model learning speed (bytes/s)
        self.use_tb = use_tb
        self.max_log_mb = float(max_log_mb)    # cap on life.log (evict earliest when full)
        self.max_tb_mb = float(max_tb_mb)      # cap on tensorboard event dir
        self._last_bound = 0.0
        self._tb = None                        # tensorboard SummaryWriter (lazy)
        self._tb_dir = os.path.join(resdir, "tb")
        self._t_start = time.time()
        self._last_tb = 0.0                    # fast-scalar throttle
        self._last_tb_slow = 0.0               # expensive-scalar throttle (gen quality, weight health)
        self._perc_count = 0                   # cumulative perceptions consumed
        self._awake_frac = 1.0                 # EMA of awake duty cycle
        self._tps = 0.0                        # thoughts/sec (EMA)
        self._last_cyc, self._last_cyc_t = 0, time.time()
        self._perc_q = queue.Queue()           # perceptions from the background sense-thread
        self._feedback = deque()               # your non-blocking feedback
        self._running = False
        self._perc_thread = None
        self._browser = None
        self._browser_dead = False
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._last_ckpt = time.time()

        self.log_lines = deque(maxlen=400)
        self.log_cb = None
        self.state = {}
        self.last_screenshot = None
        self.probe = (partner.web_topic("Human brain")[:4000] or
                      "the brain lets us think, feel, remember and learn.")
        self._time_probe = self._make_time_probe()
        self.resumed = bool(resume)
        if resume:
            self.load_life()

    # ---- logging ----------------------------------------------------- #
    def log(self, msg):
        line = f"[{self.clock.tell()}] {msg}"
        self.log_lines.append(line)
        try:
            with open(self.logpath, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if self.log_cb:
            try: self.log_cb(line)
            except Exception: pass

    def _make_time_probe(self):
        c = senses.Clock(t0=0.0); s = []
        for k in range(200):
            c.tick(now=float(k)); s += senses.frame("time", c.stamp(now=float(k)))
        return bytes(s).decode("latin1")

    # ---- learning helpers -------------------------------------------- #
    def _learn_text(self, text, steps=6):
        """Core learning path: backprop-learn perceived TEXT (§3.5 β→0 PC), advance the
        mind state, and archive to tiered episodic memory (RAM hot + SSD compressed)."""
        if not text:
            return 0.0
        self.clock.tick()
        progress = 0.0
        _t0 = time.time()
        if self.freeze_learning:                        # observe only — no weight updates
            self.brain.observe_stream(text)
        else:
            # §5 three-factor gating: the ACh neuromodulator tone M(t) scales the effective
            # plasticity (Δw = η·M(t)·e) — high in wake (encode), lower in NREM (consolidate). Under
            # e-prop this M is the LITERAL three-factor gate on the eligibility update; under BPTT it
            # falls back to scaling the optimiser lr.
            gate = self.nm.tone["ach"] if self.modules_on else 1.0
            eprop = getattr(self.brain, "learn_rule", "bptt") == "eprop"
            if gate != 1.0 and not eprop:
                for g in self.brain.opt.param_groups: g["lr"] = self.brain.lr * gate
            r = self.brain.learn_text(text, epochs=1, max_steps=steps, gate=gate)
            if gate != 1.0 and not eprop:
                for g in self.brain.opt.param_groups: g["lr"] = self.brain.lr
            if isinstance(r, tuple):                    # (first_loss, last_loss): free progress
                progress = max(0.0, r[0] - r[1])
            self.brain.observe_stream(text)             # the mind (hidden state) contexts on it
        self.learn_bps = self._ema(self.learn_bps, len(text) / max(time.time() - _t0, 1e-3))
        self.memory.write(text)                         # whole-life episodic memory
        self.mind.extend(self.brain.to_bytes(text)[-512:])
        return progress

    def _see_pixels(self, shot):
        """The visual channel. The recurrent cortex is a LANGUAGE model, so raw pixel
        bytes would be noise to it — vision therefore contributes the page's READABLE
        text (learned) plus the ASCII view (shown); the screenshot itself is perceived
        and displayed, not force-fed into the language mind-state."""
        self.last_screenshot = shot

    def _mind_text(self, n=600):
        return bytes(list(self.mind)[-n:]).decode("utf-8", "replace")

    # ---- BIRTH: distil teacher outputs + real text to COHERENCE ------ #
    def _birth_corpus(self, kb=250):
        parts = []
        if self.use_teacher:                              # the frozen teacher's "teachings"
            try:
                from .llm_teacher import QwenVLTeacher
                teacher = QwenVLTeacher(self.dev, dtype=torch.float32)
                for s in ["Explain how the world works, simply.", "Describe everyday life.",
                          "Tell a short true story.", "Explain nature simply.",
                          "Describe a city, a forest, and the sea."]:
                    _t0 = time.time()
                    t = teacher.generate(s, max_new_tokens=96, temperature=0.7)
                    if t:
                        self.teacher_cps = self._ema(self.teacher_cps, len(t) / max(time.time() - _t0, 1e-3))
                        self.teacher_name = "Qwen"
                        parts.append(t); self.log(f"teacher(Qwen {self.teacher_cps:.0f} c/s)> {t[:50]}")
                del teacher
            except Exception as e:
                self.log(f"teacher unavailable at birth: {str(e)[:50]}")
        try:                                              # real world text (wikitext-2)
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            wt = "".join(t for t in ds["text"] if len(t) > 100)
            parts.append(wt[:kb * 1000])
        except Exception as e:
            self.log(f"wikitext unavailable at birth: {str(e)[:50]}")
        return "\n".join(p for p in parts if p)

    def birth(self, on_update=None, max_epochs=60, target_bpb=1.6, corpus_kb=250):
        """Train the cortex to COHERENCE on the teacher's outputs + real text — the
        innate bootstrap ('kick-in'). It is born able to write; the life then evolves it."""
        if self.resumed:                                  # already lived — skip re-birth, resume living
            self.log("resumed — already competent, skipping birth; continuing the life")
            return 0
        import time as _t; t0 = _t.time()
        def _status(msg):                                 # push to BOTH the birth banner and the life-log
            if on_update: on_update({"status": msg})
            self.birth_stage = msg
        _status("gathering the birth corpus (teacher + real text)…")
        self.log("BIRTH ▸ gathering corpus…")
        corpus = self._birth_corpus(corpus_kb)
        if not corpus:
            self.log("born with nothing to learn from"); return 0
        self.memory.write(corpus)                         # seed episodic memory
        kb = len(corpus) // 1000
        self.log(f"BIRTH ▸ corpus ready: {kb} KB. Learning to write (target ≤{target_bpb} bpb, ≤{max_epochs} epochs)…")
        _status(f"corpus ready ({kb} KB) — starting to learn to write")
        best, stall, last_log = float("inf"), 0, _t.time()
        for ep in range(max_epochs):
            ep_t = _t.time()
            def _beat(step, steps, loss, _ep=ep):         # intra-epoch heartbeat: the dead zone is gone
                el = _t.time() - t0
                _status(f"being born · epoch {_ep} · step {step}/{steps} · loss {loss:.2f} · {el:.0f}s")
                nonlocal last_log
                if _t.time() - last_log > 4:              # keep the life-log panel visibly moving
                    self.log(f"  ▸ epoch {_ep} step {step}/{steps} · loss {loss:.2f} · {el:.0f}s elapsed")
                    last_log = _t.time()
            self.brain.learn_text(corpus, epochs=1, max_steps=40, on_step=_beat)
            bpb = self.brain.bits_per_byte(corpus[-3000:])
            gain = best - bpb
            _status(f"being born · epoch {ep} done · {bpb:.2f} bpb (best {min(best,bpb):.2f}) · {_t.time()-t0:.0f}s")
            self.log(f"  birth epoch {ep}: {bpb:.2f} bits/byte "
                     f"(Δ{gain:+.3f}, {_t.time()-ep_t:.0f}s/epoch, stall {stall}/6)")
            # stop on the target, OR when learning PLATEAUS (the spiking core floors ~3 bpb,
            # never reaching target_bpb=1.6 — without this it burned all 60 epochs / ~220s),
            # OR the wall cap. Plateau = no >0.01 bpb gain for 6 epochs.
            if bpb < best - 0.01:
                best, stall = bpb, 0
            else:
                stall += 1
            if bpb < target_bpb or stall >= 6 or _t.time() - t0 > 300:
                reason = ("hit target" if bpb < target_bpb else "plateaued" if stall >= 6 else "wall-clock cap")
                self.log(f"BIRTH ▸ stopping ({reason}) after epoch {ep}")
                break
        self.brain.observe_stream(corpus[-800:])          # prime the mind-state
        self.log(f"BORN able to write ({self.brain.bits_per_byte(corpus[-3000:]):.2f} bits/byte, "
                 f"{_t.time()-t0:.0f}s total). Now living.")
        self.birth_stage = ""
        return len(corpus)

    # ---- THINK: the always-on inner monologue ------------------------ #
    def think(self, on_update=None):
        _t0 = time.time()
        chunk = self.brain.think(n=self.think_chunk, temperature=getattr(self, "think_temp", 0.6))
        self.think_bps = self._ema(self.think_bps, self.think_chunk / max(time.time() - _t0, 1e-3))
        self.thought = chunk
        self.mind.extend(self.brain.to_bytes(chunk))
        # coalesce tiny fragments into readable, timestamped lines for the UI feed
        self._line_buf += chunk
        if len(self._line_buf) >= 56 or any(c in chunk for c in ".!?\n"):
            line = self._line_buf.strip()
            if line:
                self.thought_log.append((time.strftime("%H:%M:%S"), line))
            self._line_buf = ""
        if on_update:
            on_update({"thought_chunk": chunk, "mind": self._mind_text(500)})

    def _resonate(self, on_update=None):
        """Resonate in parallel (§ batching): k thought streams at once, in ONE batched
        forward (~the cost of a single stream). The brain keeps the stream it itself finds
        most fluent — parallel exploration → a better single thought — and that reflection
        re-enters its own stream (it learns from its own best thinking)."""
        if self.core != "spiking":
            return
        try:
            streams = self.brain.resonate(k=self.resonate_k, n=24, temperature=0.95)
        except Exception:
            return
        self.parallel_thoughts = streams
        best = min(streams, key=lambda s: self.brain.bits_per_byte(s) if len(s) >= 8 else 9.9)
        self.thought = best
        self.mind.extend(self.brain.to_bytes(best))
        b = best.strip().replace("\n", " ")
        if b:
            self.thought_log.append((time.strftime("%H:%M:%S"), "↻ " + b))   # ↻ = parallel reflection
        self.brain.observe_stream(best)          # the chosen reflection re-enters the stream
        self.log("resonate× " + " ┆ ".join(s.replace("\n", " ")[:16] for s in streams))
        if on_update:
            on_update({"parallel": streams, "thought_chunk": best, "mind": self._mind_text(500)})

    # ---- DRIVE IT: teach + steer the learning feed, LIVE (API) ------- #
    def focus(self, topics=None, urls=None, mode=None, label=None):
        """Redirect the learning feed live — steer curiosity to `topics`, browsing to `urls`,
        and set mode (random|topics|urls|mixed). e.g. focus(topics=['music theory','harmony'],
        mode='topics', label='music'). No restart; takes effect on the next sense tick."""
        if topics is not None:
            self.topics = [t.strip() for t in topics if t and t.strip()] or list(TOPICS)
        if urls is not None:
            self.browse = [u.strip() for u in urls if u and u.strip()] or list(BROWSE)
        if mode in ("random", "topics", "urls", "mixed"):
            self.feed_mode = mode
        if label is not None:
            self.focus_label = label
        self.log(f"focus → mode={self.feed_mode} label={self.focus_label!r} "
                 f"topics={self.topics[:3]}… urls={len(self.browse)}")
        return dict(mode=self.feed_mode, label=self.focus_label, topics=self.topics, urls=self.browse)

    def teach(self, text=None, topic=None, url=None, path=None, label=None):
        """Teach it something specific NOW — raw text, a wiki topic, a web page, or a local
        file/dir (code, music notation, prose). Resolved to bytes, chunked, and learned with
        PRIORITY in the live loop. It learns any byte stream, so code/ABC-music/text all work."""
        src = label or topic or url or path or "text"
        try:
            body = (text if text else
                    partner.web_topic(topic) if topic else
                    partner.web_text(url) if url else
                    self._read_path(path) if path else "")
        except Exception as e:
            return {"ok": False, "err": str(e)[:80]}
        body = (body or "").strip()
        if len(body) < 8:
            return {"ok": False, "err": "nothing to teach (empty source)"}
        n = 0
        for i in range(0, len(body), 1200):        # chunk into learnable lessons
            self._teach_q.put((src, body[i:i + 1200])); n += 1
        self.log(f"teach[{src}] queued {n} chunks ({len(body)} bytes) for priority learning")
        return {"ok": True, "source": src, "chunks": n, "bytes": len(body)}

    def use_tool(self, name, input_text=None, learn=True):
        """Interact with a registered tool / other AI: send it the brain's message (or your
        text), fold its OUTPUT into the unified byte stream (text learned as language;
        audio/image/bytes encoded via senses.py), and return the raw output. Thread-safe —
        the subprocess runs anywhere; learning is queued for the main loop."""
        inp = (input_text or self.thought or "hello, teach me something.").strip()
        res = self.tools.run(name, inp)
        if not res.get("ok"):
            self.log(f"tool[{name}] failed: {str(res.get('err',''))[:60]}")
            return res
        kind, out = res.get("kind", "text"), res.get("output", "")
        if learn:
            payload = self._encode_tool_output(kind, out)
            if payload is not None:
                src = f"tool:{name}" + (f" ({kind})" if kind != "text" else "")
                self._teach_q.put((src, payload))
        self.teacher_name = name
        self.log(f"tool[{name}]({kind})> {str(out)[:66]}")
        return res

    def _encode_tool_output(self, kind, out):
        """Fold a tool's output into the ONE byte stream. text → str (learned as language);
        audio → cochlea samples; image → retina pixels; else → raw self-tagged levels. Records
        non-text frames as REPLAYABLE observations. Returns a str (text) or a list of
        byte-levels (any other modality) for the priority learner."""
        try:
            if kind == "text":
                return out
            if kind == "audio":
                import wave, numpy as np
                w = wave.open(out.strip(), "rb"); sr = w.getframerate()
                samp = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16); w.close()
                body = senses.encode_audio(samp, sr)                # resampled to 4000 Hz, 8-bit
                self._record_obs("audio", body, sr=4000, label=os.path.basename(out.strip())[:24])
                return senses.frame("audio", body)
            if kind == "image":
                from PIL import Image
                body = senses.encode_image(Image.open(out.strip()))
                self._record_obs("image", body, size=int(round(len(body) ** 0.5)),
                                 label=os.path.basename(out.strip())[:24])
                return senses.frame("image", body)
            return senses.frame("self", list(out.encode("utf-8", "replace")))   # bytes/other
        except Exception as e:
            self.log(f"encode tool output ({kind}) failed: {str(e)[:50]}")
            return out if kind == "text" else None

    # ---- per-module live tuning (different parts of the net) --------- #
    def _net_params(self):
        """Current live-tunable parameters, per part of the net."""
        p = {"cortex": {"lr": round(self.brain.lr, 6), "read_alpha": getattr(self.brain, "read_alpha", None),
                        "seq": self.brain.seq, "think_temp": getattr(self, "think_temp", 0.6),
                        "prune_frac": getattr(self.brain, "prune_frac", 0.05),
                        "grow_syn_frac": getattr(self.brain, "grow_syn_frac", 0.15),
                        "syn_density": getattr(self.brain, "syn_density", 1.0),
                        "learn_rule": getattr(self.brain, "learn_rule", "bptt"),
                        "eprop_lr_scale": getattr(self.brain, "eprop_lr_scale", 15.0)}}
        if self.modules_on:
            p["hippocampus"] = {"beta": self.hippo.beta, "sparsity": self.hippo.a, "capacity": self.hippo.cap,
                                "thr": getattr(self.hippo, "thr", None), "g_inh": getattr(self.hippo, "g_inh", None),
                                "syn_density": getattr(self.hippo, "syn_density", 1.0)}
            p["bg"] = {"alpha_v": self.bg.av, "alpha_pi": self.bg.ap,
                       "beta": getattr(self.bg, "beta", None), "thr": getattr(self.bg, "thr", None),
                       "syn_density": getattr(self.bg, "syn_density", 1.0)}
            p["neuromod"] = dict(self.nm.tone)
            p["cerebellum"] = {"eta": self.cereb_eta, "sparsity": self.cerebellum.sparsity,
                               "g_golgi": self.cerebellum.g_golgi, "thr0": self.cerebellum.thr0,
                               "syn_density": getattr(self.cerebellum, "syn_density", 1.0)}
        return p

    @staticmethod
    def _set_mod_growth(mod, p, applied):
        """Shared: set a module's per-region synaptic growth + prune rate live."""
        if "grow_syn_frac" in p:
            mod.grow_syn_frac = max(0.0, min(1.0, float(p["grow_syn_frac"]))); applied["grow_syn_frac"] = mod.grow_syn_frac
        if "prune_frac" in p:
            mod.prune_frac = max(0.0, min(0.5, float(p["prune_frac"]))); applied["prune_frac"] = mod.prune_frac

    def set_net(self, target, params):
        """Modify a specific PART of the net live (no restart): cortex | hippocampus | bg |
        neuromod. e.g. set_net('cortex', {'lr': 1e-3}) or set_net('neuromod', {'da': 0.9})."""
        p = params or {}; applied = {}
        try:
            if target == "cortex":
                if "lr" in p:
                    self.brain.lr = float(p["lr"])
                    for g in self.brain.opt.param_groups:
                        g["lr"] = self.brain.lr
                    applied["lr"] = self.brain.lr
                if "read_alpha" in p:
                    self.brain.read_alpha = float(p["read_alpha"]); applied["read_alpha"] = self.brain.read_alpha
                if "seq" in p:
                    self.brain.seq = max(8, int(p["seq"])); applied["seq"] = self.brain.seq
                if "think_temp" in p:
                    self.think_temp = float(p["think_temp"]); applied["think_temp"] = self.think_temp
                if "prune_frac" in p:
                    self.brain.prune_frac = max(0.0, min(0.5, float(p["prune_frac"]))); applied["prune_frac"] = self.brain.prune_frac
                if "grow_syn_frac" in p:
                    self.brain.grow_syn_frac = max(0.0, min(1.0, float(p["grow_syn_frac"]))); applied["grow_syn_frac"] = self.brain.grow_syn_frac
                if "eprop_lr_scale" in p:
                    self.brain.eprop_lr_scale = max(0.1, float(p["eprop_lr_scale"])); applied["eprop_lr_scale"] = self.brain.eprop_lr_scale
                if "learn_rule" in p and p["learn_rule"] in ("eprop", "bptt"):
                    self.brain.learn_rule = p["learn_rule"]; applied["learn_rule"] = self.brain.learn_rule
            elif target == "hippocampus" and self.modules_on:
                if "beta" in p: self.hippo.beta = float(p["beta"]); applied["beta"] = self.hippo.beta
                if "sparsity" in p: self.hippo.a = float(p["sparsity"]); applied["sparsity"] = self.hippo.a
                if "capacity" in p: self.hippo.cap = max(16, int(p["capacity"])); applied["capacity"] = self.hippo.cap
                if "thr" in p: self.hippo.thr = float(p["thr"]); applied["thr"] = self.hippo.thr           # CA3 spike threshold
                if "g_inh" in p: self.hippo.g_inh = float(p["g_inh"]); applied["g_inh"] = self.hippo.g_inh  # lateral WTA inhibition
                self._set_mod_growth(self.hippo, p, applied)
            elif target == "bg" and self.modules_on:
                if "alpha_v" in p: self.bg.av = float(p["alpha_v"]); applied["alpha_v"] = self.bg.av
                if "alpha_pi" in p: self.bg.ap = float(p["alpha_pi"]); applied["alpha_pi"] = self.bg.ap
                if "beta" in p: self.bg.beta = float(p["beta"]); applied["beta"] = self.bg.beta            # MSN membrane leak
                if "thr" in p: self.bg.thr = float(p["thr"]); applied["thr"] = self.bg.thr                # MSN spike threshold
                self._set_mod_growth(self.bg, p, applied)
            elif target == "neuromod" and self.modules_on:
                for k in ("da", "ach", "ne", "ht"):
                    if k in p: self.nm.tone[k] = float(p[k]); applied[k] = self.nm.tone[k]
            elif target == "cerebellum" and self.modules_on:
                if "eta" in p: self.cereb_eta = float(p["eta"]); applied["eta"] = self.cereb_eta
                if "sparsity" in p: self.cerebellum.sparsity = float(p["sparsity"]); applied["sparsity"] = self.cerebellum.sparsity
                if "g_golgi" in p: self.cerebellum.g_golgi = float(p["g_golgi"]); applied["g_golgi"] = self.cerebellum.g_golgi
                if "thr0" in p: self.cerebellum.thr0 = float(p["thr0"]); applied["thr0"] = self.cerebellum.thr0
                self._set_mod_growth(self.cerebellum, p, applied)
            elif target in ("all", "global"):                    # set a knob on EVERY region at once
                for t in ("cortex", "hippocampus", "bg", "cerebellum", "neuromod"):
                    r = self.set_net(t, p)
                    if r.get("applied"): applied[t] = r["applied"]
                return {"ok": True, "target": "all", "applied": applied}
            else:
                return {"ok": False, "err": f"unknown or inactive target '{target}'"}
        except Exception as e:
            return {"ok": False, "err": str(e)[:80]}
        self.log(f"set_net[{target}] {applied}")
        return {"ok": True, "target": target, "applied": applied}

    # ---- sensory observation replay (see/hear what it observed) ------ #
    def _record_obs(self, modality, body, **meta):
        self._obs_i += 1
        self.last_observations.append(dict(i=self._obs_i, modality=modality, body=list(body),
                                           ts=time.strftime("%H:%M:%S"), **meta))

    def observation_media(self, i):
        """Reconstruct a stored sensory frame back into playable/viewable media (a low-fi
        rendering of exactly what the brain SENSED — the encoders are lossy). Returns
        (bytes, content_type) or None."""
        obs = next((o for o in self.last_observations if o["i"] == int(i)), None)
        if obs is None:
            return None
        body = obs["body"]
        try:
            if obs["modality"] == "audio":
                import wave, io, numpy as np
                x = (np.asarray(body, dtype=np.float32) / 255.0 - 0.5) * 2.0     # inverse quantise
                pcm = np.clip(x * 32767, -32768, 32767).astype(np.int16)
                buf = io.BytesIO(); w = wave.open(buf, "wb")
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(obs.get("sr", 4000)))
                w.writeframes(pcm.tobytes()); w.close()
                return buf.getvalue(), "audio/wav"
            if obs["modality"] == "image":
                import io, numpy as np
                from PIL import Image
                sz = int(obs.get("size") or round(len(body) ** 0.5))
                arr = np.asarray(body[:sz * sz], dtype=np.uint8).reshape(sz, sz)
                buf = io.BytesIO(); Image.fromarray(arr, "L").resize((sz * 4, sz * 4)).save(buf, "PNG")
                return buf.getvalue(), "image/png"
        except Exception as e:
            self.log(f"observation render failed: {str(e)[:50]}")
        return None

    def _read_path(self, path, max_bytes=400000):
        """Read a file, or concatenate the readable/code/notation files in a directory."""
        exts = (".py", ".js", ".ts", ".c", ".cpp", ".h", ".rs", ".go", ".java", ".txt",
                ".md", ".abc", ".ly", ".mid", ".json", ".html", ".css", ".sh", ".rb", ".lua")
        if os.path.isdir(path):
            buf = []
            for root, _, files in os.walk(path):
                for fn in sorted(files):
                    if fn.lower().endswith(exts):
                        try: buf.append(open(os.path.join(root, fn), encoding="utf-8", errors="replace").read())
                        except Exception: pass
                if sum(len(x) for x in buf) > max_bytes:
                    break
            return "\n\n".join(buf)[:max_bytes]
        return open(path, encoding="utf-8", errors="replace").read()[:max_bytes]

    # ---- FEEL: your feedback, woven in non-blocking ------------------ #
    def inject(self, text):
        with self._lock:
            self._feedback.append(text.strip())

    def _drain_feedback(self):
        with self._lock:
            fb = list(self._feedback); self._feedback.clear()
        for text in fb:
            low = text.lower()
            if low.startswith("browse ") or low.startswith("look "):
                self._perc_q.put(("cmd_browse", text.split(None, 1)[1].strip()))
            elif low in ("?time", "time"):
                self.log(f"brain's clock: {self.clock.tell()}")
            else:
                self.log(f"you → brain: {text[:70]}")
                self._learn_text("you say: " + text, steps=10)   # your words are strong feedback
                self.debt += 0.5

    # ---- SENSE: background thread — Claude + visual web -------------- #
    def _sense_worker(self):
        step = 0
        while self._running:
            if not self.awake:
                time.sleep(0.3); continue
            try:
                # the feed points where self.feed_mode says (steer it live via focus()).
                mode = self.feed_mode
                autos = self.tools.autonomous()
                want_browse = (self.use_visual and self.browse and
                               (mode == "urls" or (mode in ("random", "mixed") and step % 2 == 1)))
                if autos and step % 3 == 2:
                    # converse with a registered AI/tool on its own (generalised teacher)
                    self.use_tool(autos[step % len(autos)],
                                  self.thought or motor.utter(self.brain, seed="idea", n=60))
                elif want_browse:
                    self._perc_q.put(("browse", self.browse[step % len(self.browse)]))
                else:
                    # §2 basal-ganglia curiosity picks WHICH focus topic teaches it most
                    topic_idx = self._pick_topic(step)
                    topic = self.topics[topic_idx % len(self.topics)]
                    babble = self.thought or motor.utter(self.brain, seed=topic.split()[-1][:6], n=60)
                    _t0 = time.time()
                    lesson = partner.claude_say(motor.teach_prompt(babble, topic), budget=self.budget)
                    if _usable_lesson(lesson):
                        self.teacher_cps = self._ema(self.teacher_cps, len(lesson) / max(time.time() - _t0, 1e-3))
                        self.teacher_name = "Sonnet"
                        self._perc_q.put(("teach", topic_idx, babble, lesson))
                step += 1
            except Exception as e:
                self.log(f"sense error: {str(e)[:50]}")
            time.sleep(self.perceive_gap)

    def _eyes(self):
        if self._browser is not None or self._browser_dead:
            return self._browser
        try:
            from .visual_web import VisualBrowser
            self._browser = VisualBrowser(); self.log("opened eyes (headless browser)")
        except Exception as e:
            self._browser_dead = True; self.log(f"visual unavailable: {str(e)[:50]}")
        return self._browser

    def _do_browse(self, url, on_update):
        vb = self._eyes()
        if vb is None or not vb.open(url):
            return
        self.log(f"looking at {url}")
        for i in range(4):
            shot = vb.screenshot(); vtext = vb.visible_text(1500)
            self._see_pixels(shot)                          # visual channel (mind sees pixels)
            if on_update:
                on_update({"ascii": image_to_ascii(shot), "perceived": vtext[:1400],
                           "status": f"reading {url}"})
            self._learn_text(vtext[:1400], steps=6)         # learn the page's language
            self.debt += 1.0
            vb.scroll(1000)

    def _consume_perceptions(self, on_update):
        # directed lessons (teach() / tool outputs) are learned FIRST (priority)
        if not self._teach_q.empty():
            src, chunk = self._teach_q.get()
            if isinstance(chunk, list):                  # raw sensory byte frame (audio/image/bytes)
                if not self.freeze_learning:
                    self.brain.learn_text(chunk, epochs=1, max_steps=self.learn_steps)
                self.clock.tick()
                preview = f"[{len(chunk)} sensory bytes]"
            else:                                        # text → learned as language
                self._learn_text(chunk, steps=self.learn_steps)
                self._index_episode(chunk)
                preview = chunk[:60].strip()
            self.debt += 1.0
            self.log(f"learned[{src}]> {preview}")
            if on_update:
                on_update({"perceived": f"[{src}] " + (chunk[:1200] if isinstance(chunk, str) else preview),
                           "status": f"learning: {src}"})
            return 1
        # drain only ONE per loop iteration so BPTT learning never starves the thinking
        drained = 0
        while not self._perc_q.empty() and drained < 1:
            if self._perc_q.qsize() > 12:                    # bound the backlog
                for _ in range(self._perc_q.qsize() - 12):
                    try: self._perc_q.get_nowait()
                    except Exception: break
            item = self._perc_q.get()
            kind = item[0]
            if kind == "teach":
                _, topic_idx, babble, lesson = item
                topic = self.topics[topic_idx % len(self.topics)] if self.modules_on else topic_idx
                self.log(f"teacher({topic})> {lesson[:70]}")
                progress = self._learn_text(lesson, steps=8)       # learn from Claude's teaching
                self._reward_curiosity(topic_idx, progress)        # §2 dopamine on learning progress (free)
                self._index_episode(lesson)                        # §4 hippocampal novelty
                self._train_cerebellum(lesson)                     # §1 fast supervised forward model
                self.debt += 1.0
                if on_update: on_update({"perceived": lesson[:1400], "status": f"learning about {topic}"})
            elif kind in ("browse", "cmd_browse"):
                self._do_browse(item[1], on_update)
            drained += 1
            self._perc_count += 1
        return drained

    # ---- the other four systems, coupled by activation/reward (§6 cut) --- #
    def _fingerprint(self, text):
        """A growth-robust content fingerprint of an experience: the normalized 256-bin byte
        histogram. Decoupled from the (growing) cortex width, so the hippocampus can index
        episodes for novelty across a whole life."""
        ids = self.brain.to_bytes(text)[:2000]
        if not ids:
            return None
        h = torch.zeros(256, device=self.dev)
        h.scatter_add_(0, torch.tensor(ids, device=self.dev),
                       torch.ones(len(ids), device=self.dev))
        return (h / (h.sum() + 1e-6)).unsqueeze(0)

    def _pick_topic(self, step):
        """§2 curiosity: the basal ganglia chooses which topic to explore, learning (by
        dopamine RPE on learning-progress reward) which topics teach the cortex the most.
        Falls back to round-robin if modules are off."""
        if not self.modules_on:
            return step % len(TOPICS)
        try:
            a, _ = self.bg.act(self._topic_feat[self._last_topic:self._last_topic + 1].to(self.bg.device))
            return int(a.item())
        except Exception:
            return step % len(TOPICS)

    def _reward_curiosity(self, topic_idx, progress):
        """§2 dopamine: reward the basal-ganglia topic policy by LEARNING PROGRESS — the drop
        in training loss on the lesson, already computed during learning (zero extra compute).
        The brain comes to prefer topics that teach its cortex the most (curiosity)."""
        if not self.modules_on:
            return
        try:
            bd = self.bg.device                              # bg may live on its own device
            reward = torch.tensor([float(progress)], device=bd)
            self.bg.train_step(self._topic_feat[self._last_topic:self._last_topic + 1].to(bd),
                               torch.tensor([topic_idx], device=bd), reward)
            self._last_topic = topic_idx
        except Exception:
            pass

    def _train_cerebellum(self, text):
        """§1 cerebellum: a fast SUPERVISED next-byte forward model, learned by its own delta
        rule (climbing-fibre error) on the byte stream — gradient-cut from the cortex (§6). The
        mossy input is a bag-of-bytes of a short window → Golgi-controlled sparse granule code →
        Purkinje readout predicts the next byte. Complementary to the cortex's slow BPTT; its
        error `cereb_mse` is tracked. (Not fed back into generation — a monitored forward model.)"""
        if not self.modules_on or self.freeze_learning:
            return
        data = self.brain.to_bytes(text)
        W = 24
        if len(data) < W + 2:
            return
        try:
            t = torch.tensor(data, device=self.dev)
            n = min(48, len(data) - W - 1)
            i = torch.randint(0, len(data) - W - 1, (n,), device=self.dev)
            feats = torch.zeros(n, 256, device=self.dev)
            for j in range(n):
                feats[j].scatter_add_(0, t[i[j]:i[j] + W], torch.ones(W, device=self.dev))
            feats /= W
            tgt = torch.zeros(n, 256, device=self.dev)
            tgt[torch.arange(n), t[i + W]] = 1.0
            cd = self.cerebellum.device                      # cerebellum may live on its own device
            mse = self.cerebellum.train_step(feats.to(cd), tgt.to(cd), eta=getattr(self, "cereb_eta", 0.3))
            self.cereb_mse = self._ema(self.cereb_mse, float(mse))
        except Exception:
            pass

    def _index_episode(self, text):
        """§4 hippocampus: store the experience's fingerprint and report how NOVEL it was
        (1 − recall similarity to the closest prior memory) — the CLS novelty/salience signal
        that gates how hard sleep should replay."""
        if not self.modules_on:
            return
        fp = self._fingerprint(text)
        if fp is None:
            return
        try:
            fp = fp.to(self.hippo.device)                    # hippocampus may live on its own device
            if self.hippo.keys.shape[0] == 0:
                self._novelty = 1.0
            else:
                sim = torch.cosine_similarity(self.hippo.recall(fp), fp, dim=1).clamp(-1, 1).mean()
                self._novelty = float((1.0 - sim).clamp(0.0, 1.0))
            # §4/§7.4: only WRITE a genuinely novel episode — storing every familiar pattern
            # would fill the ring buffer and evict the novel older ones (encode without
            # overwriting). Gate is live-tunable via /api/net (cortex? no — a life attr).
            if self.hippo.keys.shape[0] < 32 or self._novelty >= self.novelty_gate:
                self.hippo.store(fp)
        except Exception:
            pass

    # ---- wake / sleep (§7.6, §8-9) ----------------------------------- #
    def _awake_seconds(self):
        return time.time() - self.wake_start

    def should_sleep(self):
        if self.freeze_sleep: return False                # cycle frozen → stays awake
        dur = self._awake_seconds()
        if dur < self.min_awake: return False
        if dur >= self.max_awake: return True
        return self.debt >= self.debt_threshold

    def _begin_sleep(self):
        self.awake = False
        # consolidation scales with how much was learned (debt) but is CAPPED at max_sleep so a long
        # wake (a "day of compute" cycle) doesn't trigger an hours-long sleep.
        self.sleep_remaining = min(getattr(self, "max_sleep", 900.0), max(30, int(self.debt * 20)))
        if self.modules_on: self.nm.set_phase("nrem")     # §5 NREM tone: low ACh/NE → consolidate
        self.log(f"sleepy (awake {self._awake_seconds():.0f}s, debt {self.debt:.0f}) — sleeping "
                 f"{self.sleep_remaining:.0f}s")

    def _sleep_tick(self):
        """§8-9 NREM: the deep-learning phase. Sleep replays many chunks of the WHOLE life
        (tiered episodic memory) with heavy BPTT (waking thought stays fast; sleep does the
        real consolidation, as in the brain), then §8.4 SHY multiplicative downscale so
        sleep also renormalises (restores ⟨Δw⟩_wake + ⟨Δw⟩_sleep ≈ 0, not only potentiates)."""
        # §4 novelty-gated replay (CLS): a life full of NOVEL experience replays harder;
        # a familiar one consolidates lightly (the hippocampal salience signal sets the load).
        # replay load = novelty-gated chunk count × per-chunk BPTT steps. Both are live-tunable
        # (`sleep_chunks`, `sleep_steps`, `sleep_seq`) because at large scale each step costs seconds
        # — a small fast brain can afford the default heavy replay; a 100k+ GPU brain must run it
        # lighter or a single night would take hours.
        base = min(getattr(self, "sleep_chunks", 10), 5 + int(5 * getattr(self, "_novelty", 0.5))) if self.modules_on \
            else min(getattr(self, "sleep_chunks", 10), 5)
        steps = int(getattr(self, "sleep_steps", 14))
        sseq = int(getattr(self, "sleep_seq", 96)) if self.core == "spiking" else None
        for _ in range(base):
            chunk = self.memory.sample(1400)
            if chunk and len(chunk) > 32 and not self.freeze_learning:
                self.brain.learn_text(chunk, epochs=1, max_steps=steps, **({"seq": sseq} if sseq else {}))
                self._replay_count = getattr(self, "_replay_count", 0) + 1   # replay-consolidation counter
        if not self.freeze_learning:
            with torch.no_grad():
                for p in self.brain.parameters():
                    p.mul_(1.0 - 2e-5)                       # §8.4 SHY homeostatic downscale
        self.sleep_remaining -= 20

    def _wake_up(self):
        self.awake = True; self.slept_count += 1; self.debt = 0.0; self.wake_start = time.time()
        d = self.brain.develop(allow_grow=not self.freeze_growth, add=int(self.grow_add))
        if self.modules_on:
            self.nm.set_phase("wake")                        # §5 wake tone
            # §1/§2/§4 develop SYNAPTICALLY alongside the cortex (fixed neurons): childhood grows
            # module synapses, adolescence prunes the weak ones — same model as the cortex.
            if not self.freeze_growth:
                for mod in (self.cerebellum, self.bg, self.hippo):
                    try:    # each region develops at its OWN synaptic growth/prune rate
                        if d.get("phase") == "child": mod.grow_synapses(getattr(mod, "grow_syn_frac", 0.15))
                        elif d.get("phase") == "adolescent": mod.prune_synapses(getattr(mod, "prune_frac", 0.05))
                    except Exception: pass
        self.save_life()                                     # checkpoint on the night boundary
        self.log(f"woke (night {self.slept_count}) {d['phase']} age {d['age']} · "
                 f"{d.get('neurons', d['n_granule'])} neurons (fixed) · "
                 f"{d.get('synapses', 0):,} synapses (+{d.get('grown',0):,}syn/-{d.get('pruned',0):,}syn) · "
                 f"model {self.brain.model_gb():.3f}GB | memory {self.memory.stats()}")

    # ---- the continuous life loop ------------------------------------ #
    def request_stop(self):
        """Ask the life to stop — from any thread. Sets the run flag AND terminates any in-flight
        teacher (Sonnet) subprocess, so the sense worker unblocks immediately instead of waiting out
        the call. Both the main loop and the sense worker check `_running` each iteration."""
        self._running = False
        try:
            partner.kill_active_calls()          # unblock a sense worker stuck in a Sonnet call
        except Exception:
            pass

    def _stop_requested(self):
        """A STOP file in the run folder = a graceful stop request from another process."""
        return os.path.exists(os.path.join(self.resdir, "STOP"))

    def run(self, on_update=None, stop_flag=None):
        self._running = True
        try:                                              # clear any stale stop request
            os.remove(os.path.join(self.resdir, "STOP"))
        except OSError:
            pass
        self._perc_thread = threading.Thread(target=self._sense_worker, daemon=True)
        self._perc_thread.start()
        self.log("alive — thinking continuously")
        try:
            while self._running and not (stop_flag and stop_flag()) and not self._stop_requested():
                if not self.awake:
                    self._sleep_tick()
                    if self.sleep_remaining <= 0: self._wake_up()
                    self._emit(on_update, force=True); time.sleep(0.05); continue
                self._drain_feedback()
                if time.time() - getattr(self, "_last_consume", 0) > 1.2:
                    self._consume_perceptions(on_update)     # learn periodically...
                    self._last_consume = time.time()
                self.think(on_update)                        # ...but THINK every iteration
                if time.time() - getattr(self, "_last_reson", 0) > 5.0:
                    self._resonate(on_update)                # resonate in parallel every ~5s
                    self._last_reson = time.time()
                self.cycle += 1
                if self.should_sleep(): self._begin_sleep()
                self._emit(on_update)
                if time.time() - self._last_ckpt > 120:      # time-based fallback (night boundary is primary)
                    self.save_life(); self._last_ckpt = time.time()
                if time.time() - self._last_bound > 60:       # keep on-disk logs bounded
                    self._bound_logs(); self._last_bound = time.time()
        finally:
            self._running = False
            try: partner.kill_active_calls()                     # stop any in-flight teacher call
            except Exception: pass
            pt = getattr(self, "_perc_thread", None)             # let the sense worker exit too
            if pt is not None:
                try: pt.join(timeout=8)
                except Exception: pass
            if getattr(self, "_killed", False):
                self.log("force-killed — NOT checkpointing")
            else:
                self.log("stopping gracefully — saving checkpoint …")
                self.save_life()
            self.close()
            try:
                os.remove(os.path.join(self.resdir, "STOP"))    # consume the stop request
            except OSError:
                pass
            self.log(f"stopped cleanly. resume: --checkpoint {self.resdir}")

    @staticmethod
    def _ema(old, new, a=0.3):
        """Exponential moving average for the throughput meters (0 = uninitialised)."""
        return new if old <= 0 else (1 - a) * old + a * new

    def _metrics(self):
        """The evals are expensive — compute them at most every ~4s and cache.
        Returns (understanding = next-byte accuracy, time-sense accuracy, bits/byte)."""
        now = time.time()
        if not hasattr(self, "_mc") or now - self._mc_t > 4.0:
            self._mc = (round(self.brain.next_byte_acc(self.probe), 4),
                        round(self.brain.next_byte_acc(self._time_probe), 4),
                        round(self.brain.bits_per_byte(self.probe[:2000]), 4))
            self._mc_t = now
        return self._mc

    def _net_diag(self):
        """Deeper 'state of the net' — perplexity (train + on its own generation), output
        entropy, spike firing rate, per-layer weight health, hippocampal recall fidelity.
        Expensive (a generation + forward + weight sweep) → cached ~6 s."""
        now = time.time()
        if not hasattr(self, "_nd") or now - self._nd_t > 6.0:
            d = {}
            if self.core in ("rnn", "spiking"):
                try: d["perplexity_train"] = round(self.brain.train_perplexity(self.probe[:1500]), 2)
                except Exception: pass
                try:
                    g = self.brain.generate_diag(n=140)
                    d["gen_entropy"] = round(g["entropy_bits"], 3); d["perplexity_gen"] = round(g["perplexity"], 2)
                except Exception: pass
                try: d["spike_rate"] = round(self.brain.spike_rate(self.probe[:512]), 4)
                except Exception: pass
                try: d["weights"] = {k: round(v, 4) for k, v in self.brain.weight_stats().items()}
                except Exception: pass
            if self.modules_on:
                try:
                    if self.hippo.keys.shape[0] > 4:
                        vv = self.hippo.vals[:64]
                        rec = self.hippo.recall(vv)
                        d["hippo_fidelity"] = round(float(torch.cosine_similarity(rec, vv, 1).clamp(-1, 1).mean()), 3)
                    d["bg_policy_entropy"] = None
                    cur = self._curiosity_stats()
                    if cur: d["bg_policy_entropy"] = round(cur[0], 3)
                    d["cerebellum_mse"] = round(self.cereb_mse, 4)     # §1 forward-model error
                    d["bg_spike_rate"] = round(self.bg.spike_rate, 4)  # §2 medium-spiny firing
                    d["hippo_spike_rate"] = round(self.hippo.spike_rate, 4)  # §4 CA3 firing
                except Exception: pass
            self._nd = d; self._nd_t = now
        return self._nd

    # ---- ARCHITECTURE census: neurons + synapses, per part -------------- #
    def _arch_diag(self):
        """Per-part neuron and synapse counts — the living architecture, so its size + growth are
        observable by API (neurons are fixed at birth; synapses are what evolve over the life)."""
        parts = {}
        # §3 cortex — the growable spiking stack
        try:
            layers = [{"neurons": int(c.hid), "in": int(getattr(c, "in_dim", 0)),
                       "sparse": bool(getattr(c, "sparse_in", False) or hasattr(c, "rec_val"))}
                      for c in self.brain.cells]
            parts["cortex"] = dict(neurons=self.brain.neuron_count(),
                                   synapses=self.brain.active_synapse_count(),
                                   synapse_capacity=self.brain.synapse_capacity(),
                                   parameters=int(sum(p.numel() for p in self.brain.parameters())),
                                   density=round(self.brain.active_synapse_count() / max(1, self.brain.synapse_capacity()), 3),
                                   grow_syn_frac=round(getattr(self.brain, "grow_syn_frac", 0.15), 4),
                                   prune_frac=round(getattr(self.brain, "prune_frac", 0.05), 4),
                                   device=str(self.brain.device),
                                   layers=layers, layer_widths=[int(c.hid) for c in self.brain.cells])
        except Exception as e:
            parts["cortex"] = {"err": str(e)[:60]}
        def _mod_arch(mod, neurons, **extra):
            cap = mod.synapse_capacity(); act = mod.active_synapse_count()
            return dict(neurons=int(neurons), synapses=int(act), synapse_capacity=int(cap),
                        parameters=int(mod.parameter_count()),
                        density=round(act / max(1, cap), 3),
                        grow_syn_frac=round(getattr(mod, "grow_syn_frac", 0.15), 4),
                        prune_frac=round(getattr(mod, "prune_frac", 0.05), 4),
                        device=str(getattr(mod, "device", self.dev)), **extra)
        if self.modules_on:
            try:    # §1 cerebellum: granule neurons; mossy→granule + granule→Purkinje synapses
                parts["cerebellum"] = _mod_arch(self.cerebellum, self.cerebellum.M.shape[0])
            except Exception: pass
            try:    # §2 basal ganglia: medium-spiny neurons; input + critic + actor synapses
                parts["bg"] = _mod_arch(self.bg, self.bg.M.shape[0])
            except Exception: pass
            try:    # §4 hippocampus: DG/CA3 units; separator synapses + stored patterns
                parts["hippocampus"] = _mod_arch(self.hippo, self.hippo.M, stored=int(self.hippo.keys.shape[0]))
            except Exception: pass
            try:    # §5 neuromodulation: 4 DIFFUSE tone channels (da/ACh/NE/5HT) — modulatory glue, not
                # neurons and not synapses. Report them as tone_channels so they don't masquerade as
                # neural units or inflate the neuron/param totals; no connectome → 0 synapses at 0 density.
                parts["neuromod"] = dict(neurons=0, synapses=0, synapse_capacity=0, parameters=0,
                                         density=0.0, tone_channels=len(self.nm.tone),
                                         device=str(getattr(self.nm, "device", self.dev)))
            except Exception: pass
        tot_n = sum(p.get("neurons", 0) for p in parts.values() if isinstance(p, dict))
        tot_s = sum(p.get("synapses", 0) for p in parts.values() if isinstance(p, dict))
        tot_c = sum(p.get("synapse_capacity", 0) or 0 for p in parts.values() if isinstance(p, dict))
        tot_p = sum(p.get("parameters", 0) for p in parts.values() if isinstance(p, dict))
        return dict(parts=parts, total_neurons=int(tot_n), total_synapses=int(tot_s),
                    total_synapse_capacity=int(tot_c), total_parameters=int(tot_p))

    # ---- RESOURCE census: RAM / VRAM / storage, current + limits ------- #
    def _resources(self):
        """Current usage AND limit for RAM, VRAM (per device) and storage, so the footprint can be
        watched against its caps live (and as a time-series). Cached ~4 s."""
        now = time.time()
        if hasattr(self, "_res") and now - getattr(self, "_res_t", 0) < 4.0:
            return self._res
        import shutil
        r = {}
        # --- RAM: this process vs the whole box ---
        try:
            import resource as _rc
            rss_gb = _rc.getrusage(_rc.RUSAGE_SELF).ru_maxrss / 1e6      # ru_maxrss is kB on linux
        except Exception:
            rss_gb = None
        total_gb = avail_gb = None
        try:
            mi = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1); mi[k.strip()] = int(v.split()[0])   # kB
            total_gb = mi.get("MemTotal", 0) / 1e6; avail_gb = mi.get("MemAvailable", 0) / 1e6
        except Exception:
            pass
        r["ram"] = dict(process_gb=round(rss_gb, 3) if rss_gb is not None else None,
                        used_gb=round(total_gb - avail_gb, 2) if total_gb else None,
                        limit_gb=round(total_gb, 2) if total_gb else None,
                        pct=round((total_gb - avail_gb) / total_gb, 3) if total_gb else None)
        # --- VRAM: 0 on CPU; per-device on CUDA ---
        vr = dict(used_gb=0.0, limit_gb=0.0, pct=0.0, device=str(self.dev))
        try:
            if self.dev.type == "cuda":
                free, tot = torch.cuda.mem_get_info()
                vr = dict(used_gb=round(torch.cuda.memory_allocated() / 1e9, 3),
                          reserved_gb=round(torch.cuda.memory_reserved() / 1e9, 3),
                          limit_gb=round(tot / 1e9, 3), pct=round((tot - free) / tot, 3), device=str(self.dev))
        except Exception:
            pass
        r["vram"] = vr
        # --- storage: each bounded store vs its cap + the disk itself ---
        try:
            ck = sum(os.path.getsize(p) for p in (self.ckpt, self.ckpt + ".life") if os.path.exists(p))
            log_b = os.path.getsize(self.logpath) if os.path.exists(self.logpath) else 0
            tbdir = os.path.join(self.resdir, "tb")
            tb_b = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(tbdir) for f in fs) if os.path.isdir(tbdir) else 0
            mem = self.memory.stats(); replay_b = mem.get("disk_mb", 0) * 1e6
            du = shutil.disk_usage(self.resdir)
            r["storage"] = dict(
                checkpoint_gb=round(ck / 1e9, 3), checkpoint_limit_gb=round(self.max_model_gb, 2),
                replay_gb=round(replay_b / 1e9, 3), replay_limit_gb=round(self.memory.hard / 1e9, 2),
                log_mb=round(log_b / 1e6, 2), log_limit_mb=round(self.max_log_mb, 1),
                tb_mb=round(tb_b / 1e6, 2), tb_limit_mb=round(self.max_tb_mb, 1),
                run_total_gb=round((ck + log_b + tb_b + replay_b) / 1e9, 3),
                disk_free_gb=round(du.free / 1e9, 2), disk_total_gb=round(du.total / 1e9, 2),
                disk_pct=round(du.used / du.total, 3))
        except Exception as e:
            r["storage"] = {"err": str(e)[:60]}
        self._res = r; self._res_t = now
        return r

    def _arch_handle(self, target):
        """The object exposing neuron/synapse ops for a region: the cortex (SpikingBrain) or a
        §1/§2/§4 module. All share grow_neurons/grow_synapses/prune_synapses/active_synapse_count/
        synapse_capacity/_init_synapse_mask, so one code path drives every region."""
        if target == "cortex":
            return self.brain
        if target in ("cerebellum", "bg", "hippocampus") and self.modules_on:
            return {"cerebellum": self.cerebellum, "bg": self.bg, "hippocampus": self.hippo}[target]
        return None

    def edit_arch(self, target, op, amount=None, density=None):
        """Live per-region OR global architecture surgery (API-driven). ops:
          grow_neurons    — add `amount` neurons to a region
          set_neurons     — grow the region to a TARGET neuron count `amount` (can only add)
          grow_synapses   — activate `amount` (or fraction if <1) silent synapses
          prune_synapses  — silence `amount`-fraction weakest synapses
          set_synapses    — grow/prune to a TARGET active-synapse count `amount` (or `density` fraction)
          refresh_synapses— re-seed the region's connectome at `density`
        target = cortex | cerebellum | bg | hippocampus, or 'all'/'global' to hit every region."""
        op = (op or "").strip()
        if target in ("all", "global"):                          # fan the same op across every region
            out = {}
            for t in ("cortex", "cerebellum", "bg", "hippocampus"):
                if self._arch_handle(t) is not None:
                    out[t] = self.edit_arch(t, op, amount=amount, density=density).get("applied")
            self._nd_t = 0
            return {"ok": True, "target": "all", "op": op, "applied": out, "arch": self._arch_diag()}
        h = self._arch_handle(target)
        if h is None:
            return {"ok": False, "err": f"unknown/inactive target '{target}'"}
        is_cortex = (target == "cortex")
        prune = h.prune if is_cortex else h.prune_synapses
        applied = {}
        try:
            if op in ("grow_neurons", "set_neurons"):
                if op == "set_neurons":
                    cur = h.neuron_count() if is_cortex else h.M.shape[0]
                    add = max(0, int(amount or 0) - cur)         # can only ADD (spiking neurons don't shrink cleanly)
                else:
                    add = int(amount or (64 if is_cortex else 32))
                if add > 0:
                    (h.grow_neurons if is_cortex else h.grow)(add)
                applied["added_neurons"] = add
            elif op == "grow_synapses":
                a = float(amount if amount is not None else getattr(h, "grow_syn_frac", 0.15))
                cap, act = h.synapse_capacity(), h.active_synapse_count()
                frac = a if a < 1 else min(1.0, a / max(1, cap - act))
                applied["added_synapses"] = h.grow_synapses(frac)
            elif op == "prune_synapses":
                applied["pruned_synapses"] = prune(float(amount if amount is not None else getattr(h, "prune_frac", 0.05)))
            elif op == "set_synapses":
                cap, act = h.synapse_capacity(), h.active_synapse_count()
                tgt = int(density * cap) if (density is not None) else int(amount if amount is not None else act)
                tgt = max(0, min(cap, tgt))
                if tgt > act and cap > act:
                    applied["added_synapses"] = h.grow_synapses((tgt - act) / (cap - act))
                elif tgt < act and act > 0:
                    applied["pruned_synapses"] = prune((act - tgt) / act)
                applied["target"] = tgt; applied["now"] = h.active_synapse_count()
            elif op == "refresh_synapses":
                dens = float(density if density is not None else getattr(h, "syn_density", 1.0))
                h.syn_density = dens; h._init_synapse_mask(dens)
                if is_cortex:
                    self.brain.opt = torch.optim.Adam(self.brain.parameters(), lr=self.brain.lr)
                applied["refreshed_density"] = dens
            else:
                return {"ok": False, "err": f"unknown op '{op}'"}
        except Exception as e:
            return {"ok": False, "err": str(e)[:100]}
        self._nd_t = 0                                   # force a fresh diag next read
        self.log(f"edit_arch[{target}] {op} → {applied}")
        return {"ok": True, "target": target, "op": op, "applied": applied, "arch": self._arch_diag()}

    def set_part_device(self, target, device):
        """Place a region on its OWN device (CPU/GPU), live. The living loop already converts
        tensors at every cross-region boundary, so parts on different devices interoperate. target =
        cortex | cerebellum | bg | hippocampus | neuromod, or 'all'/'global'."""
        try:
            dev = torch.device(device)
        except Exception as e:
            return {"ok": False, "err": f"bad device '{device}': {str(e)[:60]}"}
        try:
            if target in ("all", "global"):
                for t in ("cortex", "cerebellum", "bg", "hippocampus"):
                    self.set_part_device(t, dev)
                self.dev = dev                               # the hub/default device follows
            elif target == "cortex":
                self.brain.to(dev); self.brain.device = dev
                for st in self.brain.opt.state.values():     # move Adam moment tensors too
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(dev)
            elif target in ("cerebellum", "bg", "hippocampus") and self.modules_on:
                {"cerebellum": self.cerebellum, "bg": self.bg, "hippocampus": self.hippo}[target].move_to(dev)
            elif target == "neuromod":
                pass                                         # scalar tone — device-agnostic
            else:
                return {"ok": False, "err": f"unknown/inactive target '{target}'"}
        except Exception as e:
            return {"ok": False, "err": str(e)[:100]}
        self._nd_t = 0
        self.log(f"device[{target}] → {dev}")
        return {"ok": True, "target": target, "device": str(dev), "arch": self._arch_diag()}

    def _restore_module_masks(self, m):
        """Restore each §1/§2/§4 module's synapse mask + density from a checkpoint. A saved mask is
        used only if every matrix shape matches the (already-restored) weights; otherwise the module
        re-seeds at the saved density. Legacy checkpoints (no *_smask) leave the modules fully wired."""
        for key, mod in (("cereb", self.cerebellum), ("bg", self.bg), ("hippo", self.hippo)):
            sm = m.get(key + "_smask"); dens = m.get(key + "_density", 1.0)
            try:
                if sm and all(getattr(mod, n).shape == sm[n].shape for n in mod._synapse_matrices() if n in sm):
                    mod._smask = {n: v.to(self.dev) for n, v in sm.items()}
                    mod.syn_density = dens
                else:
                    mod._init_synapse_mask(dens if dens is not None else 1.0)
            except Exception:
                pass

    def _bound_logs(self):
        """Keep the on-disk logs bounded, like the replay buffer: truncate life.log to its most
        recent half when it passes max_log_mb, and delete the OLDEST tensorboard event files
        when the tb dir passes max_tb_mb (the active newest file is always kept). So a long
        experiment's logging stays within a fixed footprint."""
        try:
            cap = self.max_log_mb * 1e6
            if os.path.exists(self.logpath) and os.path.getsize(self.logpath) > cap:
                with open(self.logpath, "r", errors="replace") as f:
                    lines = f.readlines()
                keep, tot = [], 0                              # keep the most recent lines that fit 90% of cap
                for ln in reversed(lines):
                    tot += len(ln)
                    if tot > cap * 0.9:
                        break
                    keep.append(ln)
                keep.reverse()
                with open(self.logpath, "w") as f:
                    f.write(f"[log evicted to stay under {self.max_log_mb:.0f} MB — kept last {len(keep)} lines]\n")
                    f.writelines(keep)
        except Exception:
            pass
        try:
            tb = self._tb_dir
            if os.path.isdir(tb):
                evs = sorted((os.path.join(tb, f) for f in os.listdir(tb) if f.startswith("events.")),
                             key=os.path.getmtime)
                total = sum(os.path.getsize(p) for p in evs)
                while total > self.max_tb_mb * 1e6 and len(evs) > 1:   # keep the active (newest)
                    old = evs.pop(0); total -= os.path.getsize(old); os.remove(old)
        except Exception:
            pass

    def _ensure_tb(self):
        """Lazily open the TensorBoard writer (one event dir per life, under the run folder)."""
        if self._tb is None and self.use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(self._tb_dir)
                self.log(f"tensorboard logging → {self._tb_dir}")
            except Exception as e:
                self.use_tb = False
                self.log(f"tensorboard unavailable: {str(e)[:50]}")
        return self._tb

    @torch.no_grad()
    def _gen_quality(self):
        """Fluency proxies from a short sample: word-likeness (fraction of whitespace tokens
        that look like real words) + distinct-char ratio (catches degenerate collapse to a
        repeated character). Expensive-ish (one generation) → called on the slow tick."""
        try:
            txt = self.brain.generate("The ", n=160, temperature=0.7)[4:]
        except Exception:
            return None
        toks = [t for t in txt.split() if t]
        wl = sum(1 for t in toks if 2 <= len(t) <= 12 and t.isalpha()) / max(1, len(toks))
        dc = len(set(txt)) / max(1, len(txt))
        return wl, dc

    @torch.no_grad()
    def _weight_health(self):
        """Mean and max |weight| — a cheap tripwire for training blow-up or collapse."""
        try:
            tot, cnt, mx = 0.0, 0, 0.0
            for p in self.brain.parameters():
                a = p.detach().abs()
                tot += float(a.sum()); cnt += a.numel(); mx = max(mx, float(a.max()))
            return (tot / max(1, cnt), mx)
        except Exception:
            return None

    def _curiosity_stats(self):
        """§2 exploration health: how spread-out the topic policy is (entropy), its strongest
        preference, and the critic's mean value estimate."""
        if not self.modules_on:
            return None
        try:
            import torch as _t
            tf = self._topic_feat.to(self.bg.device)         # bg may live on its own device
            logits = tf @ self.bg.W_pi.t()
            pi = _t.softmax(logits, 1)
            ent = float(-(pi * (pi + 1e-9).log()).sum(1).mean())
            top = float(pi.max(1).values.mean())
            val = float((tf @ self.bg.w_v).mean())
            return ent, top, val
        except Exception:
            return None

    def _tb_log(self, understanding, time_sense, bpb):
        """Log EVERYTHING worth watching over a lifetime, grouped for TensorBoard. Step =
        seconds of life (a real time axis). Cheap scalars every ~5 s; expensive ones (a
        generation sample, a full weight sweep) on a ~30 s slow tick."""
        now = time.time()
        if not self.use_tb or now - self._last_tb < 5.0:
            return
        w = self._ensure_tb()
        if w is None:
            return
        self._last_tb = now
        step = int(now - self._t_start)
        neurons = self.brain.hidden
        phase_num = (0 if self.brain.age <= self.brain.grow_until else
                     1 if self.brain.age <= self.brain.prune_until else 2)
        mem = self.memory.stats()
        s = {  # --- capability: is it getting smarter? ---
             "eval/understanding_nextbyte_acc": understanding,
             "eval/bits_per_byte_probe": bpb,
             "eval/bits_per_byte_heldout": self.brain.bits_per_byte(HELDOUT_PROBE),
             "eval/time_sense_acc": time_sense,
               # --- growth / development (§10) ---
             "model/neurons": neurons,
             "model/size_gb": self.brain.model_gb(),
             "model/capacity_used_frac": self.brain.model_gb() / max(1e-6, self.max_model_gb),
             "model/age": self.brain.age,
             "model/phase": phase_num,
             "model/learning_rate_eta": self.brain.eta,
             "model/nights_slept": self.slept_count,
               # --- the living loop (§7-9) ---
             "life/thoughts_total": self.cycle,
             "life/thoughts_per_sec": self._tps,
             "life/awake": 1.0 if self.awake else 0.0,
             "life/awake_fraction": self._awake_frac,
             "life/sleep_debt": self.debt,
             "life/perceptions_total": self._perc_count,
             "life/clock_events": self.clock.events,
               # --- throughput / speed ---
             "speed/teacher_chars_per_s": self.teacher_cps,
             "speed/think_bytes_per_s": self.think_bps,
             "speed/learn_bytes_per_s": self.learn_bps,
               # --- memory system (§4, episodic) ---
             "memory/episodic_disk_mb": mem.get("disk_mb", 0),
             "memory/hot_mb": mem.get("hot_mb", 0),
             "memory/lived_chars_total": mem.get("lived_chars", 0),
             "memory/ssd_segments": mem.get("segments", 0),
             "memory/compression_ratio": mem.get("compression", 0)}
        if self.modules_on:                                  # --- the other four systems ---
            s["neuromod/novelty"] = getattr(self, "_novelty", 0.0)
            s["neuromod/dopamine"] = self.nm.tone["da"]
            s["neuromod/acetylcholine"] = self.nm.tone["ach"]
            s["neuromod/norepinephrine"] = self.nm.tone["ne"]
            s["neuromod/serotonin"] = self.nm.tone["ht"]
            s["memory/hippocampus_episodes"] = float(self.hippo.keys.shape[0])
            s["memory/hippocampus_dg_neurons"] = float(self.hippo.M)
            cur = self._curiosity_stats()
            if cur:
                s["curiosity/topic_policy_entropy"], s["curiosity/top_topic_pref"], s["curiosity/critic_value"] = cur
        # deeper 'state of the net' (perplexity / entropy / firing / recall / weights)
        nd = self._net_diag()
        for k in ("perplexity_train", "perplexity_gen", "gen_entropy", "spike_rate", "cerebellum_mse",
                  "bg_spike_rate", "hippo_spike_rate"):
            if nd.get(k) is not None:
                s["net/" + k] = nd[k]
        if nd.get("hippo_fidelity") is not None:
            s["net/hippocampus_recall_fidelity"] = nd["hippo_fidelity"]
        for k, v in nd.get("weights", {}).items():
            s["health/" + k] = v
        # expensive metrics on the slow tick
        if now - self._last_tb_slow > 30.0:
            self._last_tb_slow = now
            gq = self._gen_quality()
            if gq: s["eval/word_likeness"], s["eval/gen_distinct_char_ratio"] = gq
            wh = self._weight_health()
            if wh: s["health/weight_mean_abs"], s["health/weight_max_abs"] = wh
        try:
            for k, v in s.items():
                w.add_scalar(k, float(v), step)
            w.flush()
        except Exception:
            pass

    def _emit(self, on_update, force=False):
        now = time.time()
        if not force and now - self._last_emit < 0.35:
            return
        self._last_emit = now
        understanding, time_sense, bpb = self._metrics()
        nd = self._net_diag(); mem = self.memory.stats(); arch = self._arch_diag()
        res = self._resources()
        _now = time.time()                                    # thoughts/sec meter (from cycle deltas)
        if _now - getattr(self, "_last_cyc_t", _now) > 2.0:
            self._tps = self._ema(self._tps, (self.cycle - getattr(self, "_last_cyc", self.cycle)) / (_now - self._last_cyc_t))
            self._last_cyc, self._last_cyc_t = self.cycle, _now
        st = dict(state=("awake" if self.awake else "sleeping"), awake=self.awake,
                  cycle=self.cycle, awake_seconds=round(self._awake_seconds() if self.awake else 0),
                  min_awake=self.min_awake, max_awake=self.max_awake,
                  debt=round(self.debt, 1), debt_threshold=self.debt_threshold,
                  sleep_remaining=max(0, self.sleep_remaining), nights=self.slept_count,
                  phase=("child" if self.brain.age <= self.brain.grow_until else
                         "adolescent" if self.brain.age <= self.brain.prune_until else "adult"),
                  age=self.brain.age, eta=round(self.brain.eta, 5),
                  granules=(self.brain.hidden),
                  neurons=arch.get("total_neurons"), synapses=arch.get("total_synapses"),
                  parameters=arch.get("total_parameters"),
                  synapse_density=(round(self.brain.active_synapse_count() / max(1, self.brain.synapse_capacity()), 3)
                                   if hasattr(self.brain, "synapse_capacity") else None),
                  model_gb=round(self.brain.model_gb(), 3),
                  device=str(self.dev), dtype=str(self.dtype).replace("torch.", ""),
                  disk_gb=round(self.memory.stats()["disk_mb"] / 1000.0, 3),
                  clock=self.clock.tell(), clock_events=self.clock.events,
                  novelty=round(getattr(self, "_novelty", 0.0), 3),
                  da_tone=round(self.nm.tone["da"], 2) if self.modules_on else 0.0,
                  understanding=understanding, time_sense=time_sense, bpb=bpb,
                  teacher_cps=round(self.teacher_cps, 1), teacher_name=self.teacher_name,
                  think_bps=round(self.think_bps, 1), learn_bps=round(self.learn_bps, 1),
                  tps=round(self._tps, 1), perceptions=self._perc_count,
                  episodes=(self.hippo.keys.shape[0] if self.modules_on else 0),
                  lived_seconds=round(now - self._t_start),
                  perplexity_train=nd.get("perplexity_train"), perplexity_gen=nd.get("perplexity_gen"),
                  gen_entropy=nd.get("gen_entropy"), spike_rate=nd.get("spike_rate"),
                  cerebellum_mse=nd.get("cerebellum_mse"), bg_spike_rate=nd.get("bg_spike_rate"),
                  hippo_spike_rate=nd.get("hippo_spike_rate"), hippo_fidelity=nd.get("hippo_fidelity"),
                  bg_policy_entropy=nd.get("bg_policy_entropy"),
                  replay_total=getattr(self, "_replay_count", 0),
                  replay_mb=mem.get("disk_mb", 0), hot_mb=mem.get("hot_mb", 0),
                  lived_chars=mem.get("lived_chars", 0), segments=mem.get("segments", 0),
                  compression=mem.get("compression", 0), teach_queue=self._teach_q.qsize(),
                  net=nd, netparams=self._net_params(), arch=arch, resources=res,
                  ram_gb=res["ram"].get("process_gb"), ram_pct=res["ram"].get("pct"),
                  vram_gb=res["vram"].get("used_gb"), vram_pct=res["vram"].get("pct"),
                  storage_gb=res["storage"].get("run_total_gb"), disk_free_gb=res["storage"].get("disk_free_gb"),
                  observations=[{k: o[k] for k in ("i", "modality", "ts", "label")} for o in self.last_observations],
                  thought=self.thought, mind=self._mind_text(500),
                  feed=list(self.thought_log)[-50:],
                  ascii=image_to_ascii(self.last_screenshot) if self.last_screenshot else "")
        self.state = st
        self._tb_log(understanding, time_sense, bpb)      # watch improvement over time
        if on_update: on_update(st)

    # ---- checkpoint: save/resume the whole life ---------------------- #
    def save_life(self):
        try:
            self.brain.save(self.ckpt)
            life = dict(cycle=self.cycle, slept_count=self.slept_count, debt=self.debt,
                        awake=self.awake, clock_events=self.clock.events,
                        mind=list(self.mind), age=self.brain.age,
                        # metric meters + the life clock, so resume CONTINUES the curves
                        # (tensorboard step = seconds of life) instead of resetting to zero
                        lived_seconds=(time.time() - self._t_start),
                        perc_count=self._perc_count, awake_frac=self._awake_frac, tps=self._tps,
                        teacher_cps=self.teacher_cps, teacher_name=self.teacher_name,
                        think_bps=self.think_bps, learn_bps=self.learn_bps,
                        # LIVE config knobs — so a resumed life keeps how you tuned it
                        cfg=dict(budget=self.budget, min_awake=self.min_awake, max_awake=self.max_awake,
                                 debt_threshold=self.debt_threshold, max_sleep=self.max_sleep,
                                 perceive_gap=self.perceive_gap,
                                 think_chunk=self.think_chunk, learn_steps=self.learn_steps,
                                 sleep_chunks=getattr(self, "sleep_chunks", 10),
                                 sleep_steps=getattr(self, "sleep_steps", 14),
                                 sleep_seq=getattr(self, "sleep_seq", 96),
                                 resonate_k=self.resonate_k, grow_add=self.grow_add,
                                 grow_until=self.brain.grow_until, prune_until=self.brain.prune_until,
                                 freeze_growth=self.freeze_growth, freeze_sleep=self.freeze_sleep,
                                 freeze_learning=self.freeze_learning, use_visual=self.use_visual,
                                 max_log_mb=self.max_log_mb, max_tb_mb=self.max_tb_mb,
                                 hard_disk_gb=self.memory.hard / 1e9, think_temp=getattr(self, "think_temp", 0.6),
                                 feed_mode=self.feed_mode, focus_label=self.focus_label,
                                 topics=self.topics, browse=self.browse,
                                 novelty_gate=getattr(self, "novelty_gate", 0.15),
                                 net=self._net_params()))
            if self.modules_on:                              # §2/§4 state (curiosity + episodes)
                life["modules"] = dict(bg_wv=self.bg.w_v, bg_wpi=self.bg.W_pi, bg_M=self.bg.M,
                                       hippo_keys=self.hippo.keys, hippo_vals=self.hippo.vals,
                                       hippo_proj=self.hippo.proj, hippo_M=self.hippo.M,
                                       hippo_n_stored=self.hippo.n_stored,
                                       cereb_W=self.cerebellum.W, cereb_M=self.cerebellum.M,
                                       cereb_mse=self.cereb_mse,
                                       last_topic=self._last_topic, novelty=self._novelty,
                                       # §1/§2/§4 synapse masks + density (fixed-neuron/growing-synapse state)
                                       cereb_smask=getattr(self.cerebellum, "_smask", None),
                                       cereb_density=getattr(self.cerebellum, "syn_density", 1.0),
                                       bg_smask=getattr(self.bg, "_smask", None),
                                       bg_density=getattr(self.bg, "syn_density", 1.0),
                                       hippo_smask=getattr(self.hippo, "_smask", None),
                                       hippo_density=getattr(self.hippo, "syn_density", 1.0))
            torch.save(life, self.ckpt + ".life")
        except Exception as e:
            self.log(f"checkpoint failed: {str(e)[:50]}")

    def load_life(self):
        if os.path.exists(self.ckpt):
            self.brain.load(self.ckpt)
        p = self.ckpt + ".life"
        if os.path.exists(p):
            s = torch.load(p)
            self.cycle = s.get("cycle", 0); self.slept_count = s.get("slept_count", 0)
            self.debt = s.get("debt", 0.0); self.awake = s.get("awake", True)
            self.clock.events = s.get("clock_events", 0)
            self.mind = deque(s.get("mind", []), maxlen=self.mind.maxlen)
            # continue the metric curves: offset the life clock so tensorboard steps pick up
            # where they left off, and restore the throughput meters / counters
            self._t_start = time.time() - s.get("lived_seconds", 0.0)
            self._perc_count = s.get("perc_count", 0); self._awake_frac = s.get("awake_frac", 1.0)
            self._tps = s.get("tps", 0.0)
            self.teacher_cps = s.get("teacher_cps", 0.0); self.teacher_name = s.get("teacher_name", "—")
            self.think_bps = s.get("think_bps", 0.0); self.learn_bps = s.get("learn_bps", 0.0)
            self._last_cyc, self._last_cyc_t = self.cycle, time.time()
            c = s.get("cfg") or {}                            # restore how you tuned it
            for k in ("budget", "min_awake", "max_awake", "debt_threshold", "max_sleep", "perceive_gap",
                      "think_chunk", "learn_steps", "sleep_chunks", "sleep_steps", "sleep_seq",
                      "resonate_k", "grow_add", "freeze_growth",
                      "freeze_sleep", "freeze_learning", "use_visual", "max_log_mb", "max_tb_mb",
                      "think_temp", "feed_mode", "focus_label", "topics", "browse", "novelty_gate"):
                if k in c:
                    setattr(self, k, c[k])
            if "grow_until" in c: self.brain.grow_until = c["grow_until"]
            if "prune_until" in c: self.brain.prune_until = c["prune_until"]
            if "hard_disk_gb" in c: self.memory.hard = int(c["hard_disk_gb"] * 1e9)
            # set the wake tone FIRST, so a live-tuned neuromod tone restored via set_net below WINS
            # (the old order reset the tone to wake defaults AFTER set_net, clobbering the tuning).
            if self.modules_on:
                try: self.nm.set_phase("wake")
                except Exception: pass
            for tgt, ps in (c.get("net") or {}).items():      # per-module tuning
                try: self.set_net(tgt, ps)
                except Exception: pass
            m = s.get("modules")
            if self.modules_on and m:                        # restore §2 curiosity + §4 episodes
                # growth-INVARIANT guard: feat_dim (M cols) + n_actions (W_pi rows) never change under
                # neuron growth, so a GROWN bg still restores (the old exact-shape guard silently
                # discarded a grown bg's learned critic/actor — a persistence bug).
                if ("bg_M" in m and m["bg_M"].shape[1] == self.bg.M.shape[1]
                        and m["bg_wpi"].shape[0] == self.bg.W_pi.shape[0]):
                    self.bg.M = m["bg_M"].to(self.dev)
                    self.bg.w_v = m["bg_wv"].to(self.dev); self.bg.W_pi = m["bg_wpi"].to(self.dev)
                self.hippo.keys = m["hippo_keys"].to(self.dev); self.hippo.vals = m["hippo_vals"].to(self.dev)
                self.hippo.proj = m["hippo_proj"].to(self.dev); self.hippo.M = m["hippo_M"]
                self.hippo.n_stored = m.get("hippo_n_stored", int(self.hippo.keys.shape[0]))
                if "cereb_W" in m:                           # §1 forward-model weights
                    self.cerebellum.W = m["cereb_W"].to(self.dev); self.cerebellum.M = m["cereb_M"].to(self.dev)
                    self.cereb_mse = m.get("cereb_mse", 0.0)
                self._restore_module_masks(m)                # §1/§2/§4 synapse masks + density
                self._last_topic = m.get("last_topic", 0); self._novelty = m.get("novelty", 1.0)
            # resume AWAKE — restoring a mid-sleep state would drive sleep_remaining<0 on the
            # first tick and fabricate a spurious night + develop/grow (age++). Wake up fresh.
            # (wake tone already set above, before the set_net tuning restore.)
            self.awake = True; self.sleep_remaining = 0; self.wake_start = time.time(); self.debt = 0.0
            sz = self.brain.hidden
            self.log(f"RESUMED: cycle {self.cycle}, {self.slept_count} nights, age {self.brain.age}, "
                     f"size {sz}, model {self.brain.model_gb():.3f}GB")

    def stop(self):
        self._running = False

    def close(self):
        if self._browser is not None:
            try: self._browser.close()
            except Exception: pass
            self._browser = None
        if self._tb is not None:
            try: self._tb.flush(); self._tb.close()
            except Exception: pass
            self._tb = None
