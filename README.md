# The Living Brain

A standalone model built from the field guide *"The Learning Brain as a system of equations"*
that **lives**. It is born from a frozen small LLM, then **thinks continuously**, learns by
**talking to Claude (Sonnet 5)**, **browsing the web**, and **any tool or other AI you register**,
on an autonomous **wake/sleep** rhythm it runs itself — **growing** over a lifetime, with a
**sense of time**, inside fixed compute and memory budgets. Everything it does is **spiking** and
faithful to the five-system architecture of the field guide (§1 cerebellum, §2 basal ganglia, §3
cortex, §4 hippocampus, §5 neuromodulation).

You watch it, talk to it, and **drive it entirely** — teach it things, steer its attention, add
tools, tune any part of the net live — through **one web dashboard that is also a complete HTTP
API**. Runs on CPU or GPU.

```
 sight (web) ─┐                                            ┌─ write / speak
 language ────┤                                            │
 tools/AIs ───┼─▶ senses ─▶  FIVE SPIKING SYSTEMS  ─▶ motor ┼─ browse / act
 time (clock)─┤    one byte code   §1 §2 §3 §4 §5           │
 own voice ───┘   ("electricity")  cortex + 4 modules      └─ call a tool
   WAKE  babble → teachers teach → learn + browse   ·   SLEEP  replay + consolidate + downscale
   DEVELOP  fixed neurons · synapses grow → prune (§10)  ·  MEMORY  RAM·SSD·evict  ·  all watchable & steerable by API
```

---

## Install

Requires Python 3.10+. A conda env is recommended.

```bash
pip install -r requirements.txt          # core deps (torch, numpy, tensorboard, datasets, transformers, …)
# optional extras:
pip install textual rich                 # only for the terminal UI (interface/tui.py)
pip install playwright && playwright install chromium   # only for web browsing / vision
```

- The live **teacher** loop uses the **`claude` CLI** (Claude Code) on your `PATH`.
- The **birth teacher** is a cached **Qwen3.5-0.8B** (via `transformers`); skip it with `--no-teacher`.
- GPU: install a CUDA/ROCm build of `torch`. CPU works out of the box (`auto` uses all cores).

## Quick start

```bash
python interface/dashboard.py            # start the board + a fresh brain; prints the URL
```
It prints the URL, and on a remote/SSH host the **exact `ssh -L` tunnel** to reach it from your
laptop. Open the URL in a browser. Other entry points:

```bash
python interface/dashboard.py --checkpoint <run>   # resume a saved run (skips re-birth)
python interface/dashboard.py --no-autostart       # board only; launch the brain from the web form
python interface/dashboard.py --stop               # graceful stop (checkpoints), --kill to force
python run_life.py                                 # headless (same lifecycle; TensorBoard under runs/<run>/tb)
python interface/tui.py                             # terminal UI (needs textual + rich)
```

Viewing from a laptop over SSH (the dashboard also prints this itself):
```bash
ssh -L 8181:localhost:8181 user@host      # then open http://localhost:8181 locally
```

---

## The dashboard

One page, updating live (each panel has a **⛶ fullscreen** toggle):

- **Metric charts** — every metric as its own tile with real **X/Y axes + auto-scaled units**, a
  **Plotly-style hover tooltip** (time · value at the cursor), and **drag-to-reorder + resize**
  (your layout is remembered in the browser).
- **Status / config** — running state, device, and **every live hyperparameter**.
- **Net diagnostics** — perplexity, entropy, spike-rate, weight health, per-module state.
- **Stream of thought** — timestamped, modality-tagged.
- **Life log**, **vision** (ASCII), **perception**.
- **Chat box** — talk to it / inject feedback.
- **🎓 teach & steer** — teach content now, redirect the learning feed.
- **🔧 tools & other AIs** — register / run / install a CLI tool or another AI.
- **🎚 tune a module** — change any part of the net live.
- **👁👂 observations** — replay the audio it heard / see the images it saw.
- Header controls: **⚙ launch/config · 💾 save · ⏸ stop · ⛔ kill**.

## What it can do

- **Five spiking systems, operating together** (coupled by activation/reward, gradient-cut §6):
  **§3 cortex** (growable LIF, membrane readout; learns by DEFAULT by **faithful e-prop** — see the
  faithfulness stack below — with surrogate-BPTT kept as an opt-in, non-plausible capability reference),
  **§1 cerebellum** (fast supervised next-byte forward model, Golgi-controlled granule code,
  delta rule), **§2 basal ganglia** (dopamine actor-critic — curiosity), **§4 hippocampus**
  (modern-Hopfield episodic store — novelty-gated replay), **§5 neuromodulation** (wake/NREM/REM
  tone). Its performance/health is measured live (below).
- **Born, then lives** — Qwen distillation kick-in, then continuous thinking (~9 thoughts/s),
  **resonating k thoughts in parallel** (batched).
- **Learns from many teachers** — Claude Sonnet 5, the open web (reads by scrolling), and any
  tool/other-AI you register — all folded into one byte stream.
- **Autonomous wake/sleep** (it decides, min/max windows), NREM replay + SHY downscale.
- **Grows over a lifetime** — the way a brain actually develops: the **neuron count is fixed at
  birth** (a settable population — small or large), and it is the **synapses that evolve**.
  Childhood **synaptogenesis** densifies the connectome; adolescence **prunes** the weak synapses
  (mask-persistent). Set the initial neuron count + synapse density + per-region growth/prune rates,
  edit neurons/synapses per region or globally live (`/api/arch`), or freeze it.
- **Scales to large brains via a SPARSE cortex.** Above a threshold the O(neurons²) recurrence
  switches to a **CSR connectome** (int32 wiring + a dense value Parameter + a 1-D active mask), so
  the neuron count reaches the **hundreds of thousands within RAM** (memory is O(neurons·fan-in)). A
  custom autograd keeps the training backward O(nnz·B) — no dense-H² gradient — so a half-million-
  neuron brain is genuinely *trainable*, not just constructible. The small-net **dense path is
  byte-identical** and used below the threshold. On CPU the recurrent loop is slow at scale
  (`sparse.mm` is poorly threaded); it is built for **GPU** (`device=cuda`, optional bf16).
- **Every region on its own device.** Cortex on GPU, cerebellum on CPU, etc. — set live via
  `/api/device`; the living loop converts tensors at the boundaries.
- **Internal clock**, **unified byte-code senses** (text / image=retina / audio=cochlea / time /
  own-voice), **bounded memory** (RAM hot → zstd SSD → eviction).
- **Bounded logging/checkpoints** — log, TensorBoard, replay, and checkpoint all have live caps.
- **Checkpoint / resume** across devices (CPU ↔ GPU), continuing cycle, age, growth, clock, mind,
  and **all your live tuning**.

## Faithful learning — the biological stack (measured, not asserted)

The cortex learns by **e-prop by default** (forward-in-time eligibility traces + a top-down learning
signal + a three-factor neuromodulator gate — **no** backprop-through-time, **no** weight transport;
`learn_rule="bptt"` keeps surrogate-BPTT as an opt-in capability reference). On top of that, **every
biological constraint is an independent, live-toggleable axis** (`POST /api/net {target:cortex, …}` or
the dashboard's *faithfulness stack* panel), so each one's capability cost is *measured*:

| toggle | what it makes faithful |
|---|---|
| `feedback_mode` = `learned`\|`random` | learned (Kolen–Pollack) vs. fixed-random (DFA) top-down feedback |
| `two_compartment` | **unified apical circuit** — error runs *through* an apical dendrite, gated by a VIP→SOM disinhibition microcircuit driven by the neuromodulator, bursting onto somatic spikes (subsumes `dendritic`) |
| `dale` | Dale's law — each neuron excitatory or inhibitory (sign-locked outgoing synapses) |
| `dendritic` | standalone apical burst-coded error (not yet routed through the two-compartment neuron) |
| `bounded_synapses` | Fusi bounded weights (±`w_max`) |
| `homeostasis` | intrinsic firing-rate homeostasis (per-neuron threshold → target rate) |
| `btsp` | behavioral-timescale eligibility (the trace outlives the membrane) |
| `diff_neuromod` | the 4 tones gate 4 pathways — ACh→encoding, DA→reward, NE→gain, 5-HT→apical patience (couples learning to wake/sleep) |
| `stochastic` | probabilistic (noisy) spiking |
| `metabolic` | spike-rate energy penalty in the objective |

The effective learning rate is **self-adapting**, not a dial: it is `eprop_lr_scale · attention`, where
`attention` tracks the brain's own learning health (loss vs. a running baseline) — a loss spike drops it
so the update shrinks and the representation self-heals; a healthy loss raises it to engage (Yerkes–Dodson).
This is **scale-invariant by construction** (fan-in-normalized base + relative-loss attention), so the same
settings move from 16k to a million neurons without retuning. Sleep cycles NREM↔REM with replay depth set
by the day's novelty/debt. Each constraint surfaces a metric, and a suite of leading-indicator diagnostics
(`attention`, `eff_lr_scale`, `loss_ema`, `mem_mag` = representation magnitude / the true runaway signal,
`update_mag`, `grad_mag`, `surprise`, plus `fb_align_cos`, `ei_frac_excit`, `burst_frac`, `apical_mag`,
`homeo_thr_mean`, `synapse_sat_frac`) is exposed in `GET /api/state` → `net.weights`. The measured **capability-vs-fidelity curve**
— how many bits/byte each constraint costs — is `runs/fidelity_capability_curve.{py,md}`; the honest
cortical-algorithm survey + roadmap (real interneuron populations, STDP, prospective-configuration, ripple-gated
consolidation, DG neurogenesis, embodiment) is `runs/cortical_algorithm_research.md` and paper §15.17.

## Everything is API-driven

Every button on the board is one HTTP call (`GET /api/help` returns them with curl examples).
The endpoints:

| endpoint | what |
|---|---|
| `GET /api/state` | full live state: status, config, metrics, history, thought feed, logs, vision, per-part diagnostics |
| `GET /api/help` | list all endpoints + examples |
| `GET /api/runs` | list saved checkpoint folders (under `runs/`) |
| `GET /api/arch` | **per-region NEURON + SYNAPSE + PARAMETER census** (all 5 systems) + global totals + per-layer widths + density + per-region grow rate + device |
| `POST /api/device` `{target,device}` | place a region (or `all`) on its own **CPU/GPU** live; tensors convert at the cross-region boundaries |
| `GET /api/diag` | **one-call health check**: alive? learning (bpb/perplexity + recent trends)? all 5 systems firing? arch totals? + explicit warnings — built for driving by API without eyes |
| `GET /api/resources` | **RAM / VRAM / storage usage vs limits** (current + time-series): process & box RAM, per-device VRAM, checkpoint/replay/log/tb vs caps, disk free/total |
| `GET /api/value?key=PATH` | fetch **any** value from the live snapshot by dotted path (e.g. `resources.vram.used_gb`, `arch.parts.cortex.synapses`, `state.understanding`); omit key to list top-level keys |
| `GET /api/logs?n=N` | full-resolution recent log lines (up to 400; state gives last 120) |
| `GET /api/history?key=K` | full-resolution time-series for any metric K (omit key = every series) |
| `POST /api/chat` `{text}` | talk to it / inject non-blocking feedback |
| `POST /api/teach` `{text\|topic\|url\|path, label}` | teach specific content NOW (any byte stream: prose, a wiki topic, a web page, a file/dir of **code / music notation**) |
| `POST /api/focus` `{topics,urls,mode,label}` | redirect the learning feed to a target area (e.g. "learn coding") |
| `POST /api/set` `{…}` | change **global** hyperparameters live (see below) |
| `POST /api/net` `{target,…}` | tune a **part of the net** live: `cortex\|cerebellum\|hippocampus\|bg\|neuromod` |
| `POST /api/arch` `{target,op,amount?,density?}` | **live neuron/synapse surgery per region or `all`**: `grow_neurons`, `set_neurons` (→target), `grow_synapses`, `prune_synapses`, `set_synapses` (→target count/density), `refresh_synapses` |
| `POST /api/start` `{device,hidden,syn_density,layers,checkpoint,…}` | launch / relaunch (needed for device / initial size / core) |
| `POST /api/save` | force a checkpoint now (weights + full config) |
| `POST /api/stop` | graceful stop (checkpoints, exits) |
| `POST /api/kill` | force stop (no checkpoint) |
| `GET /api/tools` · `POST /api/tools/{add,install,run,toggle,remove}` | register / run / install a CLI tool or other AI |
| `GET /api/observations` · `GET /api/observe?i=N` | list + replay non-text observations (audio→wav, image→png) |

**Live-tunable via `/api/set`** (no restart): `budget, min_awake, max_awake, debt_threshold,
perceive_gap, think_chunk, learn_steps, resonate_k, threads, grow_add, grow_until, prune_until,
freeze_growth, freeze_sleep, freeze_learning, max_model_gb (=checkpoint cap), hard_disk_gb (replay
cap), max_log_mb, max_tb_mb, visual, teacher`. Only initial `hidden` (neuron population), `syn_density`,
`layers`, `device`, `core` need `/api/start` (which resumes the checkpoint, so nothing is lost).

**Deeper-brain R&D (§16, paper).** Three literature-grounded research threads + an integrated design live in
`runs/{memory_architecture,drive_stress,dynamics_oscillations}_research.md` + `runs/deeper_brain_integrated_design.md`:
memory should be *generative-in-the-net* not a raw buffer, plus a subcortical drive/cortisol layer and dynamic
oscillatory states. **P0–P2 built + verified** (all opt-in, live-tunable, metric'd, device/dtype-safe, persisted; default OFF):
- **P0 generative self-replay** — `sleep_mode=generative` dreams from the net, buffer-free (forgetting-resistance >
  raw-buffer, `runs/generative_replay_test.py`). `/api/set {sleep_mode, gr_dreams, gr_dream_len, gr_temperature,
  gr_anchor_frac}`; metrics `sleep_mode/gr_probe_drift/gr_dream_entropy`.
- **P1 endocrine** (`brain/endocrine.py`) — drive-deficit + cortisol + mood: satiation→reward→focus, cortisol
  inverted-U on plasticity, chronic-stress→allostatic-load→impaired→sleep-recovers (`runs/endocrine_test.py`).
  `/api/net {target:'endocrine', on, alpha_D, tau_C, C_star, …}`; metrics in `/api/state → netparams.endocrine`.
- **P2 dynamics** (`brain/dynamics.py`) — entropy knob β (normal↔psychedelic), **selective ignition** (not all
  systems active every cycle), attention→**processing frequency** (eligibility window). `/api/net {target:'dynamics',
  on, beta0, ignite_thr, …}`; metrics `beta/n_active/eff_freq` in `netparams.dynamics`.

**Faithfulness stack via `/api/net`** `{target:'cortex', …}` (no restart, each independent): `learn_rule,
feedback_mode, two_compartment, diff_neuromod, dale, dendritic, bounded_synapses, homeostasis, btsp,
stochastic, metabolic` + their hyperparameters (`eprop_lr_scale, fb_decay, burst_thr, w_max, target_rate,
homeo_lr, btsp_beta, g_ap, beta_ap, som_baseline, pv_gain, spike_noise, metabolic_lambda`). Read the live
settings back from `GET /api/state` → `netparams.cortex`.

**Per-module live via `/api/net`**: cortex `{lr, read_alpha, seq, think_temp, prune_frac, grow_syn_frac}`
· cerebellum `{eta, sparsity, g_golgi, thr0}` · hippocampus `{beta, sparsity, capacity, thr, g_inh}`
· bg `{alpha_v, alpha_pi, beta, thr}` · neuromod `{da, ach, ne, ht}`.

**Architecture surgery live via `/api/arch`** — neurons are the fixed population, synapses are what
evolve: `grow_synapses` (activate silent connections; `amount<1` = fraction), `prune_synapses`
(silence weakest; `amount` = fraction), `grow_neurons` (deliberately enlarge a part), `refresh_synapses`
(re-seed a part's connectome at `density`). Works on `cortex | cerebellum | bg | hippocampus`.

Examples:
```bash
B=http://localhost:8181
curl -XPOST $B/api/focus  -d '{"topics":["Python (programming language)","Algorithm"],"mode":"topics","label":"coding"}'
curl -XPOST $B/api/teach  -d '{"path":"/path/to/a/codebase","label":"code"}'
curl -XPOST $B/api/set    -d '{"resonate_k":8,"grow_add":128,"freeze_sleep":true}'
curl -XPOST $B/api/net    -d '{"target":"hippocampus","beta":12,"capacity":8000}'
# diagnose the whole brain in one call (no eyes on the dashboard needed):
curl $B/api/diag
# per-region neuron/synapse/parameter census (all 5 systems + global totals):
curl $B/api/arch
# grow synapses on one region, or set to a target density, or do it globally:
curl -XPOST $B/api/arch -d '{"target":"cortex","op":"grow_synapses","amount":0.2}'
curl -XPOST $B/api/arch -d '{"target":"cortex","op":"set_synapses","density":0.8}'
curl -XPOST $B/api/arch -d '{"target":"all","op":"grow_synapses","amount":0.1}'
curl -XPOST $B/api/arch -d '{"target":"bg","op":"set_neurons","amount":256}'
# per-region synapse grow rate + per-region device:
curl -XPOST $B/api/net    -d '{"target":"cerebellum","grow_syn_frac":0.25}'
curl -XPOST $B/api/device -d '{"target":"cortex","device":"cuda"}'
# watch the architecture evolve over time:
curl "$B/api/history?key=synapses"
# LARGE BRAIN on GPU (sparse cortex): e.g. 64k neurons, fan-in 64
python interface/dashboard.py --device cuda   # then POST /api/start with hidden=32000, sparse=true, rec_fanin=64
# register another AI and let the brain converse with it:
curl -XPOST $B/api/tools/add -d '{"name":"opencode","cmd":"opencode run {input}","kind":"text","install":"npm i -g opencode-ai","autonomous":true}'
curl -XPOST $B/api/tools/run -d '{"name":"opencode","input":"explain recursion simply"}'
```

## Metrics (live on the board + in TensorBoard)

`tensorboard --logdir runs/<run>/tb` — ~45 series across: **eval** (understanding, bits/byte on
probe + held-out, time-sense, word-likeness, generation entropy), **net** (train + generation
perplexity, spike-rate, per-layer weight health, cerebellum MSE, hippocampus recall fidelity, BG
policy entropy, BG/hippo spike rates), **model** (neurons=fixed population, **synapses=active +
density (what evolves)**, size, capacity, age, phase, learning-rate, nights, replays),
**life** (thoughts, thoughts/s, awake fraction, sleep debt, perceptions, clock), **speed**
(teacher / think / learn), **memory** (disk, hot, lived chars, segments, compression, episodes),
**neuromod** (novelty, DA/ACh/NE/5HT), **curiosity** (policy entropy, top preference, critic).

## Layout

```
run_life.py            headless runner (the one "run" file)
requirements.txt
interface/             the interfaces
  dashboard.py           unified web board + HTTP API (recommended)
  tui.py                 terminal UI
brain/                 the package — brain.life.BrainLife is the whole life
  life.py                the living loop
  spiking.py, spiking_brain.py, spiking_modules.py   the 5 spiking systems + substrate
  rnn_brain.py           rate byte-GRU cortex (--core rnn, the fluency reference)
  tools.py               the tool/plugin registry
  senses.py, motor.py, partner.py, llm_teacher.py, visual_web.py, memory.py, ascii_art.py, device.py, ops.py
paper/                 THE_LIVING_BRAIN.md (the §15 chapter) + field-guide.html/.pdf + render_pdf.py + figures/ + COVERAGE.md
runs/                  checkpoints (one folder per run: brain.pt, life.log, tb/, memory/, tools.json)
test/                  the test suite (run_tests.py + test_*.py)
```

## Tests

The whole codebase is covered by a fast, self-contained suite (no pytest needed — it mocks the
network and uses tiny CPU configs, so it runs in seconds):

```bash
python test/run_tests.py              # run everything (~85 tests)
python test/run_tests.py cortex       # only test_cortex.py
pytest test/                          # also works if you prefer pytest
```

What's covered: the spiking **substrate** (`run_seq` == stepping, identity-preserving growth,
surrogate gradient) · the **cortex** (learn / think / generate / resonate / grow / **synaptic
prune** / save-load / diagnostics) · the four other **systems** (cerebellum Golgi + delta rule,
basal-ganglia actor-critic learns a policy, hippocampus modern-Hopfield recall + grow, neuromod
tone) · **senses** + clock + **memory** eviction · the **tools** registry · the full **life**
(five systems, teach/focus, `use_tool`, `set_net`, observation replay, config persistence +
resume-awake, freezes, bounded logs) · and the **dashboard** Controller + the whole API surface
(help, snapshot shape, live-tune, teach/focus/tools/net/save, kill). Everything is exercised.

## Honest ceiling

The faithful spiking route is **fidelity first**: a strong *local-context* byte-level model (real
words, spelling, morphology, local fluency ~3 bits/byte) — **not** long-range coherence,
in-context learning, or reasoning, which exceed local credit assignment. `--core rnn` swaps in a
rate byte-GRU (~0.5 bits/byte, coherent sentences) when fluency is the priority. Multi-GPU is
selectable but currently uses one device (FSDP2 sharding is the documented next step, not yet
wired). See `paper/THE_LIVING_BRAIN.md` for the full architecture and honest ledger.

**Precision note:** bf16 is mixed-precision (fp32 master weights, bf16 compute) on both GPU and
CPU, but on a CPU without AVX512-BF16 it is *slower* than fp32 (emulated) — it's a GPU lever.

**Paper PDF:** `python paper/render_pdf.py paper/field-guide.html paper/field-guide.pdf` renders the
field guide (MathJax LaTeX and all) to PDF via headless Chromium — every equation typeset exactly
as on screen. Same command works for any MathJax/KaTeX HTML.
