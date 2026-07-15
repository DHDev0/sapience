# §15 · The Living Brain — from equations to a life

*A new chapter for "The Learning Brain as a system of equations." §14 turned the model on
for four static experiments. This chapter is the living system: one standalone brain, born
from a frozen teacher, that then thinks continuously, learns by talking to a stronger model
and by seeing the web, on a wake/sleep rhythm it runs itself, growing over a lifetime, with
a sense of time — inside fixed compute and memory budgets, watchable and interactive. It is
built to be faithful FIRST: five different coupled structures, exactly as §1–§5 describe, not
one monolithic network — four of them **growable spiking networks** (cortex, cerebellum,
hippocampus, and the basal-ganglia medium-spiny actor-critic), bound by a §5 neuromodulatory
glue (a diffuse scalar tone that gates their plasticity — modulator, not network). Every
mechanism is the field guide's; this chapter is the implementation and the honest ledger.*

Code: `sapience/brain/*.py`, `run_life.py`, `interface/{dashboard,tui}.py`. Runs on CPU or GPU.

---

## 15.1 Five spiking modules, not one network

The guide is emphatic (§11) that the brain is five *different* structures coupled by
activation but cut off from each other's gradients (§6). The implementation honours that —
each module is its own structure on a shared LIF substrate (`brain/spiking.py`), and each uses
the architecture that best mimics *its* region. Four are genuinely spiking and growable; §5 is
the neuromodulatory glue — a scalar tone, not a network:

| module | structure (region-specific) | teaching signal | file |
|---|---|---|---|
| §3 **Cortex** (spiking, growable) | stacked leaky-integrate-and-fire recurrent layers; **readout taps the analog membrane** of the top layer, not the binary spike | next-byte error via **e-prop** (forward-in-time eligibility + top-down signal + 3-factor gate; §15.17); BPTT (= β→0 PC) opt-in | `spiking_brain.py` |
| §1 **Cerebellum** (spiking, growable) | sparse LIF granule expansion → Purkinje readout, with a **Golgi feedback-inhibition loop** holding granule sparsity at target | climbing-fibre error, delta rule ΔW=−η·c·g | `spiking_modules.py` |
| §2 **Basal ganglia** (spiking, growable) | LIF **medium-spiny neurons** → critic + softmax actor; the dopamine RPE trains *both* (policy gradient) | dopamine RPE δ = r + γV′ − V | `spiking_modules.py` |
| §4 **Hippocampus** (spiking, growable) | sparse dentate-gyrus separation + a modern-Hopfield store with **spiking CA3 winner-take-all** recall (LIF memory neurons + lateral inhibition → CA1 read-out) | one-shot, novelty-gated write; pattern completion | `spiking_modules.py` |
| §5 **Neuromod. glue** (scalar) | ACh/NE/DA/5HT tone as the three-factor gate M(t); the **ACh tone scales the cortex learning rate** per phase | Δw = η·M(t)·e | `spiking_modules.py` |

The cortex is the deep learner that carries language; the cerebellum corrects, the
hippocampus stores/recalls, the basal ganglia supplies reward/curiosity, the neuromodulators
gate plasticity and set the wake/sleep tone (§7.2). Spikes and reward scalars cross the
boundaries; gradients do not (§6). Unlike an earlier version where these four were built but
**orphaned** (only the cortex ran), they are now wired into the living loop (§15.14) so the
brain actually *operates* as five systems.

## 15.2 The spiking substrate

`brain/spiking.py`. Leaky integrate-and-fire neurons: `v ← β·v·(1−s) + Wx + Rs`, fire when
`v > θ`. The Heaviside spike is non-differentiable, so the *pseudo-derivative* used by both
learning rules is a **surrogate gradient** (fast-sigmoid). By default the cortex learns by
**e-prop** (§15.17) — forward-in-time, local, no backward pass; the opt-in `learn_rule="bptt"`
reference is surrogate-BPTT, which §3.5 identifies as predictive coding in the β→0 limit, so even
the non-plausible reference stays inside the paper's framework while the architecture is genuinely
spiking (membranes, thresholds, spikes). The cortex's `TwoCompartmentLIF` implements §3.7 directly:
a somatic compartment (basal feedforward + recurrent drive) and an apical dendrite carrying the
top-down error that modulates firing — the substrate the §15.17 dendritic/burst-error toggle uses.

**Membrane readout (a measured 2× win).** The original cortex applied its readout head to the
binary *spike* vector — ~1 bit/neuron, discarding the analog membrane the neuron actually
integrates. A controlled A/B (wikitext-2, identical seed/data/budget) showed that reading the
graded top-layer **membrane** instead nearly doubles next-byte accuracy (0.17 → 0.34) and
drops bits/byte 4.48 → 3.39 — at **zero** faithfulness cost, because the internal computation
(inter-layer signals, recurrence) stays entirely spike-based; only the final observation taps
the graded population code (as downstream cortex reads rates/LFP, not single spikes). This is
now the default. A spike/tanh(membrane) *mix* and a squashed membrane were also tried and are
no better; the plain membrane wins.

**Two documented, off-by-default cortex levers.** (i) `ALIFCell` — an adaptive-threshold LIF
(spike-frequency adaptation, the LSNN long-timescale working memory). It is built and correct,
but a four-arm A/B found it *loses* at the short byte-context this model uses (seq 48–128),
where there is no long-range dependency for its slow state to exploit; it is kept for future
long-horizon tasks. (ii) The §3.7 two-compartment top-down apical: faithful hierarchical
top-down prediction requires stepping time-outer (layer-within-timestep), which would forfeit
the vectorised speed-up below; since the membrane readout already supplies the graded-potential
benefit, the two-compartment cell remains in the substrate for the §3.7 exposition but is not
the running default. Both are honest trade-offs, not omissions.

**A ~7× speed-up, bit-identical.** The per-timestep Python loop originally recomputed the input
projection `Win(x)` and the readout head every step. Because a feedforward LIF stack has no
within-timestep top-down, the layers can run one-at-a-time over the whole sequence
(`LIFCell.run_seq`): the input projection is one matmul over all T, the head is one matmul over
all T, and only the genuinely-recurrent `Wrec(s)` stays in the loop. This is *mathematically
identical* to stepping (verified max-diff 0.0) and cuts a training call from ~1.6–3.4 s to
~0.24 s. Growth stays identity-preserving under it.

## 15.3 Development: fixed neurons, evolving synapses (§10)

Development follows the biology: **the neuron count is largely set at birth** (neurogenesis is
~complete), so it is a *fixed, settable population* — small for a laptop, or scaled up toward the
100k–1M range the field guide imagines. What **evolves over the lifetime is the synaptic
connectome**. The brain is born with a sparse connectome (a settable `syn_density`, default 0.5 of
possible connections active); each night of the **child** phase runs **synaptogenesis** —
`grow_synapses` activates a fraction of the currently-silent connections with fresh small weights,
densifying the wiring — and each night of the **adolescent** phase runs **synaptic pruning** —
`prune` silences the weakest active connections, held at zero by a persistent mask so the pruning
sticks. The **adult** does slow turnover. So the *synapse count* rises then settles while the
*neuron count stays fixed*, exactly as synaptic density peaks in childhood and is pruned in
adolescence — all riding the critical-period learning-rate envelope η(t)=η_max/(1+t/t_c) (§10.4).

Neurons can still be grown *deliberately* — `grow_neurons(add)` inserts LIF units whose *outgoing*
weights are zero, so the network computes the same function the instant capacity appears (verified:
post-grow bits/byte identical to pre-grow), and the synapse mask is padded identity-safely. This is
the explicit lever (via `/api/arch`, or a large developmental step) to enlarge the population when
there is real experience to justify it — capped at `max_model_gb`. Every part of the architecture
(cortex, cerebellum, basal ganglia, hippocampus) reports its live **neuron and synapse counts**
(`/api/arch`), and all of it — grow/prune synapses, grow neurons, re-seed a connectome — is
editable live per part by API.

## 15.4 The capability fork, chosen honestly

The design study established a hard truth: a purely local-rule model over raw bytes learns
byte statistics, not language. Two faithful ways forward, both grounded in §3.5:
- **Spiking, five-module, growable (chosen — fidelity first):** the architecture above, learning
  by default via **faithful e-prop** (§15.17). Five distinct structures, four genuinely spiking +
  growable (cortex, cerebellum, hippocampus, basal ganglia) plus the §5 neuromodulatory glue. With the membrane readout
  and a longer context window its capability has improved from the original ~4 bits/byte to
  **~3.1–3.2 bits/byte** (born at ~3.3, evolving down over its first nights), with real
  word-like fragments emerging (*"…was from the … to the … and …"*). Still modest versus a
  rate GRU — the accepted cost of fidelity — but markedly better than before, and it keeps
  improving as it lives.
- **Rate recurrent cortex (kept as the capability reference):** a byte GRU trained by plain
  backprop (= β→0 PC). It reaches ~0.5 bits/byte and writes coherent sentences (*"The
  ocean is … He also added with the starter Cullen …"*), but it is not spiking and cannot grow
  cleanly. Available as `--core rnn` for when capability is the priority. `brain/rnn_brain.py`.

The default is the spiking core; the fork is a one-line switch, and the paper states plainly
which fidelity/fluency point each occupies.

## 15.5 Birth — the distillation kick-in

`life.birth`. Before it lives, the cortex is trained to competence on a **frozen Qwen3.5-0.8B**
teacher's generated outputs (`brain/llm_teacher.py`) plus real text (wikitext-2) — the innate
bootstrap. (This birth teacher is distinct from the in-*life* teacher it later converses with,
Claude Sonnet 5 via `brain/partner.py`; the UI even labels the two, `teacher_name` flipping
Qwen→Sonnet at birth's end.) It is *born able to write* (the rate cortex reaches coherent English at birth; the spiking cortex reaches its
modest ceiling faster), then the life evolves it. This front-loads the "fluency needs
wall-time" cost into a one-time developmental window, as a genome + rapid early learning do.
Birth stops adaptively — on the target bits/byte, OR on a **learning plateau** (no >0.01
bits/byte gain for six epochs), OR a wall-clock cap — because the spiking core floors near
~3 bits/byte and never reaches the rate-cortex target, so without the plateau stop it would
burn every epoch; this cut birth from ~220 s to ~150 s.

## 15.6 One code for every sense

`brain/senses.py`. Sight, language, time and the brain's own voice become one stream of
byte-levels 0–255 ("electricity"), each frame tagged by a "nerve" marker — the unified
sensorimotor code. Motor output (`brain/motor.py`) is the same code in reverse: writing/
speaking = emitting bytes decoded to text; navigating = emitting a command. Vision enters via
a headless browser (`brain/visual_web.py`) that navigates, screenshots and SCROLLS ("reads by
scrolling"); its *readable text* is what the language cortex learns, its screenshot is shown as
ASCII — raw pixels are not force-fed into the language mind-state (that would be noise).

## 15.7 The internal clock (a sense of time)

`senses.Clock`. A pacemaker-accumulator plus banks of oscillators at many periods (the
striatal beat-frequency model + circadian rhythm) encodes elapsed time to byte-levels and
STAMPS every perception — the fastest of §0's clocks, made a sensory channel. The brain always
knows what time it is; a fixed time-probe measures how well it has learned to tell it.

## 15.8 Autonomous wake/sleep, and sleep that learns

`brain/life.py`. Sleep debt (§7.6) rises as the brain learns while awake; the brain DECIDES
when to sleep — at least `min_awake`, by `max_awake` at the latest, and in between once the
debt crosses a threshold. Sleep is the deep-learning phase, as in biology: it replays the
whole-life memory heavily under the default learning rule (faithful e-prop, §15.17; waking
thought stays fast while sleep does the consolidation) (§15.10), then applies a §8.4 SHY
multiplicative downscale so it
also renormalises rather than only potentiating. (Honestly this is an *open-loop* constant
downscale — a small fixed factor over all weights — not yet a closed-loop restore of the §9
balance ⟨Δw⟩_wake + ⟨Δw⟩_sleep ≈ 0 scaled to the night's actual potentiation; that, and a
protection term for replayed synapses, are named as the next refinement.) Moving deep
consolidation into sleep more than doubled the
learning rate while keeping waking thought fast (~9 thoughts/s). Each night the brain develops
(§15.3).

## 15.9 Continuous thinking

The brain is never idle: a fast inner-monologue loop generates the next fragment of thought
from the persistent spiking mind-state and streams it; a background sense-thread fetches
Claude's teaching and web pages without ever pausing the thinking; your typed input is
**non-blocking feedback** woven into the same stream (weighted stronger than passive input —
more learning steps, tagged "you say:"), with in-band commands (`browse <url>`/`look <url>`
redirect vision, `?time` queries the clock); stop and it keeps learning on its own. Every
teacher reply is screened first (`_usable_lesson` rejects CLI-error strings and stubs) so the
brain never distils *"Error: exceeded budget"* as if it were language — a small but load-bearing
robustness guard on the core learning path.

**Resonating in parallel.** Because the spiking substrate batches over streams, the brain can
run *k* thought-streams at once in a single forward pass — `SpikingBrain.resonate(k)` — for
essentially the wall-time of one (6 streams measured at 9 ms, the same as one). Every few
seconds the living loop resonates, keeps the stream it itself finds most fluent (lowest
self-perplexity), and lets that reflection re-enter its mind — parallel exploration collapsing
to a better single thought, and it learns from its own best thinking (§9). The teaching
conversation with Claude Sonnet 5 runs through `brain/partner.py` (`claude_say` shells the
`claude` CLI for a teacher paragraph; `web_topic` pulls clean Wikipedia extracts), verified to
feed this same learning path end-to-end: a real lesson measurably lowers the cortex's bits/byte
on that lesson. Every teacher reply is screened first (`_usable_lesson`) so a CLI error string
is never distilled as language.

## 15.10 Memory hierarchy — RAM · SSD · eviction

`brain/memory.py`. The whole life of experienced text is tiered within fixed budgets: RAM
holds the recent hot buffer; at the RAM/soft limit the oldest hot text is flushed and
**compressed** to an SSD segment (zstd-19, ~100×); at the global/hard disk limit the **oldest
compressed segments are deleted**. Sleep replays random chunks from hot and a decompressed old
segment, so the brain rehearses its whole life. Text compresses enormously, so a long life
fits in a small, bounded footprint. Model parameters are bounded by `max_model_gb`.

## 15.11 Device, precision, scale

`life.pick_device`. Auto CPU/GPU; **bf16 mixed precision (fp32 master weights, matmuls autocast
to bf16) on GPU AND CPU** — bf16 *parameters* would wreck Adam's running averages, so the weights
stay fp32 and only the compute casts. bf16 on CPU is honoured when asked for, with the honest
caveat that without AVX512-BF16 it is *slower* than fp32 (emulated) — it is a GPU-throughput lever,
not a CPU one. The checkpoint is device-agnostic (start on CPU, resume on GPU), and — new — **each
of the five systems can run on its own device** (`set_part_device` / `POST /api/device`): the
living loop converts tensors at every cross-region boundary, so a large cortex can sit on the GPU
while a module stays on CPU. Multi-GPU FSDP2 sharding remains the documented next step.

**Scaling to a large brain — the sparse cortex.** A dense recurrent cortex is O(neurons²): at
500 000 neurons the recurrent matrix alone is ~750 GB — impossible. Above a threshold each cortical
layer therefore switches to a **CSR sparse connectome** — int32 wiring buffers + a dense value
`Parameter` + a 1-D active-synapse mask — so memory is O(neurons·fan-in) and the neuron count can
reach the hundreds of thousands within RAM. The subtle part is *training*: `torch.sparse.mm`'s own
backward transiently densifies a full neurons² gradient (measured ~4 GB/layer at 64 k, ~250 GB at
500 k → OOM). A custom autograd (`_SparseConnMM`) computes the value-gradient by a sampled
gather/scatter instead — **O(nnz·B), no neurons² transient, verified bit-exact against the dense
scatter** — so a half-million-neuron brain is genuinely *trainable*, not merely constructible. The
small-net **dense path stays byte-identical** and is used below the threshold, so nothing about the
default behaviour (or the test suite) changes. All four growable systems are sparse; the cortex is
the only one that needs CSR (the modules are neurons×const, i.e. linear, so a masked dense store is
already efficient). Honest ceiling: on CPU the recurrent per-timestep `sparse.mm` is poorly
threaded, so at 64 k a learn step is tens of seconds — the sparse cortex is built for **GPU**, where
that loop is fast; on CPU the trainable-in-real-time sweet spot stays a few-thousand-neuron dense net.

## 15.12 Checkpoint and resume a life

`life.save_life/load_life`. A life is checkpointed on night boundaries (not per step, which
would stall at scale) plus a time fallback: weights + optimiser, and the life state — cycle,
age, nights, clock, wake/sleep, mind. `--resume` continues exactly; the compressed episodic
archive persists on disk.

## 15.13 Running it

- `python interface/dashboard.py` — the unified web board + HTTP API (recommended; §15.15). One
  URL: live charts, thought stream, log, vision, chat, status/config, and launch/stop/kill.
- `python run_life.py` — headless (same life; TensorBoard under `runs/<run>/tb`).
- `python interface/tui.py` — an opencode-style terminal interface. `--core spiking|rnn`,
  `--checkpoint <run>` to resume.
- `python test/run_tests.py` — the full self-contained test suite (every module + the full
  life + the dashboard API), 54 tests in seconds, no pytest required.

## 15.14 The five systems, actually coupled

An earlier version built the four non-cortex modules but never ran them — only the cortex
lived. They are now wired into the loop, coupled by **activation and reward only, never
gradients** (§6), at *zero* throughput cost (measured 1.00× with the modules on):

- **§2 basal ganglia → curiosity.** A spiking LIF **medium-spiny** actor-critic (growable — its
  neurons grow alongside the cortex) chooses *which topic* the brain asks Claude about next; its
  dopamine reward is the **learning progress** on the lesson — the drop
  in training loss, already computed during learning, so the reward is free. Over time the
  brain prefers the topics that teach its cortex the most. (Fixing the frozen-actor bug — the
  old `train_step` updated only the critic — was what made this possible; on a contextual
  bandit the fixed version learns the optimal policy, 0.33 → 0.98, while the old one stays at
  chance.)
- **§4 hippocampus → complementary learning systems.** Every experience is fingerprinted; its
  **novelty** (one minus the best recall match) both **gates whether it is written at all**
  (familiar patterns are not re-stored, so the buffer keeps the novel ones — encode without
  overwriting, §7.4) and sets how hard sleep replays it — a novel life consolidates harder. The
  hippocampus grows its dentate gyrus alongside the cortex each night. The store is a spiking
  modern-Hopfield associative memory: measured pattern-completion fidelity 1.00 → 0.96 as the
  number of stored memories goes 10 → 400, where the old covariance-Hopfield version collapsed
  and could not even reconstruct the pattern (it returned only a DG code) and had no `grow()`
  at all — a §10 growability violation now fixed.
- **§5 neuromodulation → plasticity gate.** The ACh/NE/DA/5HT tone is set on every wake↔sleep
  transition, and — this is the real coupling, not just a readout — the **ACh tone scales the
  cortex's effective learning rate** on every wake learning step (the three-factor gate M(t) in
  Δw = η·M(t)·e), so waking encodes at full plasticity and you can throttle it live via
  `/api/net`. (Only wake and NREM tones are entered; a REM sub-phase is named as roadmap, not
  claimed as running.)
- **§1 cerebellum → a fast supervised forward model.** Alongside the slow e-prop cortex it runs a
  *fast* supervised next-byte predictor, learned by its own climbing-fibre delta rule and
  gradient-cut from the cortex (§6): a bag-of-bytes window is the mossy input, a Golgi-controlled
  sparse granule expansion the hidden code, a Purkinje readout the prediction. Its error
  (`cerebellum_mse`) is tracked live. It complements the cortex (a fast vs slow learner, as in
  the cerebellar–cortical division of labour) and is a monitored forward model, not yet fed back
  into generation — the honest state of its integration.

On a homogeneous mocked feed the modules are within run-to-run noise on the cortex's own
language metric (the mock ignores which topic was chosen, so curiosity cannot differentiate);
their measured value is that they are now **free** (no regression) and **faithful** — all five
are instantiated and running concurrently, coupled via reward (§2), tone/plasticity-gate (§5),
novelty-gated replay (§4) and the cerebellar forward model (§1), where an earlier version had
never even instantiated the cerebellum. (What is *not* yet built is the §6 router that would
fuse the modules' competing outputs by reliability — the cortex alone drives generation and the
cerebellum's prediction is monitored, not fed back; that fusion is named as the next step.)
Their real payoff shows on heterogeneous real-world input. Checkpoint/resume persists **all five modules' state and every live-tuned
knob**, and a latent resume-after-growth bug (the loader
assumed uniform layer width, but growth widens only the top layer) was found and fixed —
resume is now byte-identical after arbitrary growth.

## 15.15 The cockpit — driving and extending the brain by API

A living thing you cannot watch or steer is a black box. The implementation therefore ships a
single **web control-plane** (`dashboard.py`) that is also a complete HTTP API — *every* action
the browser can take is one documented call (`GET /api/help` lists them), so the brain can be
driven entirely programmatically. One command (`python interface/dashboard.py`) serves, on one URL:
live metric charts, the timestamped stream of thought, the life log, the ASCII vision, a chat
box, a full status/hyperparameter panel, and controls. On a remote host it prints the exact
`ssh -L` tunnel to reach it. It supersedes the separate `tail -f` + TensorBoard + TUI (though
TensorBoard still logs **~46 metrics** — capability, growth, wake/sleep, the five modules,
curiosity, memory, throughput, weight-health — for deep, zoomable history).

- **Lifecycle & compute.** Launch, **graceful stop** (a STOP control file → it checkpoints and
  exits; no `kill -9`), or force-kill — from the board or the CLI. A run with no checkpoint gets
  a fresh timestamped folder (old runs untouched); `--checkpoint` continues one in place, and
  because the checkpoint is device-agnostic you can start on CPU, stop, and resume on GPU or
  multi-GPU. An **`auto`** compute mode maximises the hardware (every GPU, else all CPU cores);
  parallelism (CPU threads, the width *k* of parallel thought-resonance) is a knob. Resume
  skips re-birth — it is already competent.
- **Everything is live-tunable.** The design principle is that *any* parameter that can change
  without rebuilding the tensors changes **live**, no restart — the loop reads them each
  iteration, so a `POST /api/set` propagates on the next tick. That covers the global knobs
  (budget, wake/sleep windows, perceive-gap, think-chunk, learn-steps, resonance width, threads),
  the **development / cycle controls** (per-night growth amount, grow/prune ages, and hard
  toggles to *freeze growth*, *freeze the wake/sleep cycle* so it stays awake, or *freeze
  learning* so it observes without updating weights), and every **storage cap** below. Only the
  genuinely structural choices — initial neuron count and layer depth, device, core — need a
  `POST /api/start` relaunch (which resumes the checkpoint, so nothing is lost). Individual
  **parts of the net** are tuned through `POST /api/net` (per region, or `target:"all"` at once):
  the cortex's learning rate / readout mix / context length / sampling temperature, the
  hippocampus's recall sharpness / sparsity / capacity, the basal ganglia's actor & critic rates,
  the neuromodulator tones, and **each region's own synaptic growth/prune rate**. (On development:
  the *neuron count is fixed at birth* — a settable population — and it is the *synapses* that
  evolve; you set the initial count, the initial synapse density, and per-region grow/prune rates,
  or freeze it. See §15.3.)
- **Architecture census & live surgery.** `GET /api/arch` reports, per region **and** globally,
  the neuron count, active-synapse count, parameter count, synapse density, per-layer widths, the
  per-region grow rate, and the region's device. `POST /api/arch` performs live surgery on any
  region (or `all`): grow neurons, **set** neurons to a target, grow/prune synapses, **set**
  synapses to a target count or density, or re-seed a connectome — so the living architecture is
  both fully observable and directly editable while it runs. `POST /api/device` places any region
  on its own CPU/GPU (tensors convert at the boundaries).
- **Resource budget, watched against its limits.** `GET /api/resources` (and time-series charts +
  a usage-bar panel) report RAM (this process **and** the box), VRAM per device, and every bounded
  store — checkpoint, replay, log, TensorBoard — as *current vs cap*, plus disk free/total. A
  single `GET /api/diag` is a one-call health check (alive? learning, with recent trends? all five
  systems firing? + explicit warnings), and `GET /api/value?key=<dotted.path>` fetches *any* value
  from the live snapshot — all built so the brain can be driven and diagnosed with no eyes on the
  dashboard at all.
- **Deeper diagnostics (state of the net).** Beyond capability, the board and TensorBoard track
  the *internal* state: perplexity on what it trains on **and** on what it generates, the entropy
  of its output distribution, the mean spiking rate (dead-neuron / saturation tripwire),
  per-layer weight magnitude and spread, the cerebellum's forward-model error, the basal-ganglia
  policy entropy, and the hippocampus's live recall fidelity — so you can see not just *whether*
  it is learning but *how healthy each of the five systems is* while it does. And because every
  live-tuned knob and each module's state is checkpointed, a resumed life continues exactly how
  you left it — same tuning, same growth schedule, same focus.
- **Bounded everything.** Like the replay buffer (§15.10), all on-disk artifacts stay within
  live-settable caps that evict the earliest data when full: the life-log truncates to its most
  recent lines, the TensorBoard event dir deletes its oldest files, the replay buffer its oldest
  segments, and the checkpoint is a single overwritten file bounded by the model-size cap — so a
  long experiment's footprint is fixed and known.
- **Directed teaching & attention.** `POST /api/teach` folds specific content in *now* — raw
  text, a wiki topic, a web page, or a local **file/dir of code, music notation, or prose** (it
  learns any byte stream, so its own source code is a valid lesson, verified). `POST /api/focus`
  redirects the learning feed from broad/random to a target area (topics, urls, mode), so you
  can say "learn music" or "learn coding" and the curiosity loop curates toward it.
- **Tools & other minds (the unified code, extended).** `POST /api/tools/add` registers any CLI
  tool or other AI — *opencode* with a free model, a local LLM, a text-to-speech or image
  command — as a template with an `{input}` slot and an output **modality**. The brain converses
  with it (autonomously or on demand), and its output is folded into the ONE byte stream by
  §15.6's encoders: text is learned as language; audio becomes cochlea bytes, an image retina
  bytes, anything else raw levels. This is the field guide's thesis made operational — *any*
  sense or tool, or a combination, becomes "electricity" the brain learns from — and it is now
  extensible at runtime rather than fixed at build time. Each tool carries an optional install
  command (`POST /api/tools/install`) and enable / manual↔autonomous toggles
  (`POST /api/tools/toggle`); specs persist with the checkpoint.
- **Replaying what it sensed.** Non-text observations (an audio clip, an image a tool produced)
  are kept as their encoded byte frames and can be **reconstructed back into playable/viewable
  media** (`GET /api/observe`) — a deliberately low-fidelity rendering, because it decodes
  exactly the down-sampled, quantised "electricity" the brain actually received, not the
  original file. On the board you hear the cochlea bytes and see the retina bytes; in the thought
  feed each perception is tagged with its modality. It is the closest thing to looking through
  the brain's own senses.

The point is not the UI; it is that the whole living system — its rhythm, its diet, its senses,
the minds it talks to — is inspectable and steerable while it runs, by a human at a glance or by
another program through the API.

## 15.16 The honest ledger

What is real, on CPU: five different structures (§1–§5) — four growable spiking networks
(cortex, cerebellum, hippocampus, and the basal-ganglia LIF medium-spiny actor-critic) plus a
scalar neuromodulatory glue — coupled by the gradient cut and now **actually operating
together** (§15.14), the neuromodulator ACh tone genuinely gating cortical plasticity; a spiking
cortex that learns by DEFAULT by **faithful e-prop** (forward-in-time eligibility + a top-down
learning signal + the three-factor gate; no BPTT, no weight transport — §15.17), with surrogate-BPTT
(= β→0 PC) kept as an opt-in capability reference, reads its graded membrane, runs ~7× faster after a
bit-identical vectorisation, and grows by §10 synaptogenesis with the function preserved; parallel thought
resonance; a unified byte code for every sense; an internal clock; an autonomous,
homeostatically-balanced, learning-during-sleep 24-hour loop with novelty-gated replay and
dopamine-driven curiosity; continuous thinking with non-blocking human feedback; a compressed,
evicting, whole-life memory in fixed RAM/SSD budgets; checkpoint/resume (byte-identical after
growth); born-competent via teacher distillation, then learning from a verified Claude Sonnet 5
conversation; the full §10 developmental arc is real — a **fixed neuron population** with an
**evolving synaptic connectome**: the child runs **synaptogenesis** (activating silent connections,
densifying the wiring), the adolescent **prunes** the weakest synapses (mask-persistent), the adult
stabilises — with deliberate neuron growth available on demand; and all five systems are genuinely
instantiated and running, including the §1 cerebellum as a monitored fast forward model. The whole living system is watchable and
drivable through one web board that is also a complete API (§15.15): launch/stop/resume across
devices, live hyperparameter tuning, directed teaching, learning-feed steering, and
runtime-registered tools / other AIs whose any-modality output folds into the same byte code.
The chosen spiking route is faithful FIRST, so its fluency is still modest —
improved from ~4 to **~3.1–3.2 bits/byte** by the membrane readout and longer context, with
real word-like fragments — with the rate-cortex reference (`--core rnn`, ~0.5 bits/byte,
coherent sentences) one switch away when fluency is wanted. Every architecture choice here was
settled by a controlled A/B, and the ones that *lost* (adaptive-threshold ALIF at short
context, the spike/membrane mix, extra depth) are documented as such, not hidden. Neither route
yet has long-range coherence, in-context learning, or reasoning — those exceed local credit
assignment and are named, not oversold, as the next chapter: a deeper temporal spiking trunk
and the two-compartment top-down path (which needs the slower time-outer schedule). The step
§13 called last — instantiate the whole thing and watch it live — is taken here.

## 15.17 The faithfulness stack, and its measured price

The cortex no longer learns by backprop-in-a-costume. **e-prop is now the default rule** — a
forward-in-time approximation to BPTT (Bellec et al. 2020): a per-synapse *eligibility trace* is
kept online, a *top-down learning signal* gates it, and a *three-factor* neuromodulator term
(the ACh tone, §15.14) opens or closes plasticity. There is no backward pass through time, no
weight transport, no separate error network — the update is `@torch.no_grad`, computed by hand
from local traces and applied with `w.add_`. Surrogate-BPTT + Adam is kept only as the opt-in,
non-plausible **capability reference** (`learn_rule="bptt"`). The single non-local operation that
remained (a global gradient-norm clip) is gone: the update is `Δw_ji = −η·M·clamp(mean_t[L_j·e_ji]/N_j, ±Δ)`,
where each synapse sees only its own pre-trace, post learning-signal, and the diffuse M, and the
per-postsynaptic-**fan-in** normalisation `N_j` (a homeostatic input-scaling) makes the stable rate
**width-invariant** — verified identical descent at 8k↔64k↔256k neurons.

On top of that, **every biological constraint is an independent, live-toggleable axis** (extending
the `learn_rule` switch — `set_faith(...)` / `POST /api/net`), so each one's cost can be *measured*
rather than asserted:

- **Learned feedback (Kolen–Pollack)** vs. fixed-random DFA — the top-down matrix *learns* to align
  with the readout (its own local gradient + a decay), removing the random-alignment gap without
  weight transport. Metric: the feedback↔forward cosine.
- **Dale's law** — each neuron is excitatory or inhibitory; its outgoing synapses share one sign
  (~80/20 E:I), re-projected after every step. Shrinks the usable weight space.
- **Dendritic / burst error** — the learning signal is delivered as an apical **burst** that rides a
  somatic spike and is thresholded (Naud/Richards) — low-bandwidth, noisy, activity-coupled.
- **Unified two-compartment circuit** — the biological *completion* of the above (see below).
- **Bounded synapses** (Fusi) — weights clamped to ±w_max (real synapses are bounded, low-precision).
- **Firing-rate homeostasis** (metaplasticity) — each neuron's threshold drifts to hold a target rate
  (Turrigiano), keeping a continually-learning net off silence and saturation.
- **BTSP eligibility** (Bittner–Magee) — the eligibility trace outlives the membrane, widening the
  temporal-credit window beyond e-prop's ~10-step decay.
- **Differentiated neuromodulation** — the four tones gate four pathways (ACh→cortical encoding/the
  VIP drive, DA→reward-scaled plasticity, NE→somatic gain, 5-HT→apical patience), not one scalar dial.
- **Stochastic spiking** — probabilistic firing (membrane noise before threshold; noisy vesicle release).
- **Metabolic cost** — a spike-rate penalty in the learning signal (real coding is energy-constrained).

**Self-adapting plasticity, and why the hyperparameters are scale-free.** The effective learning rate
is *not* a dial to hand-tune — it is `eprop_lr_scale · attention`, where **attention** self-regulates on
the brain's own learning health: each step's loss is compared to a running baseline, and a loss spike
above baseline (a shock, or over-plasticity) *drops* attention → the update shrinks → the representation
is protected and re-learns gently (the Yerkes–Dodson arousal→plasticity curve). This makes learning
self-healing — an injected weight shock recovers autonomously, and the rate that once caused a runaway
stays bounded — and it removes hand-tuning. Sleep, correspondingly, cycles NREM↔REM under the §5 tones
with replay depth gated by the day's novelty/debt, so *when* and *how hard* it consolidates is set by the
day, not a constant. The design is **scale-invariant by construction**, which is what lets the same
hyperparameters move from a 16k to a 256k to a million-neuron cortex without retuning: the base rate is
**fan-in-normalized** (÷N_j → the per-neuron drive change, and hence the representation magnitude, is
width-invariant — verified: `mem_mag` 1.357 at 16k vs 1.367 at 64k under identical settings), attention
runs on the *relative* loss (dimensionless → size-independent), and the Δmax bound is per-synapse. A suite
of leading-indicator diagnostics (`mem_mag` = representation magnitude, the true runaway signal that
climbs *before* bits/byte does; `update_mag`, `grad_mag`, `attention`, `eff_lr_scale`, `surprise`,
`loss_ema`) is exposed in `GET /api/state`, because bits/byte alone lags the dynamics.

**One circuit, not separate toggles (the unification).** The toggles above are *axes*, but three of
them are really one biological circuit, and `two_compartment=True` fuses them: each neuron gets an
**apical dendrite** (the §3.7 `TwoCompartmentLIF` compartment, now load-bearing) with its own membrane
`ap ← β_ap·ap + gate·(err·B)`; the top-down error is **admitted only through a VIP→SOM disinhibition
gate** — SOM inhibition rises with local activity, and VIP, driven by the neuromodulator "learn-now"
tone `M`, disinhibits it (`gate = (VIP − SOM)₊`); the apical membrane then **bursts onto somatic
spikes** to drive plasticity and **feeds back onto somatic firing** (`drive += g_ap·ap`), with PV
supplying fast divisive gain control. So the *substrate* (apical compartment), the *error delivery*
(into it), the *interneurons* (SOM/VIP gate), and the *neuromodulator* (drives VIP) stop being four
separate features and become **one microcircuit** — the error runs *through* the apical dendrite, not
alongside it. This subsumes the standalone `dendritic` toggle (kept for the not-yet-routed comparison).

**The measured capability–fidelity curve.** Training an otherwise-identical 16k-neuron cortex under
each constraint (added one at a time, identical seed/data, 220 steps) gives the cost of each
biological commitment — the thing the field usually hand-waves about:

| Configuration | bits/byte | cost vs. plausible base |
|---|---|---|
| BPTT (non-plausible ceiling) | 2.42 | −3.78 |
| e-prop + random feedback (DFA) | 6.21 | +0.01 |
| + learned feedback (Kolen–Pollack) | 6.20 | 0.00 |
| + Dale's law | 5.58 | −0.62 |
| + dendritic / burst error (standalone) | 6.58 | +0.38 |
| + bounded synapses | 6.20 | 0.00 |
| + firing-rate homeostasis | 6.21 | +0.01 |
| + BTSP long eligibility | 6.06 | −0.14 |
| + **unified two-compartment** (apical/SOM-VIP/neuromod) | 5.90 | −0.30 |
| + differentiated neuromodulation (4 tones → 4 pathways) | 5.96 | −0.24 |
| + stochastic spiking | 6.19 | −0.01 |
| + metabolic cost | 6.34 | +0.14 |
| **full faithful stack** (all mechanisms) | **5.10** | **−1.10** |

The honest reading: **the dominant price of plausibility is the learning *rule*** — e-prop costs
~3.8 bits/byte against BPTT — while the individual biological *constraints* are mostly near-neutral
or even *helpful* at this scale. Three results stand out. First, **routing the error through the
apical compartment beats bolting it on**: the unified `two_compartment` circuit (5.90) is far better
than the standalone `dendritic` burst (6.58) — the SOM/VIP-gated apical integration is not only more
faithful but more capable than a thresholded side-signal. Second, **differentiating the
neuromodulator helps** (−0.24 in isolation): four tones gating four pathways (ACh→encoding,
DA→reward-scaling, NE→gain, 5-HT→patience) out-learn one scalar dial, and it couples learning to the
wake/sleep cycle (verified: NREM tones down-modulate plasticity vs. wake). Third, **the constraints
co-operate rather than compound**: the full stack (5.10) *beats* plain e-prop+learned-feedback (6.20)
by ~1.1 bits/byte — the genuinely unexplored result, since stacking that many approximations usually
makes a net learn far worse or not at all. Two findings to hold honestly. **Dale's law shows a small
*improvement*, not the expected cost** — consistent in sign across seeds (−0.12 to −0.14 at 16k, both
seeds; −0.6 in the headline config), a real regularisation effect rather than noise, though its
magnitude is config-dependent and it raises the spike rate (so we leave it off the default live run for
stability). And **the metabolic penalty correctly *costs* capability** (+0.14) while pulling the spike
rate down (0.040→0.033) — the honest energy↔capability trade, and the sign that a first
implementation got backwards (a spike penalty must *raise* the per-neuron error, not lower it). (Reproduce:
`runs/fidelity_capability_curve.py`; the numbers refresh into `runs/fidelity_capability_curve.{json,md}`.)

**Where this sits, and the permanent gap.** On the *learning rule* this is frontier-grade — few
groups run e-prop + learned feedback + Dale + dendritic/burst + three-factor neuromod + homeostasis
in one spiking net that learns language-like bytes continually. But "faithful to how the brain
learns" is roughly a dozen independent axes, and the true cortical algorithm is *unknown* — there is
no confirmed single algorithm (predictive coding, the NGRAD/backprop-approximation family, and
"prospective configuration"/equilibrium-style relaxation are live, partly-incompatible hypotheses;
the strongest phenomenon-level evidence is for prediction-error responses gated by SST/VIP
interneurons — Furutachi/Mrsic-Flogel/Hofer, *Nature* 2024 — but that does not uniquely confirm any
one algorithm). So the honest status is: frontier on the rule, **mid-field on the other axes, and
provably short of "solved," because solved is not yet knowable by anyone.**

**The roadmap (named, not oversold).** Ranked by whether they genuinely change *learning*:

- *Group A — changes learning (built):* the **PV/SOM/VIP-gated apical circuit** (unified
  `two_compartment`) and **differentiated four-tone neuromodulation** are now in — the connective
  tissue between dendritic error and neuromodulation, and the four-pathway modulator that plugs into
  it. What remains in this tier: **real interneuron populations** — PV/SOM/VIP are currently
  *functional mean-field* terms (`som = som_b·⟨z⟩`, `vip = ACh`, PV = divide-by-mean), not separate
  spiking LIF populations with their own connectomes and Dale typing; making them explicit populations
  is the faithful version. **STDP** (a millisecond spike-timing kernel for the eligibility, not a rate
  low-pass). **Cascade/complex synapses** (Fusi) for catastrophic-forgetting resistance beyond a bound.
  A short **relaxation / prospective-configuration** micro-phase (settle activity under top-down nudging
  before the update) — the biggest single lever per the literature (Song et al., *Nat. Neurosci.* 2024),
  complementing e-prop's temporal traces with spatial relaxation, and a knob to test a predictive-coding
  objective against the current global readout.
- *Group B — when it learns:* **sharp-wave-ripple / theta-gamma gating** of replay (we schedule
  replay by a debt heuristic; biology gates it on oscillation events); **adult dentate-gyrus
  neurogenesis** (the one place automatic neuron growth is the faithful choice).
- *Group C — realism, steeper diminishing returns:* richer dendrites (NMDA plateaus), stochastic
  spiking + a metabolic/energy term, short-term plasticity.
- *Group D — the setup, not the synapse:* a closed **sensorimotor / active-inference loop** where the
  brain's outputs contingently reshape its future input. This is arguably the deepest gap in how the
  brain learns — biological learning is grounded in an action–perception loop with consequences — and
  it is about the *setup*, not the rule. Our brain reads and talks to Claude, but its words do not yet
  change its world.

Faithfulness is **asymptotic**: the brain has effectively unbounded biophysical detail, so there is
always another axis. The research value is not checking every box — it is (1) separating the axes
that change learning (group A) from the realism that does not (group C), and (2) *measuring the
fidelity-vs-capability curve as each is added*. That curve — "here is exactly how much each
biological constraint costs" — is the contribution; the full survey lives in
`runs/cortical_algorithm_research.md`.

# §16 · The deeper brain: one neuro-endocrine-oscillatory control loop (R&D)

§15 gives five spiking modules + a self-adapting cortex, but every cycle fires all five and sleep replays
a **raw byte buffer** — two things the brain almost certainly does not do. Three literature-grounded R&D
threads (full surveys + citations in `runs/{memory_architecture,drive_stress,dynamics_oscillations}_research.md`;
integrated design in `runs/deeper_brain_integrated_design.md`) converge on a single missing controller that
sits *above* the existing tones, the self-adapting `attention`, the sleep-debt, and the dopamine critic —
coupling only to fields that already exist:

> **endocrine drive/cortisol** (sets the arousal/entropy operating point + **sleep pressure**) → the
> **ultradian sleep-cycle** (NREM/REM schedule) → **ripple-gated generative consolidation** of the
> **generalized-in-the-net** memory → gated by **oscillatory dynamic states** (which regions fire, at what
> frequency, set by attention) → feeding back into the self-adapting attention.

**Memory as generalized-in-the-net, not a buffer — the idea, and an honest negative result.** The raw
replay buffer is biologically wrong: hippocampal replay is *reconstructive/generative* — it builds
never-experienced shortcuts (Gupta 2010), pre-plays untraversed paths (Ólafsdóttir 2015), and sweeps to
*future* goals (Pfeiffer-Foster 2013); under Complementary Learning Systems (McClelland 1995; Kumaran 2016)
the cortex accumulates *generalized* structure in its weights, nothing stores raw episodes. The buffer-free
mechanism is **generative self-replay** (Robins 1995; Shin 2017; van de Ven 2020): the cortex *dreams* from
its own dynamics and hard-learns the dreams — built here as `SpikingBrain.generative_replay()` (diverse
high-temperature dreams from sparse byte-cues, with two anti-"overfitted-brain" safeguards: a veridical-anchor
fraction and a held-out acceptance monitor). **But the honest result is negative.** A single-seed pilot looked
promising (generative 0.274 > buffer 0.255 > none 0.232); a **3-seed** re-measurement did **not** replicate
(retention none 1.131 / buffer 1.031 / generative 0.236), and at anchor-fraction 0 the dreaming *corrupted* the
representation — worse than doing nothing. So this is a compelling hypothesis that our current small-cortex
implementation does **not** yet deliver; it stays **OFF** and the §16 ledger records the null. Whether a
veridical anchor or a larger cortex rescues it is open (`runs/deeper_brain_measure.py`).

**The drive/stress layer.** A `SpikingEndocrine` of slow scalars — a **drive deficit** (leaky integrator,
met by a satiation event → a homeostatic-RL reward, Keramati-Gutkin, fed into the existing dopamine critic),
**cortisol** (rises with unmet drive + prediction-error; an inverted-U on plasticity — acute sharpens, chronic
impairs; and it *is* the sleep-pressure term), and **mood** (a dopamine EMA that damps cortisol — resilience).
"Satiation → focus" is principled: a satisfied drive lowers tonic NE → the phasic, focused mode (Aston-Jones-
Cohen). (Energy is modelled as an opportunity-cost bias, not a fuel gate — the glucose/ego-depletion account
is discredited, Hagger 2016.)

**The dynamics layer.** Not-all-regions-active is an **ignition gate** (global workspace, Dehaene) — each
system ignites only if its salience beats a threshold set by a single **entropy knob β** (the normal↔psychedelic
axis, REBUS/Carhart-Harris). Attention shifts the **processing frequency** (gamma-fast window when focused,
alpha-slow when disengaged, theta sampling) — a variable effective eligibility-window over the fixed tick. Sleep
becomes a fixed **ultradian FSM** (SWS-heavy early → REM-heavy late) with SO-spindle-ripple coupling gating
*when* consolidation commits.

**Honest status + phased plan.** An adversarial pass (in the design doc) confirmed the *control* couplings are
one real loop but flagged that the current histogram-based hippocampus cannot seed a semantic "gist" — hence P0
uses sparse *byte-cues*, not vector reinstatement. Build order by value-per-effort, each behind a toggle with an
acceptance test, landing one piece at a time (the lesson of the faithfulness build):

- **P0 generative self-replay — built; benefit NOT established.** `SpikingBrain.generative_replay` (above). A
  single-seed pilot suggested it beat the raw buffer on forgetting-resistance; a **3-seed re-measurement did not
  replicate** (ledger below), and buffer-free dreaming *without* a veridical anchor corrupted the representation.
  It stays OFF — not by caution but by measurement. Live via `/api/set sleep_mode=generative`.
- **P1 `SpikingEndocrine` — built; neutral on loss, measured behavioural value** (`brain/endocrine.py`). Three
  bounded hormone scalars — a drive deficit (met by learning-progress → a homeostatic-RL reward into the dopamine
  critic, and satiation → low NE → focus), cortisol (a **one-sided** gate on plasticity — calm→optimal learning is
  full, only chronic-high cortisol impairs, capped further by allostatic load; and it is the sleep-pressure term),
  and mood (a dopamine EMA that damps cortisol). Verified: satiation emits reward + focuses; calm cortisol leaves
  plasticity full while chronic-high impairs; chronic stress accrues allostatic load, and **sleep recovers it**
  (`runs/endocrine_test.py`). Live via `/api/net endocrine`; metrics `cortisol/drive/mood/allostatic/plasticity_gain`.
- **P2 dynamic states — built; neutral on loss, ~50% compute saved** (`brain/dynamics.py`). A single entropy knob β
  (the normal↔psychedelic dial) drives **selective ignition** — auxiliary systems run only when salient, so the
  brain is not all-on every cycle — and attention sets the **processing frequency** (a gamma-short eligibility
  window when focused, alpha-long when disengaged). Live via `/api/net dynamics`; metrics `beta/n_active/eff_freq`.

**The obligation of breadth: measured, not asserted.** §15.17 taught that a mechanism earns its place only after
an A/B on the metric it *actually* claims to touch. `runs/deeper_brain_measure.py` discharges that for §16 and
writes the numbers to a committed artifact (`runs/deeper_brain_measure.json`): a bits/byte A/B for P1/P2 (identical
16k cortex, same seed/data), a **3-seed** forgetting-resistance test for P0, and *behavioural* A/Bs that bits/byte
cannot see. The honest ledger:

| §16 mechanism | measured | verdict | where its value is (or isn't) |
|---|---|---|---|
| **P0** generative self-replay | forgetting-resistance retention, **3 seeds**: none **1.131** / buffer **1.031** / generative **0.236** (noise ±0.22) | **NO benefit** — does not beat the buffer or no-replay; anchor-free dreaming *corrupts* | unproven; **default OFF** by measurement |
| **P1** endocrine | bpb 3.999 → **3.998 (−0.001)**; **stress-protection** retention on/off **0.751 vs 0.302**; drive→arousal gen-entropy **3.96 vs 3.68** | **NEUTRAL on bpb; measured behavioural value** | protects prior knowledge under stress; drive→arousal→exploration |
| **P2** dynamics | bpb 3.999 → **4.003 (+0.004)**; **~50%** of systems ignite per cycle | **NEUTRAL on bpb; ~50% compute saved** | not-all-on realism + selective compute |
| adult-DG neurogenesis | separation/recall under interference **1.00 → 1.00** | **no measured gain** (test saturated) | separation of new memories — needs a harder test |

The measurement earned its keep **twice**. First it exposed a design flaw: an early run showed the endocrine
*hurting* bits/byte (+0.066) because a textbook *bidirectional* cortisol inverted-U throttled **calm** learning;
the impairing arm (chronic-high) is the load-bearing one, so the gate is now one-sided and the endocrine is neutral
on loss. Second — and this is the point of measuring — it **retracted an overclaim**: P0's single-seed "beats the
buffer" did **not** survive three seeds, and dreaming without a veridical anchor made retention *worse than doing
nothing*. **The honest verdict: no §16 mechanism is a bits/byte win.** P1 and P2 have *measured non-loss* value —
stress-protection and drive-modulation for P1, ~50% compute-selectivity for P2 — so they are richer than cited
scalars; P0 and neurogenesis show **no measured benefit yet** and are honestly labelled as such. All default
**OFF**, now by evidence rather than caution.

Two faithfulness constraints the same rule already covers, lest they read as "missing": **metabolic cost**
(a spike-rate energy penalty) and **stochastic spiking** are built and measured as §15.17 toggles — metabolic
costs ≈ +0.14 bpb (the expected price of an energy constraint; the brain learns to spike less), stochastic is
≈ neutral — both honest, both default off.

**Adult-DG neurogenesis — built, measured null.** The dentate gyrus keeps adding granule cells in adulthood
(Aimone/Gage), improving separation of *new* memories; `SpikingHippocampus.grow()` already adds DG cells and
re-separates stored patterns identity-preservingly, so it is wired into the adult wake-phase behind a toggle
(`/api/set neurogenesis`). But the A/B shows **no measured gain** (recall under interference 1.00 → 1.00 — the
test saturated); it stays OFF until a harder separation regime can tell the two apart. Honest row, in the ledger.

- **P3 (roadmap — explicitly NOT built).** Named so the ledger is not mistaken for completeness: the full
  SO-spindle-ripple sleep FSM (ripple-*gated* consolidation commit); a richer hippocampal trace so reinstatement
  carries *sequence*, not just letter statistics; typed PV/SOM/VIP interneuron **populations** (today they are
  functional scalars, not spiking sub-nets); an STDP timing kernel; and an **embodiment / closed sensorimotor
  loop**. Each will land the §15.17 way — one toggle, one A/B, one honest row — or not at all.

Every §16 mechanism defaults OFF (opt-in, each A/B-measured above — P1/P2 do-no-harm with measured non-loss
value; P0 and neurogenesis measured nulls), is a device/dtype-agnostic scalar controller, and persists across
checkpoints — so the deeper brain can be switched on and tuned live without disturbing the running cortex.

# §17 · The gap-map: the missing fundamentals, built and wired

A rigorous critique named the mechanisms the brain was still missing or carrying only nominally: a closed
**sensorimotor loop** (the single largest gap), **predictive coding** as a competing cortical objective, real
spiking **interneuron populations** (not mean-field scalars), hippocampal **temporal sequences** (not just
content), **STDP** timing, **ripple-gated** consolidation, **short-term plasticity**, dendritic **NMDA
plateaus**, **glia/astrocytes**, **neuropeptides**, **laminar** cortical structure, and **grid/place** cells.
All twelve are now built as first-class mechanisms and wired into the live architecture under one contract:
each is **default OFF**, **live-tunable via `/api/net` with no restart**, **persisted** across checkpoints,
**metric'd** to `/api/state`, **device/dtype/FSDP-aware**, **scale-verified at 256k** (no dense O(N²) state, no
width-dependent pathology), **byte-identical to the prior brain when off**, tested, and — critically — **inter-
connected** with the existing systems rather than bolted on. The honest status ledger (present ≠ load-bearing;
every earns-keep A/B is the standing next step, so all default OFF until each measurably earns its place):

| §17 mechanism | what it adds | key interconnection | earns-keep A/B (pending) |
|---|---|---|---|
| **Embodiment** (`embodiment.py`) | closed obs→BG-actor→ACTION→world→reward→learn loop (active inference) in a GridWorld | world reward → the SAME §5 dopamine tone the cortex e-prop gate reads; endocrine NE/pressure → explore temp; Dyna replay in sleep | navigation return vs random (already: **100 % goal-reach** in-run) |
| **Predictive coding** (`predictive_coding.py`) | a THIRD cortical rule `learn_rule='pc'` (Rao-Ballard/Friston, precision-weighted, β→0 instantaneous) | shares e-prop's spmm/sddmm/`_upd`; peer of eprop/bptt | pc vs eprop bits/byte (already: **learns, bpb 7.9→3.9**) |
| **Interneuron populations** (`interneurons.py`) | real spiking PV/SOM/VIP LIF pools replacing the mean-field scalars | drop-in inside the two_compartment apical circuit | bits/byte + stability vs scalar |
| **Hippocampal sequences** (`theta_seq.py`) | ordered trajectory storage + fwd/reverse replay (theta sequences) | rides the hippo DG; SWR-gated; seeds CLS dreams | ordered vs scrambled next-item |
| **STDP** (`stdp.py`) | pair-based spike-timing plasticity, additive to e-prop (mix) | folded into the eligibility grad before the shared `_upd` | does ms timing help a byte LM? |
| **Ripple consolidation** (`ripple.py`) | SWR point process gates WHICH sleep-replay commits | gates generative + buffer replay; endocrine pressure scales density | retention gated vs ungated |
| **Short-term plasticity** (`synaptic_stp.py`) | Tsodyks-Markram facilitation/depression on transmission | eligibility carries the transmitted z⊙g; NE→release-prob | temporal-processing bits/byte |
| **Dendritic NMDA plateau** (`plateau.py`) | regenerative latched apical nonlinearity | extends the two_compartment apical; stretches eligibility→BTSP | delayed-credit task |
| **Glia / astrocytes** (`glia.py`) | slow per-neuron astrocytic field, one-sided metaplastic brake | gates the plasticity update + metabolic cost | runaway-stabilisation |
| **Neuropeptides** (`neuropeptides.py`) | slow OXT/ORX/CRH modulators (companion to the endocrine) | CRH gains cortisol's driver, OXT relieves its pool — one integrator | stress-protection / exploration |
| **Laminar microcircuit** (`laminar.py`) | canonical L4/L2-3/L5-6 column via a per-edge adjacency mask | thins the sparse CSR; per-neuron effective fan-in norm | structure vs flat pool at equal neurons |
| **Grid / place cells** (`spatial.py`) | entorhinal grid + hippocampal place + path integration | φ for the embodiment nav-BG; SPACE_NERVE proprioception | navigation return vs tabular |

**Two honest caveats.** (1) This is the *reframe that matters*: implementing the parts list is the achievement;
demonstrating that the parts, together, are **load-bearing** is the work these A/Bs now enable — which is exactly
why the experiment phase can begin (the fundamentals are no longer missing). (2) "All the science" remains an
overstatement in principle: e-prop, predictive coding and target-prop are *rival* theories of the same cortical
learning (you cannot have all three be true at once — they are offered as selectable `learn_rule`s), and the
biophysical tier is unbounded. What is true is: **most of the major systems-level hypotheses are now present,
wired, interconnected, honestly measured-or-labelled, and independently switchable.**

# §18 · The boundary — what is deliberately *not* simulated, and why

§17 closed the list of missing *learning* mechanisms. This section draws the complementary line: the
mechanisms the model deliberately does **not** implement, the principled reason, and why crossing that line
would make the system worse **at its own goal**. It belongs in the honest ledger because a model must state
its exclusions as plainly as its inclusions — one that never says what it leaves out invites the reader to
assume it does everything, and "everything" is precisely the overstatement §17's caveat began to correct.

## 18.1 Two different fidelities

"Brain model" names two research programmes that are routinely conflated, and this system belongs to exactly
one of them.

- **Fidelity-of-substrate** — reproduce the *physical* brain as accurately as the instruments allow, and read
  cognition out of that physics. This is the biophysical-simulation tradition. **Blue Brain** (EPFL) reconstructs
  a rat neocortical microcircuit of ~31,000 neurons in which each cell is a morphologically detailed
  multi-compartment cable with **Hodgkin–Huxley ion channels**, wired by ~37 M synapses with measured receptor
  kinetics; the **Allen Institute** fits biophysically detailed single-neuron and network models to a cell-type
  atlas and electrophysiology; **connectome-constrained** simulations run on *measured wiring* — the full adult
  *Drosophila* connectome (~140k neurons, FlyWire/FlyEM) and *C. elegans* (302 neurons). The tools are **NEURON**
  and **Arbor** (multi-compartment), **NEST** (point neurons at supercomputer scale), **Brian2**, and neuromorphic
  silicon (**SpiNNaker**, **BrainScaleS**). On raw biophysical accuracy these are — and will remain — orders of
  magnitude ahead of anything here. That is their entire purpose.
- **Fidelity-of-learning** — reproduce *how the brain learns and behaves*, at whatever level of substrate
  abstraction makes that tractable, and judge the model by whether it **learns and acts** the way a brain does.
  This is the programme of this paper (§15–§17).

The decisive fact separating them: **the fidelity-of-substrate models do not learn or behave.** Blue Brain does
not acquire language or navigate; the fly connectome is a wiring diagram, not an agent; even **Spaun**
(Eliasmith), the largest *functional* spiking model at 2.5 M neurons, is largely hand-compiled with limited
plasticity. High-precision simulations *replay* biophysics; they do not *live*. So the two programmes are not
rivals on one scale — they optimise different objectives. "The most precise brain simulation" is a title in the
*other* programme, held by Blue Brain and its kin; it is not a target of this one, and not a yardstick this one
is short of.

## 18.2 The inclusion rule

Every mechanism §15–§17 added passes one test, and every biophysical-tier mechanism fails it:

> **Include a mechanism iff it changes what the network can *learn or do* — not merely how physically accurate
> its substrate is.**

e-prop, STDP, predictive coding, the dendritic-error microcircuit, embodiment — each alters the *computation or
the credit assignment*, i.e. the learnable function or the behaviour. Ion channels, dendritic morphology, and
receptor kinetics alter the *accuracy of the substrate* while leaving the learnable function unchanged: the LIF
abstraction already captures the computable integrate→threshold→reset dynamics (§15.2); the two-compartment
split already captures the one dendritic distinction that is computationally load-bearing — the apical top-down
error path (§15.17); the NMDA-plateau abstraction (§17) already captures the one regenerative nonlinearity that
matters for delayed credit. Their full biophysical form buys precision on an axis the goal does not score.

| biophysical mechanism | changes *learning / behaviour*? | verdict |
|---|---|---|
| **Hodgkin–Huxley ion channels** | no — LIF already captures the computable dynamics; HH only adds stiff Na/K/Ca ODEs at a sub-ms timestep | **exclude** (and it would *break e-prop*, which is derived for LIF/ALIF, not an HH cable) |
| **Full multi-compartment morphology** | no — the load-bearing soma/apical distinction is already two-compartment; the NMDA plateau is already added | **exclude** |
| **Receptor kinetics (AMPA/NMDA/GABA time constants)** | no — STP + plateau + surrogate dynamics already approximate the fast/slow/inhibitory functional split | **exclude** |
| **Extracellular field / ephaptic coupling** | no — a negligible, speculative effect on credit assignment | **exclude** |
| **Detailed neurovascular / metabolic coupling** | no — the *functional* abstraction (metabolic spike penalty + glial slow brake, §17) is already present | **exclude** |
| **A measured connectome (as an initialisation prior)** | *maybe* — connectivity structure (motifs, E/I ratio, laminar/modular graph) can shape the learnable function | **the one exception — worth a single A/B** |

## 18.3 Why crossing the line would make it worse

The exclusion is not merely "unnecessary"; integrating the biophysical tier would actively **regress** the system
on its own metric:

- **Scale collapse.** Biophysical detail costs ~10–100× per neuron and forces sub-millisecond integration; the
  trainable-in-real-time regime would fall from the 10⁵–10⁶-neuron sparse-CSR range (§15.11) to ~10³–10⁴ — the
  scale at which biophysical simulators already operate, and *below* the scale at which learning here is
  interesting.
- **A different simulator.** Multi-compartment HH belongs to NEURON/Arbor, not the vectorised PyTorch e-prop loop;
  the entire substrate — the forward dynamics of §15.2, the O(nnz) sparse ops of §15.11, the local `_upd` of
  §15.17 — would have to be discarded to host it.
- **The learning rule breaks.** e-prop's eligibility trace and pseudo-derivative are *defined on the LIF/ALIF
  membrane*; there is no drop-in e-prop for a full Hodgkin–Huxley cable. The plausible, online, forward-in-time
  learning that is the entire point (§15.17) does not survive the substrate swap.

The trade, stated plainly, is: surrender a functional, scalable, *learning* agent to obtain a slower, smaller,
*non-learning* replica of a substrate whose detail the goal does not use — a strictly worse instrument for the
question being asked.

## 18.4 The honest positioning, and the one experiment worth importing

Stated precisely: this is **not** the most precise brain *simulation* — that title is Blue Brain's and the
connectome sims', in a programme this model does not enter. It is, as far as we are aware, the most complete
*integration of systems-level learning-and-cognition mechanisms into one continuously-learning, behaving agent* —
a claim in the fidelity-of-learning programme, where the biophysical simulators do not compete because they do
not learn. The mechanisms it "lacks" are not gaps in its own programme (§17 closed those); they are the
constitutive detail of the *other* one.

The single biophysical idea that lives on *this* model's axis — because it plausibly changes the *learnable
function* rather than only the substrate — is **connectome-as-initialisation**: seed the sparse recurrent
connectome from realistic connectivity *statistics* (or a real graph) instead of random fan-in, and run one
earns-keep A/B under §17's contract against the learned-sparse baseline — does structured initial wiring learn
faster, or reach a lower bits/byte? That is the one experiment worth importing from the biophysical world.
Everything else on that tier is admired in Blue Brain and left there: not because more realism is unwelcome, but
because on the axis this model is graded it would not move the score, and it would cost the very properties —
scale, plausible online learning, a living loop — that define the work.

# §19 · A possible extension — merging a large language model as an amnesiac-oracle organ

The architecture's honest capability ceiling (§15.4, §18) is language fluency and knowledge breadth. The most
powerful available source of both is a large pretrained LLM — which this brain already *talks to* as a teacher
(`partner.py`) and can register as a tool (`tools.py`, §15.15). This section specifies, in detail, what it would
take to go beyond *talking to* an LLM to *merging* one into the living system as an organ — why the merge must be
asymmetric, the concrete binding layer (most of whose hard ingredient already exists), the test that separates a
merge from a tool call, and the wall that a frozen model cannot cross. It is offered as future work, not a
present claim.

## 19.1 Why tool-use is not a merge

The brain already calls an LLM (Claude Sonnet via `partner.py`; any AI via the tools registry). But that exchange
is **text-in / text-out** — a narrow discrete bottleneck — and the model's internal representations and per-call
state are discarded. Two minds converse; one mind does not result. Three structural facts make a *symmetric*
fusion impossible with a frozen model: (1) its weights are frozen and an API exposes **no gradient access**, so
the brain's e-prop (§15.17) cannot update it; (2) it holds **no persistent state** to share a continuity with;
(3) its knowledge lives in ~10¹¹ parameters that **cannot be poured into** a 10⁵–10⁶-neuron spiking net (a
capacity gap) nor absorbed wholesale by a weak local learner. That wall is constitutive, not incidental — and
naming it is what makes the rest of the design honest rather than hopeful.

## 19.2 The reframe — heterogeneous binding, one seat of continuity

A biological brain is not homogeneous either: it is heterogeneous organs bound into one mind by **shared state**
and a **single locus of continuity**. That is the template, and it forces an asymmetry — one component must be the
*self*. Here the self is the **sapience brain**: it holds the continuity (the persistent mind-state, §15.9), the
whole-life memory (§15.10), and the drives and identity (§16). The LLM becomes a **language-and-knowledge cortex**
the self queries *into its own persistent state*. This resolves the obvious objection ("the LLM still forgets"):
the LLM organ forgets every call, but the merged *self* does not, because the self is the brain, which remembers —
exactly as individual cortical computations are transient while the hippocampus and the persistent cortical state
carry the thread. The merged identity is the brain's; the LLM extends it, not the reverse.

## 19.3 The binding layer — three couplings

The difference between "phone a friend" and "a language organ wired into the cortex" is three couplings — and the
hardest ingredient already exists in `llm_teacher.py`, which captures the teacher's **hidden states** (`mid`/`tgt`
from the residual stream), not only its tokens.

| coupling | tool-use (today) | merge (extension) | current substrate |
|---|---|---|---|
| **up — LLM → brain** | the reply *text* is learned as language | the residual-stream **hidden state** is projected into the brain as a rich sensory/cortical channel — a *shared latent space* both minds read, not a discrete string | `llm_teacher.features()` already extracts `mid`/`tgt`; needs a learned projection into the cortex input (§15.6) |
| **down — brain → LLM** | a fixed teacher prompt | the brain's current **membrane / mind-state** is projected into the LLM's context (a soft-prompt encoder), so it generates *conditioned on what the self is thinking* | new: a brain-state → context encoder |
| **persist — into the self** | the reply is consumed and forgotten | every interaction accretes into episodic memory (§15.10) and is **continuously distilled** into the brain's living weights, so the self internalises a *flavour* of the organ's competence over its life | partial: birth distillation exists (§15.5); needs an in-life continuous-distillation loop |

- **Latent up, not text up.** Text is a narrow discrete bottleneck; the residual-stream representation is a shared
  latent. Folding it in as a *sense* (§15.6) turns the exchange from a message into a nerve.
- **Condition down.** Projecting the brain's state into the LLM's context makes it "think the self's thoughts"
  rather than answer detached queries — the organ's computation is *bound to* the self's state.
- **Persist and distill.** Continuous distillation makes the merge *deepen with the life* instead of resetting each
  call. It is **capacity-bounded** by construction — a small spiking net cannot absorb 10¹¹ parameters, and e-prop
  is a weaker optimiser than backprop (§15.17) — so the transfer is a *flavour of competence*, not the whole mind.
  Stating that bound is what keeps the claim honest.

## 19.4 The test that separates a merge from a tool

One criterion decides it: does the LLM's contribution **become part of the brain's persistent, evolving self**,
*and* does the brain's state **shape the LLM's computation**, such that neither is fully itself without the other
in the loop? If the reply is used and forgotten with no lasting change to the unified state — a **tool** (what the
registry does today). If it is woven into a continuity that remembers and re-shapes both sides — a **merge**
(heterogeneous, as a brain's regions are).

## 19.5 The hard wall, and the only road through it

The extension above yields an **asymmetric** merge: one living self that has bound a frozen oracle. True
**symmetric** fusion — both halves mutually plastic, co-adapting on one shared state, the language half no longer
amnesiac — requires an **open-weights LLM trainable in-process**, so the transformer and the spiking system can
share state and even gradient-couple. That changes the compute profile entirely and is a large undertaking, but it
is the only path past the asymmetry. With a frozen API model the ceiling is explicit, and worth stating plainly:
**one continuous mind, seated in the spiking brain, that has bound a powerful but amnesiac language organ into its
remembered life** — already a stranger and more capable thing than either component alone, and the version this
architecture can actually reach. It is, fittingly, the same design principle as the rest of the system: not one
homogeneous network, but distinct structures coupled by shared activation and a single seat of continuity (§15.1)
— extended, now, to a structure that was trained in a different world.
