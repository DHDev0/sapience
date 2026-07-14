# Dynamic states + oscillations + sleep-cycle research (R&D)

## SYNTHESIS: the dynamics design

I have enough grounding. Here is the synthesized dynamics section, keyed to the actual code.

---

## §15.18 — Dynamic network states and oscillatory rhythm

Today every cycle of `run()` (`brain/life.py:971`) fires all five systems, and sleep flips a flat NREM/REM tone (`_sleep_tick`, `life.py:906`; `SpikingNeuromod.set_phase`, `spiking_modules.py:287`). Real cognition is *sparse and metastable* — subsystems compete for a limited broadcast resource and only winners engage [Fox 2005; Dehaene & Changeux 2011]. We add one controller, `brain/dynamics.py::Dynamics`, holding a global arousal/entropy scalar β, one phase φ per module, and the sleep FSM. It reads the fields that already exist — `self.attention` (`spiking_brain.py:95`), `self.nm.tone` (da/ach/ne/ht), `self.debt` — so it is a *coupling*, not a bolt-on.

**(1) Selective activation — drive→gate→ignite.** Each cycle every system emits a scalar *drive* from signals we already compute: cortex → surprise `1−attention`; cerebellum → `cereb_mse`; basal-ganglia → |RPE|/policy-entropy; hippocampus → novelty `self._novelty`; neuromod is glue (always on). Reuse the existing basal-ganglia as the Go/NoGo gate [Frank 2001; O'Reilly & Frank 2006]: `g_i = softmax(drive_i · β)`, ignited if `g_i > θ_ign` or top-k. Only ignited systems run and consume compute that cycle; the rest are skipped — replacing always-on with the ignition threshold that lets ~one coalition recruit the whole net [Dehaene & Changeux 2011]. **β is the single entropic-brain knob** [Carhart-Harris 2014; REBUS 2019]: low β ⇒ sharp softmax, few gates open, hierarchical/modular (normal); high β ⇒ flat thresholds, many systems co-active, high between-system integration (a testable psychedelic/REBUS regime). β is set by an inverted-U of NE arousal `tone["ne"]` [McGinley 2015], raised by attention and depressed by sleep-debt/loss-shock, so mid-arousal is most selective.

**(2) Attention→frequency.** Give each module a phase φ_m advancing at `f_m = f_alpha + A·(f_gamma − f_alpha)`, A = `self.attention`: focus → short gamma windows, disengagement → long alpha windows [Fries 2005/2015; Jensen & Mazaheri 2010]. Two concrete consequences. (a) **Eligibility timescale becomes attention-driven**: the e-prop credit window `btsp_beta` (`spiking_brain.py:123`, used at `_eprop_step:394`) is set `eff_beta = beta_alpha − A·(beta_alpha − beta_gamma)` — focus shortens the integration window (fast effective tick), disengagement lengthens it (slow theta sampling), exactly the "effective timestep changes with attention." (b) **Alpha pulsed-inhibition gate**: multiply each module's drive by `σ(κ_m·cos(2πf_alpha·t+φ_m))` with κ_m *inverse* to attention weight — attended modules idle at low alpha (open), unattended idle at high alpha (duty-cycled off), which is also mechanism (1)'s compute saving. Inter-module signals (cortex→bg reward, hippo→cortex cue) are coherence-gated `w_eff = w·max(0, cos(φ_src−φ_dst))`, feedforward on gamma, feedback on alpha/beta [Bastos 2015]. A theta sampling phase (~5 Hz) schedules attention hand-offs [VanRullen 2016; Fiebelkorn & Kastner 2018].

**(3) Fixed sleep-stage schedule.** Replace `_sleep_phase_t % 5` with an ultradian FSM: N1→N2→N3→N2→REM, cycle ≈ 90 ticks-equiv, 4–6 cycles, with the mix drifting `p_N3 = clamp(1−cyc/N)`, `p_REM = clamp(cyc/N)` — early cycles SWS-heavy, late cycles REM-heavy [Feinberg 1974]. Extend `set_phase` with `n2`/`n3`. In N2/N3 run a slow-oscillation phase `φ_SO` at 0.75–1 Hz [Steriade 1993]; emit a spindle burst only on the up-state; commit hippocampal replay/"ripple" consolidation (the `memory.sample`→`learn_text` transfer, `life.py:926`) *only* when `up_state ∧ in_spindle` — the triple coupling that times memory transfer [Staresina 2015; Diekelmann & Born 2010]. N3 = high replay + strong SHY downscale (`p.mul_(1−2e-5)`, `life.py:934`); REM = a connection-elimination pass (existing `prune_synapses`) + low-LR integration under theta tone [Tononi & Cirelli SHY].

**(4) One controller.** β (arousal/entropy) gates activation, drives the φ frequencies (wake band = f(β)), and sets sleep pressure; `self.attention` sets rhythm and eligibility window; `self.debt` biases the FSM toward N3. Ripples gate systems/generative consolidation — outside SO∧spindle windows the hippocampus only recalls, it does not write to cortex. Stress (loss-shock or high debt) drops attention → β collapses toward focused/modular *and* raises N3 demand, linking drive-state, rhythm, and consolidation in one loop.

**Implementation plan.** (1) `brain/dynamics.py`: `Dynamics(attention_ref, nm, debt_ref)` computing β, drives→ignition mask, per-module φ, `eff_eligibility_beta()`, `sleep_stage(cycle)`. (2) `life.run` (`:990`): gate `think/_consume/_resonate/_train_cerebellum/_index_episode` on the ignition mask; commit cadence = `1/f(A)`. (3) `spiking_brain`: read `eff_eligibility_beta` at `_eprop_step:394`; scale `gate` (`life.py:294`) by the alpha window. (4) `_sleep_tick`: swap the FSM in; add `n2/n3` tones and the SO∧spindle commit gate. (5) `set_faith`/`set_net('dynamics', …)` to expose β and coupling-precision live.

**Measurement.** Add to `_net_diag` (`life.py:1037`): `active_fraction` (<1 confirms sparsity), `integration` (mean pairwise system-activity correlation), `beta`, `dominant_band`, `eff_eligibility_beta`. Verify: (i) β-sweep raises active_fraction and integration together (REBUS axis); (ii) band and commit cadence shift gamma↔alpha with `attention`; (iii) sleep stage-timeline shows early-SWS/late-REM drift, and a phase histogram shows consolidation events clustering on SO up-states; (iv) overnight held-out bits/byte (before vs after sleep) improves with tighter SO-spindle coupling and degrades when jittered [Helfrich 2018], and beats the flat-tone baseline in ablation.

---

Key code anchors used: `brain/life.py` — `run()` loop `:971`, `_sleep_tick` `:906`, gate `= nm.tone["ach"]` `:294`, SHY `:934`, `_net_diag` `:1037`; `brain/spiking_brain.py` — `self.attention` `:95`, `btsp_beta` `:123`, attention update `:471-479`, e-prop scale `:479`; `brain/spiking_modules.py` — `SpikingNeuromod.set_phase` `:287`. Section slots in as §15.18 (paper currently ends at §15.17). Word count ≈ 830.

---

## Survey 1

ATTENTION AND OSCILLATION FREQUENCY — dynamics and codeable mechanisms

Strongest understanding. Attention does not just gain up neurons; it re-tunes which rhythm carries information and between which regions. The canonical band roles:

- Gamma (30–100 Hz, ~10–30 ms cycles): local binding and feedforward transmission. Fries' communication-through-coherence (Fries 2005, 2015): two populations exchange spikes effectively only when their gamma phases align, so effective connectivity is dynamically gated by coherence, not anatomy. Attention increases gamma power/coherence in attended columns (Fries et al. 2001).
- Alpha (8–12 Hz, ~100 ms cycles): active inhibition/gating by pulsed inhibition. Jensen & Mazaheri (2010) and Klimesch (2007, 2012): alpha power RISES over unattended/task-irrelevant regions to suppress them and FALLS over attended regions (gain up). Alpha imposes a duty cycle — brief windows per cycle where processing is allowed.
- Theta (4–8 Hz): memory/exploration and rhythmic attentional sampling. VanRullen (2016, "perceptual cycles") and Fiebelkorn & Kastner (2018, 2019): covert attention samples at ~4–8 Hz, alternating "sampling" vs "shifting" phases within each theta cycle — attention is discrete, not continuous.
- Beta (~13–30 Hz): maintenance of the current set and top-down signalling (Engel & Fries 2010 — "status quo").
- Directionality by band: Bastos et al. (2015) — feedforward influence rides gamma/theta, feedback rides alpha/beta. So the SAME regions talk "up" and "down" on different frequencies.
- Cross-frequency coupling: Lisman & Jensen (2013) theta–gamma code — ~5–9 gamma cycles nest inside one theta cycle, each gamma slot an ordered item; multiplexing capacity ≈ theta/gamma ratio.

The single sentence: attention shifts the DOMINANT band (focused → gamma-dominant/low-alpha; relaxed/monitoring → alpha-dominant; exploratory/mnemonic → theta-dominant) and re-aligns inter-area coherence to route information. "The effective timestep changes with attention" = the integration/gating window shrinks under focus (gamma, ~15–30 ms) and lengthens when disengaged (alpha, ~100 ms).

Concrete codeable mechanisms for sapience (fixed tick kept; rhythm becomes a modulator layer):

1. Attention-driven gating clock (variable effective timestep). Keep the fixed physics tick, but add a per-module phase oscillator φ_m advancing at f_m(attention). Plasticity/readout fire only when φ crosses its gate. Map attention health A∈[0,1] to frequency: f = f_alpha + A·(f_gamma − f_alpha), so focus → short gamma windows (fast effective timestep), disengagement → long alpha windows. This makes "processing frequency changes with attention" literal without changing the integrator.

2. Alpha pulsed-inhibition gate (Jensen–Mazaheri). Multiply each module's input drive by g_m(t)=σ(κ·cos(2π f_alpha t+φ_m)), and set alpha amplitude κ_m INVERSELY to that module's attention weight: attended modules → low alpha → open gate (this is also mechanism (1) for "not all parts always active"). Unattended modules idle at high alpha, saving compute — a duty-cycle, not a hard off.

3. Coherence-gated inter-module routing (Fries CTC + Bastos direction). Scale each inter-module weight by phase alignment: w_eff = w · max(0, cos(φ_src−φ_dst)). Carry bottom-up spikes on the gamma-phase channel and top-down/attentional bias on a separate alpha/beta-phase channel, so feedforward and feedback are routed on different rhythms. Two modules communicate only when phase-locked — coherence becomes the connectivity switch.

4. (Optional) Theta–gamma nesting for hippocampus (Lisman–Jensen). Nest N≈7 gamma sub-slots per theta cycle; write/replay sequence items into ordered slots by gamma phase, coupling strength gated by the neuromodulator. Gives a principled sequence-encoding capacity and a natural theta "sampling vs shifting" split (Fiebelkorn–Kastner) to schedule attention hand-offs.

Implementation note: mechanisms 1–3 share ONE per-module phase variable and one attention→frequency map, so they compose cheaply and directly convert "attention" into band, gate, and routing simultaneously.

---

## Survey 2

# Sleep-Cycle Architecture: dynamics + codeable mechanisms for sapience

**Strongest understanding.** Human sleep is not a flat NREM/REM tone but a stereotyped ultradian program. Cycles of NREM→REM recur every ~90–110 min, 4–6 times/night (Feinberg & Floyd 1979). Crucially the mix *shifts across the night*: slow-wave sleep (N3) dominates the first third, REM dominates the last third (Feinberg 1974) — early cycles are SWS-heavy with brief REM; late cycles are REM-heavy with little N3. Within NREM, stages deepen N1→N2→N3. N3 (SWS) is defined by <1 Hz **slow oscillations / cortical up-down states** (Steriade et al. 1993): the "up" state is depolarized/firing, the "down" state silent.

The consolidation engine is a **nested triple coupling** timed by the slow oscillation (Diekelmann & Born 2010, "active systems consolidation"; Klinzing, Niethard & Born 2019). The SO up-state (~0.75–1 Hz) gates thalamocortical **spindles** — slow ~9–12.5 Hz and fast ~12.5–16 Hz (Mölle & Born). Spindle troughs in turn nest hippocampal **sharp-wave ripples** (~140–200 Hz rodent; ~80–100 Hz human intracranial). Staresina et al. (2015, *Nat Neurosci*) showed hierarchical phase-amplitude coupling: SO phase modulates spindle power, spindle phase modulates ripple power — SO→spindle→ripple. Helfrich et al. (2018) showed the *precision* of SO–spindle coupling (fast-spindle peak locked to SO up-state) predicts overnight retention; jitter degrades it with age. So consolidation fires in discrete windows, not continuously.

**REM** carries hippocampal/cortical **theta (4–8 Hz)**, phasic **PGO waves**, and high cholinergic tone. It does emotional/schema integration and net **synaptic downscaling/pruning** (synaptic homeostasis, Tononi & Cirelli SHY; REM broadly weakens while NREM SOs selectively downscale). So NREM = write/replay-consolidate (potentiation of replayed traces on an overall-downscaling background), REM = integrate + prune.

**Codeable mechanisms for sapience:**

1. **Stage-scheduler (replace flat NREM/REM tone).** Drive sleep from an explicit ultradian FSM: N1→N2→N3→N2→REM per cycle, cycle length ≈ 90 "ticks-equiv", 4–6 cycles. Weight the schedule across the night: `p_N3 = clamp(1 − cycle/N_cycles)`, `p_REM = clamp(cycle/N_cycles)` so early cycles allocate most steps to N3, late cycles to REM. Each stage sets a plasticity profile: N3 = high replay-consolidation + strong SHY downscale factor; N2 = spindle-gated consolidation only; REM = pruning + integration (low LR, connection-elimination pass), theta-band tone.

2. **SO-phase gate for *when* consolidation fires.** In N3/N2, run a slow-oscillation phase variable `φ_SO` at 0.75–1 Hz (relative to your tick rate). Emit a **spindle** train (12–15 Hz burst, ~0.5–1 s) only on the SO up-state (`φ_SO∈[up window]`). Fire hippocampal **replay/"ripple"** consolidation events only inside spindle troughs. Consolidation eligibility = `up_state AND in_spindle` — so e-prop eligibility traces are committed to cortex in these coincidence windows, not every tick. This directly implements the triple-coupling that times memory transfer.

3. **Coupling-precision as a learning-health knob.** Make the SO–spindle phase offset tunable; tight locking (fast-spindle peak on SO up-state) = high consolidation gain, deliberate jitter = weaker consolidation. Tie this to your existing attention/learning-health signal (poor health → widen the gate to over-consolidate; good health → precise gate), mirroring Helfrich's precision→retention finding.

Net: an ultradian **schedule** (early-SWS/late-REM), a **coincidence gate** (SO up-state ∧ spindle ∧ ripple) deciding *when* plasticity commits, and stage-specific profiles (N3 potentiate+downscale, REM prune+integrate) — replacing the current flat NREM/REM tone with a timed, stage-structured consolidation cycle.

Sources: [Staresina 2015, Nat Neurosci](https://www.nature.com/articles/nn.4119) · [Diekelmann & Born 2010, Nat Rev Neurosci](https://www.nature.com/articles/nrn2762) · [Helfrich/precise SO–spindle coupling](https://www.nature.com/articles/s41598-018-36557-z) · [ultradian cycle review](https://www.sciencedirect.com/science/article/pii/S2352721823002048) · [Klinzing/Niethard/Born systems consolidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576410/) · [PGO waves in REM](https://pmc.ncbi.nlm.nih.gov/articles/PMC10526299/)

---

## Survey 3

RESEARCH (selective-activation-states)

**Core principle.** A healthy brain is *sparsely and metastably* engaged: subsystems compete for a limited "broadcast" resource, and only winners become globally active per moment. Three literatures converge on this.

**1. Resting-state networks + metastability (anticorrelation).** Fox et al. (2005) and Raichle et al. (2001) showed the brain is intrinsically organized into *anticorrelated* systems: the default-mode network (DMN, internally-directed) and the task-positive/dorsal-attention network wax and wane in *opposition*. These are infra-slow BOLD fluctuations, **0.01–0.1 Hz** (10–100 s periods). The brain never sits in one attractor — it wanders a repertoire of metastable states (Tognoli & Kelso 2014, "the metastable brain"; Deco & Kringelbach 2016). Dynamics: mutual inhibition + slow drift ⇒ internal-vs-external modes trade off rather than co-activate.

**2. Global workspace / ignition (broadcast is selective).** Dehaene & Changeux (2011) global neuronal workspace: local processing is continuous and modular, but content becomes *globally* available only when it crosses a nonlinear **ignition** threshold ~**200–300 ms** post-stimulus (the P3b), triggering frontoparietal broadcast. Subthreshold content stays local and never engages the whole net. This is the key mechanism: *most* activity is local; a competitive threshold selects the ~one coalition that ignites and recruits everyone else.

**3. Thalamic + basal-ganglia gating.** The thalamic reticular nucleus and higher-order thalamus/pulvinar act as attentional gates (Halassa & Kastner 2017; Harris & Thiele 2011). The basal ganglia implement Go/NoGo *gating* of cortical/PFC updating — striatal Go opens a gate, NoGo holds (Frank 2001; O'Reilly & Frank 2006). Gating is multiplicative and disinhibitory: default = closed; a learned signal opens specific channels.

**4. Brain-state switching (arousal sets the regime).** Cortex oscillates between synchronized up/down states (slow oscillation **0.5–1 Hz**) and desynchronized states. Pupil-linked arousal (LC-norepinephrine, basal-forebrain ACh) tracks this: mid arousal = desynchronized, optimal; too low/high = worse (inverted-U) (McGinley et al. 2015; McCormick; Harris & Thiele 2011). Neuromodulators set *gain*, changing how easily regions ignite.

**5. Entropic brain / REBUS (the contrast case).** Carhart-Harris (2014, "revisited" 2018; REBUS, Carhart-Harris & Friston 2019): psychedelics (5-HT2A agonism) *raise* signal entropy/Lempel-Ziv, *flatten* the functional hierarchy, dissolve within-network integrity (DMN disintegrates), and *increase* between-network integration — a near-"all-on" global state. This proves normal cognition is the opposite: sparse, hierarchical, selectively gated.

**Codeable mechanisms for sapience**

1. **Ignition gate (sparse subsystem activation).** Each of the 5 systems computes a scalar *drive* per cycle (e.g., its prediction-error / salience). A soft-competition selects active systems: `active_i = drive_i > θ_ign` OR top-k winner-take-most over a shared "workspace bus." Only ignited systems write to the bus and consume compute; the rest are skipped that cycle. Reuse the existing basal-ganglia as the Go/NoGo gate producing multiplicative masks `g_i ∈ {0,1}` (or soft [0,1]).

2. **Anticorrelated DMN/task toggle.** Add a slow (infra-slow) latent oscillator, period ~10–100× the tick window, driving a two-state cross-inhibition: *internal* mode (hippocampal replay/consolidation dominant) vs *external* mode (cortex+attention dominant). They mutually suppress so both are never fully on — matches Fox 2005.

3. **Entropy/temperature knob (single tunable dial).** Let neuromod arousal set a global gain/temperature `β` that scales the ignition threshold and the softmax sharpness: low `β` ⇒ few gates open, hierarchical, sparse (normal); high `β` ⇒ thresholds flatten, many systems co-active, high between-system integration (a testable "psychedelic"/REBUS regime). Bind `β` to an inverted-U of arousal so mid-arousal is most selective/performant.

Net effect: replace always-on with drive→gate→ignite→broadcast, one scalar controlling how selective vs global the whole net is.

Sources: [Fox 2005 PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC2694109/) · [Dehaene & Changeux 2011 Neuron](https://www.unicog.org/publications/DehaeneChangeux_ReviewConsciousness_Neuron2011.pdf) · [REBUS/entropic brain](https://pmc.ncbi.nlm.nih.gov/articles/PMC6588209/) · [pupil-linked arousal/cortical state](https://elifesciences.org/articles/51501)

---

