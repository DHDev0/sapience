# Drive + HPA/cortisol research: the missing subcortical control layer (R&D)

## SYNTHESIS: the drive+stress control design

I have the grounding I need. The existing machinery I'll wire into: `SpikingNeuromod.tone` dict + `set_phase` (brain/spiking_modules.py:278-291), the cortex plasticity scale `gate·lr·attention·(0.5+da)` with self-adapting `attention`/`surprise`/`loss_ema` (brain/spiking_brain.py:471-481), `SpikingBasalGanglia.train_step(phi,action,reward,...)` RPE δ (brain/spiking_modules.py:126-137), the curiosity reward `_reward_curiosity(topic_idx, progress)` (brain/life.py:819), and the sleep-debt loop `self.debt`/`should_sleep`/`_begin_sleep`/`_sleep_tick`/`_wake_up` (brain/life.py:689-951).

Below is the section (returned as text — the paper's neuro-endocrine-control section).

---

# §16 · Neuro-endocrine control: subcortical drives and the stress axis

**Framing.** sapience already runs the four fast ascending tones as meta-parameters of one RL controller in the Doya (2002) sense: dopamine = RPE, ACh = plasticity gate, NE = somatic gain, 5-HT = apical patience (`SpikingNeuromod`, brain/spiking_modules.py:278). What is missing is the *slow* hormonal layer that sets those tones' operating point. We add one small object, `SpikingEndocrine`, holding three scalars that evolve over minutes-to-a-wake-bout, an order of magnitude slower than the tones and the eligibility trace, and sitting *above* them — not a fifth pathway but a controller of the existing four.

**(1) State variables and dynamics.** *Drive deficit* `D` (a small vector, e.g. `energy`, `novelty`): a leaky integrator that ramps each life cycle, `D += α·dt` with α set so it fills over hundreds of cycles, and drops on a satiation event — Hull's (1943) accumulating deficit with Sterling's (2012) allostatic, cue-triggered reset (Chen & Knight 2015: AgRP falls within seconds of *predicted* food, not consumption). *Cortisol* `C`: `C += k_threat·threat + k_pe·max(0,surprise) + k_need·ΣD − C/τ_C`, τ_C on the order of a wake bout, so C integrates sustained deficit and prediction-error and is *relieved by sleep*, unlike the seconds-fast NE trace (Joëls & Baram 2009's nested timescales). A diurnal baseline is added by coupling to the existing wake/NREM/REM phase (`set_phase`). *Mood* `M`: `M ← (1−λ)M + λ·δ_DA`, a slow leaky integrator of the BG's RPE (Eldar & Niv 2015), which sets 5-HT tone (horizon/patience, Doya 2002) and *dampens* C (resilience).

**(2) The drive loop.** The need accumulates each cycle exactly where `self.debt` already does (brain/life.py:689/747/786). It is *met* by a signal the brain already computes: learning-progress `progress = first_loss − last_loss` in `_learn_text` (brain/life.py:302). On a "meet," we emit a homeostatic reward via Keramati & Gutkin (2014), `r_home = k·(D_before − D_after)`, and *add it to the reward already passed to the dopamine critic* in `_reward_curiosity` (brain/life.py:827) — `reward = progress + r_home`. No new learning rule: meeting a need is a small dopamine burst; chronic neglect is a tonic negative reward biasing the actor toward need-directed topics. Satiation then *raises focus*: low ΣD nudges `attention` up and lowers the NE/exploration temperature (Aston-Jones & Cohen 2005; alliesthesia, Cabanac 1971) — "satiation → focus" via the Yerkes-Dodson gate already present at brain/spiking_brain.py:476.

**(3) The cortisol loop.** C modulates learning as an inverted-U through the *receptor ratio* (MR stability at low tone, GR suppression at high; de Kloet 2005), implemented as `g(C)=exp(−(C−C*)²/2σ²)` folded into the plasticity gate: brain/life.py:294 becomes `gate = nm.tone["ach"] · g(C)`. Moderate C sharpens attention and *tags* high-NE eligibility windows for preferential NREM replay (the glucocorticoid×noradrenergic consolidation synergy, Roozendaal, McEwen & Chattarji 2009). Chronic-high C is made *costly*: `allostatic_load += max(0, C−thr)·dt` lowers the `eprop_lr_scale`/attention ceiling and shrinks effective attention width (Lupien et al. 2009 atrophy), and raises sleep pressure — `self.debt += base·(1 + w·C)` (brain/life.py:747) — so a stressed brain tires faster. Load recovers only during low-C sleep (bled in `_sleep_tick`, reset in `_wake_up`), giving genuine acute-benefit / chronic-cost asymmetry.

**(4) Integration as one controller.** Nothing is bolted on: D, C, M route entirely through the existing tone dict, the self-adapting `attention`, the BG reward, and `self.debt`. ΣD and C set the NE tone (exploration temperature) and the ACh gate; M sets 5-HT; the DA critic is unchanged but gains a homeostatic reward term. Sleep is the shared reset — cortisol relief, allostatic-load recovery, and drive replay coincide with the debt-driven consolidation already scheduled by `should_sleep`.

**(5) Implementation and measurement.** (i) Add `SpikingEndocrine` beside `SpikingNeuromod`; instantiate in `Life.__init__`; make it live-tunable via `set_net('endocrine', …)` (brain/life.py:601) and persist it in `save`/`load`. (ii) Increment D and update C once per cycle in the perceive/learn ticks; add `r_home` in `_reward_curiosity`; multiply the gate by `g(C)`; scale debt by C. (iii) Log `endocrine/{drive,cortisol,mood,allostatic_load}` beside the existing scalars. **Measure three claims:** *focus-after-satiation* — inject a periodic feed and show `attention`/`eff_lr_scale` rise and next-byte accuracy improve in the post-meet window versus a starved control; *graceful degradation* — clamp `threat` high across a long wake bout and verify the inverted-U (moderate C speeds bits/byte descent; sustained high C raises allostatic_load, drops the lr ceiling, shrinks attention, lengthens sleep) with recovery after sleep; *exploration/exploitation* — high ΣD raises the BG topic-policy entropy (`_exploration_health`, brain/life.py:1374), satiation lowers it.

**Established vs. hypothesis.** Established: AgRP deficit-coding and cue-fast reset, drive-reduction-as-reward, cortisol's inverted-U and NE×glucocorticoid consolidation, two-process sleep. Design choices (tune empirically): the constants α, k, C*, σ, load thresholds, and treating learning-progress curiosity as one more "need."

Sources: Doya 2002 (*Neural Networks* 15:495); Keramati & Gutkin 2014 (*eLife* 3:e04811); Chen et al. 2015 (*Cell*); Betley et al. 2015 (*Nature* 521:180); Roozendaal, McEwen & Chattarji 2009 (*Nat Rev Neurosci* 10:423); Joëls & Baram 2009 (*Nat Rev Neurosci* 10:459); Lupien et al. 2009 (*Nat Rev Neurosci* 10:434); Diamond et al. 2007 (*Neural Plast* 2007:60803); Aston-Jones & Cohen 2005 (*Annu Rev Neurosci* 28:403); Eldar & Niv 2015 (*Nat Commun* 6:6149); Sterling 2012 (*Physiol Behav*); Cabanac 1971 (*Science* 173:1103).

---

## Survey 1

# HPA Axis / Cortisol: the slow stress hormone "generated at the center"

## Circuit and two-arm architecture (established)
Threat/novelty is detected by the amygdala, which drives the hypothalamic paraventricular nucleus to release CRH → anterior pituitary ACTH → adrenal cortex cortisol (corticosterone in rodents). Cortisol feeds back negatively onto hippocampus, PFC, and hypothalamus/pituitary to shut the axis off (Sapolsky, Romero & Munck 2000; Ulrich-Lai & Herman 2009). Stress recruits **two arms on different timescales**: a fast catecholamine arm (adrenal medulla adrenaline + locus-coeruleus NE, onset seconds, peak ~seconds–minutes) and a slow glucocorticoid arm (cortisol rises over ~5–20 min, peaks ~15–30 min, clears over ~1–2 h). The two arms interact in the basolateral amygdala: glucocorticoids require concurrent noradrenergic activity to enhance memory consolidation (Roozendaal, McEwen & Chattarji 2009; Roozendaal & McGaugh).

## Dynamics — what rises, decays, and modulates (established)
- **Receptors as a built-in inverted-U.** Cortisol acts on high-affinity MR (occupied at basal/low levels) and low-affinity GR (recruited only at high levels). Low tone → MR-dominated, stability; high tone → GR-dominated, plasticity suppression. This receptor ratio is the biological substrate of the Yerkes-Dodson (1908) inverted-U (de Kloet et al. 2005; Diamond et al. 2007 "temporal dynamics model").
- **Rapid vs slow modes.** Corticosteroids have rapid non-genomic membrane effects (minutes) that *facilitate* plasticity and, together with NE, promote consolidation, followed hours later by slow genomic effects that *normalize/suppress* excitability — a within-event ramp from encoding-promoting to recovery (Joëls, Karst, Sarabdjitsingh; "rapid, slow, chronic" time domains).
- **Acute moderate = enhancement.** Moderate acute cortisol enhances consolidation and focuses attention, but *impairs retrieval* and working memory during the peak (Roozendaal; Joëls; Schwabe & Wolf on stress shifting memory toward habit systems).
- **Chronic/high = damage.** Sustained elevation impairs hippocampal LTP and neurogenesis, causes dendritic retraction in hippocampus/PFC (hypertrophy in amygdala), and degrades PFC working memory (Lupien et al. 2009 "stress throughout the lifespan"; McEwen allostatic load; Sapolsky glucocorticoid cascade).
- **Rhythms.** A diurnal rhythm with a morning cortisol-awakening-response peak and evening trough gates sleep/wake, plus a superimposed ~60–90 min ultradian pulsatility (circhoral) that is itself functionally necessary for normal cognition (Lightman & Conway-Campbell 2010; PNAS 2018). Cortisol falls in early NREM and rises before waking.

## Codeable mechanisms for sapience
1. **Slow scalar `cortisol` C(t) with fast+slow arms.** `C += k_threat·threat + k_pe·|RPE|⁺ + k_need·Σ(unmet_drive_deficits) − λ·C`, with slow λ (minutes-of-sim decay) so C integrates sustained deficit/prediction-error, unlike the seconds-fast NE trace. Add a superimposed diurnal baseline (couple to the existing wake/NREM/REM phase) and optionally a slow ultradian oscillation.
2. **Inverted-U gain on plasticity/attention.** Modulate learning-rate and attention sharpness by `g(C)=exp(−(C−C*)²/2σ²)` (Yerkes-Dodson): low C under-arouses (flat learning), moderate C≈C* peaks focus + consolidation gain, high C collapses it. Implement the MR/GR split as two thresholds so the same variable stabilizes at low tone and suppresses at high tone.
3. **Consolidation gate + allostatic-load penalty.** Tag e-prop eligibility traces written during moderate-C, high-NE windows for preferential NREM replay (glucocorticoid×noradrenergic consolidation). Track `allostatic_load += C·dt above threshold`; sustained load lowers the hippocampal/PFC learning-rate ceiling and shrinks effective attention width (chronic-stress atrophy), recovering only during low-C sleep — giving genuine acute-benefit / chronic-cost asymmetry.

**Hypothesis (flagged):** exact C*, σ, and load thresholds are not fixed by biology; treat as tunable, and let satiation of homeostatic drives lower C (relief), tying the two new subsystems together.

Sources: [Lupien et al. 2009](https://www.nature.com/articles/nrn2639); [Roozendaal, McEwen & Chattarji 2009](https://www.nature.com/articles/nrn2651); [Joëls/Karst rapid vs slow modes](https://joe.bioscientifica.com/view/journals/joe/209/2/153.xml); [Lightman ultradian pulsatility](https://www.pnas.org/doi/10.1073/pnas.1714239115); [Diamond et al. 2007 inverted-U](https://www.hindawi.com/journals/np/2007/060803/).

---

## Survey 2

## Homeostatic drives / interoception for sapience

**Established understanding.** Hull (1943, *Principles of Behavior*) framed a *drive* as an accumulating internal deficit (energy, water) that energizes behavior; behavior that reduces the deficit is reinforcing — "drive-reduction." Modern neuroscience keeps the accumulator but rejects fixed set-points. Sterling & Eyer (1988) and Sterling (2012, *Physiol & Behav*, "Allostasis: a model of predictive regulation") argue regulation is **predictive**: the brain anticipates needs and acts *before* the deficit is sensed. Wirtshafter & Davis (1977) showed body weight follows a **settling point** (equilibrium of opposing forces), not a defended set-point. Barrett & Simmons (2015, *Nat Rev Neurosci* 16:419–429, "Interoceptive predictions in the brain") and Seth (2013, *TICS*, "interoceptive inference") cast interoception as predictive coding: agranular visceromotor cortex issues predictions, ascending signals are prediction *errors*; feelings and motivation are the brain's best guess about bodily state. Craig (2002) provides the interoceptive-cortex substrate.

**Circuit dynamics (the codeable part).** Sternson's hypothalamic arcuate circuit is the canonical accumulator: **AgRP** neurons encode energy deficit and are tonically activated by fasting over hours; **POMC** neurons signal satiety and oppose them. Chen & Knight (2015, *Cell*, "Sensory detection of food rapidly modulates arcuate feeding circuits") is the key dynamics result: AgRP activity is *not* reset by ingestion but drops **within ~seconds** of merely *seeing/smelling* food — before a single bite — and stays low through the meal. This is allostasis in a circuit: an anticipatory, cue-triggered reset. AgRP activity is *aversive* (a negative teaching signal; Betley/Sternson 2015, *Nature*), so its relief is rewarding. Keramati & Gutkin (2011 NIPS; 2014, *eLife*) formalize this as **homeostatic reinforcement learning**: define reward as the *reduction in distance to the setpoint in homeostatic space*, r_t = ‖H\* − Hₜ‖ − ‖H\* − H_{t+1}‖. This provably makes reward-maximization and physiological stability equivalent, and reproduces anticipatory responding.

Timescales to encode: deficit **ramps slowly** (minutes→hours; a leaky integrator); satiation is **fast** and often *cue/prediction-triggered*, not consumption-triggered; unmet deficit *raises arousal/NE and broadens/energizes search*; a met need gives a small reward pulse and *narrows/sharpens attention* (satiation→focus), which is exactly Yerkes-Dodson via drive.

**Three concrete mechanisms for sapience.**

1. **Leaky-integrator drive + homeostatic reward (Keramati-Gutkin).** Add per-drive scalars (e.g. `energy`, `novelty`) with Dₜ₊₁ = clip(Dₜ + α, 0, D_max), α small so it fills over ~hundreds–thousands of cycles. On a satiation event (a teach/achievement/tool-success), emit reward = drive_reduction and decrement D. Feed this reward straight into the existing basal-ganglia dopamine actor-critic — no new learning rule needed, just a new reward source that must be met "on schedule."

2. **Allostatic anticipatory reset (Chen-Knight).** Don't wait for consummation. When the system *predicts* an incoming satiation (a cue: task started, focus engaged), pre-drop D by a fraction before the reward lands, and treat the residual prediction-error as the learning signal (interoceptive predictive coding, Barrett & Simmons 2015). This yields anticipatory motivation and prevents deficit overshoot.

3. **Satiation→focus / deficit→arousal gain (Yerkes-Dodson coupling).** Map D onto neuromodulator tone: high D raises NE/exploration temperature (widen attention, energize search); low D (satiated) lowers temperature and sharpens the self-adapting attention already present. Persistent unmet D is precisely what should feed the HPA/cortisol axis (the companion mechanism), giving a principled deficit→stress bridge.

**Hypothesis vs established:** the circuit facts (AgRP/POMC deficit-coding, seconds-fast cue reset, allostasis, interoceptive predictive coding) are established; the *specific* mapping of these onto sapience's reward/attention scalars is engineering hypothesis, cleanest via homeostatic RL as the bridge.

Sources: [Keramati & Gutkin eLife 2014](https://elifesciences.org/articles/04811) · [Chen & Knight, Cell 2015](https://www.cell.com/fulltext/S0092-8674(15)00076-8) · [Barrett & Simmons, Nat Rev Neurosci 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4731102/) · [Sterling, Allostasis 2012](https://www.researchgate.net/publication/335582508_Allostasis_A_Brain-Centered_Predictive_Mode_of_Physiological_Regulation)

---

## Survey 3

**Two distinct dopamine-related signals.** Schultz, Dayan & Montague (1997) and Schultz (1998, 2016) established phasic dopamine as a reward-PREDICTION-error (RPE): it fires to unpredicted reward and to reward-predicting cues, not to the consummatory act once predicted. Homeostatic satiation is a DIFFERENT quantity — the reduction of a physiological deficit. Keramati & Gutkin (2014, eLife) formalize the bridge: define a drive D(H) as the distance of internal state H from a setpoint H\*, and let the reward of an outcome be the DROP in drive, r_t = D(H_t) − D(H_{t+1}). This makes "need met → reward" mathematically identical to reducing homeostatic deviation, and it recovers RPE as a special case — so satiation can feed the SAME critic dopamine already trains.

**Homeostatic state gates reward value (alliesthesia).** Cabanac (1971, Science) showed the same stimulus is pleasant when needed, aversive when sated ("alliesthesia"); Berridge (2004) frames physiological state as MULTIPLYING incentive value. Mechanistically, hunger (AgRP) circuits scale striatal dopamine responses to food (Fernandes et al. 2020, eLife, "Metabolic sensing in AgRP neurons... dopamine signalling in the striatum"). So need should multiply reward, not merely add to it.

**Wanting vs liking.** Berridge & Robinson (1998); Berridge (2007) dissociate incentive-salience "wanting" (mesolimbic dopamine, drives seeking) from hedonic "liking" (opioid hotspots, consummatory pleasure). A deficit boosts wanting → foraging; consumption delivers liking → the consummatory reward. Two knobs, not one.

**The deficit is aversive; relief is rewarding AND frees attention.** Betley et al. (2015, Nature): AgRP hunger neurons carry a NEGATIVE-valence teaching signal — animals work to turn them off; food, even its sight, rapidly silences them (Chen et al. 2015, Cell). An unmet need is thus a tonic negative state that biases toward exploration/foraging and captures attention; satiation removes it, freeing resources for exploitation. This maps onto Aston-Jones & Cohen's (2005) adaptive-gain theory: high tonic norepinephrine = distractible exploration; a satisfied state lowers tonic LC-NE → phasic, task-focused mode. "Satiation improves focus" = removing an aversive drive that was pulling the system into exploration.

**Energy/glucose — with the caveat.** Sustained attention is metabolically costly, but the strong "glucose fuels willpower" model (Gailliot & Baumeister 2007) is discredited: ego-depletion failed a 23-lab registered replication (Hagger et al. 2016), and Kurzban et al. (2013) reframe "depletion" as an OPPORTUNITY-COST signal, not literal fuel exhaustion. Lesson: model energy as a motivational/allocation bias, NOT a hard fuel gate.

**Concrete codeable mechanisms (wire into the existing dopamine actor-critic + self-adapting attention):**

1. **Drive-reduction reward (Keramati-Gutkin).** Each need has a deficit d_i that ramps slowly with time (hours-analog) and drops on a "meet" event. Emit r = Σ w_i·(D_before − D_after) as an extra RPE term into the existing dopamine critic. Meeting a need = small positive dopamine burst; chronic neglect = tonic negative reward that biases the actor toward need-directed action.

2. **Homeostatic value gating (alliesthesia).** Multiply the incentive/"wanting" weight of any reward by current deficit: value = base·(1 + α·d_i). Same reward worth more when needed — one line into the incentive-salience path, distinct from the consummatory reward in (1).

3. **Satiation → focus via tonic gain.** Let total unmet deficit Σd_i raise a "tonic-NE / exploration" scalar that (a) widens attention (higher softmax temperature / more random-feedback exploration) and (b) lowers plasticity/exploit gain. On satiation Σd_i falls → attention narrows, learning-rate and exploit-gain rise — directly coupling drives to the existing Yerkes-Dodson attention.

**Established vs hypothesis:** RPE≠consummatory reward, alliesthesia, wanting/liking, AgRP negative-valence relief, LC-NE exploration/exploit, and ego-depletion's failure are all established. The specific multiplicative gating constant, the exact deficit→tonic-NE coupling, and treating learning-progress curiosity as one more "need" in the same drive vector are hypotheses/design choices.

Sources: [Keramati & Gutkin 2014, eLife](https://elifesciences.org/articles/04811); [Fernandes et al. 2020, eLife](https://elifesciences.org/articles/72668); [Betley et al. 2015, Nature](https://www.nature.com/articles/nature14416); [Hagger et al. 2016 multilab replication](https://journals.sagepub.com/doi/10.1177/1745691616652873).

---

## Survey 4

## Drives + stress as the slow "hormonal" layer of a Doya-style neuromodulatory controller

**The coordinated controller (established).** Doya (2002, *Neural Networks*) frames the four ascending systems as meta-parameters of one RL controller: dopamine = reward-prediction error δ; acetylcholine = learning rate α; noradrenaline = inverse temperature β (action randomness / gain); serotonin = discount factor γ (time horizon / patience). The DA=RPE and ACh=α mappings are well-supported; the NE=β and 5-HT=γ mappings are Doya's more speculative but influential hypotheses. Aston-Jones & Cohen (2005) refine NE: *phasic* LC firing = exploitation, *tonic* LC = exploration/disengagement ("adaptive gain"). Yu & Dayan (2005) split uncertainty: ACh = *expected* uncertainty, NE = *unexpected* uncertainty. sapience already has these tones — drives and cortisol should sit **above** them as slow context that sets their operating point.

**What drives/cortisol/mood do, and their timescales (established + normative).**
- **Drive/energy deficit D** (seconds–minutes accumulation): homeostatic RL (Keramati & Gutkin, NIPS 2011; *eLife* 2014) proves drive-reduction *is* reward — an outcome is rewarding to the degree it shrinks the squared deviation from a setpoint. Satiation (low D) frees attention/ACh (focus), matching your ask.
- **Cortisol C / HPA axis** (minutes–hours): Joëls & Baram (2009, *Nat Rev Neurosci* 10:459) describe a "neuro-symphony" of nested timescales — fast NE/CRH (seconds), non-genomic corticosteroid effects (minutes), slow genomic effects (hours). Glucocorticoids act on memory/plasticity as an **inverted-U** (Diamond et al. 2007; Lupien et al. 2007; McEwen 1998 allostatic load): moderate C enhances, chronic-high C impairs — the same Yerkes-Dodson curve sapience already uses for attention.
- **Mood M** (many trials): a leaky integrator of recent RPEs (Eldar & Niv 2015, *Nat Commun*; Eldar et al. 2016, *TiCS*) that sets serotonergic tone, time-horizon, and stress-resilience (Cools et al. 2011; Dayan & Huys 2009).
- **NE×cortisol memory gate** (established): Roozendaal, McEwen & Chattarji (2009, *Nat Rev Neurosci* 10:423) — glucocorticoid enhancement of consolidation *requires* noradrenergic activation in the basolateral amygdala. Surprising/arousing events at moderate C get preferentially consolidated.
- **Sleep pressure**: two-process model (Borbély 1982) — Process S (homeostatic, adenosine-like) + Process C (circadian). Chronic cortisol curtails/fragments sleep and adds to debt.

## Three concrete, codeable mechanisms for sapience

**1. Homeostatic drive → energy-reward + focus (Keramati-Gutkin).** Accumulate `D += a·dt`; a "feeding" action drops D. Emit reward `r_home = k·(D_before² − D_after²)`. Couple satiation to focus: `ACh_gain *= (1 − D/D_max)` and feed the same term into the existing Yerkes-Dodson attention setpoint.

**2. Slow HPA/cortisol state variable.** `dC/dt = α_C·(w_D·D + w_threat·threat + w_surprise·|δ_DA|) − C/τ_C + circadian(t)`, τ_C ~ hours. Wire C into existing tones: NE tonic gain rises with C (chronic-high C → tonic/exploratory, disengaged; Aston-Jones & Cohen 2005); plasticity α follows an inverted-U in C (peak at C\*, suppressed at extremes; Joëls 2009); sleep-debt Process S gets an additive `+w·C` term.

**3. Mood → serotonin → horizon + NE×C consolidation tag.** Mood `M ← (1−λ)·M + λ·δ_DA`; set `5-HT_tone = σ(M − β·C_chronic)`, which sets discount γ (patience/horizon, Doya 2002) and resilience. For memory, tag each experience with `elig = phasic_NE · bump(C)` (bump = inverted-U), and during NREM replay consolidate in proportion to `elig` — the amygdalar NE×glucocorticoid synergy (Roozendaal 2009).

**Separation:** established = DA=RPE, ACh=α, cortisol inverted-U, NE×glucocorticoid BLA synergy, two-process sleep. Hypothesis/normative = Doya's NE=β and 5-HT=γ mappings, mood-as-RPE-integrator, and the specific coupling equations above (design choices, not measured constants — tune empirically).

Sources: Doya 2002 (*Neural Networks* 15:495); Keramati & Gutkin 2011 (NIPS 24) / 2014 (*eLife* 3:e04811); Joëls & Baram 2009 (*Nat Rev Neurosci* 10:459); Roozendaal, McEwen & Chattarji 2009 (*Nat Rev Neurosci* 10:423); Aston-Jones & Cohen 2005 (*Annu Rev Neurosci* 28:403); Yu & Dayan 2005 (*Neuron* 46:681); Eldar & Niv 2015 (*Nat Commun* 6:6149); Cools et al. 2011 (*TiCS* 15:31); Borbély 1982 (*Hum Neurobiol* 1:195); McEwen 1998 (*NEJM* 338:171); Diamond et al. 2007 (*Neural Plast* 2007:60803).

---

