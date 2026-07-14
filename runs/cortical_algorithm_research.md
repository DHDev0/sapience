# Cortical-algorithm research — surveys + synthesis (background R&D workflow, for the paper)

## Synthesis: toward the true cortical algorithm

# The "true cortical algorithm": one algorithm or competing hypotheses?

**Honest state: no confirmed single algorithm.** There is broad agreement that the neocortex repeats a stereotyped laminar microcircuit (Mountcastle; Douglas & Martin's canonical microcircuit; Harris & Shepherd 2015), which motivates the *hope* for one algorithm — but the algorithm itself is contested. Two largely orthogonal debates are live: (a) what objective cortex optimizes, and (b) how it assigns credit. The leading candidates are not mutually exclusive.

**1. Predictive coding / predictive processing** (Rao & Ballard 1999; Friston's free-energy principle; Keller & Mrsic-Flogel 2018, "Predictive processing: a canonical cortical computation"). *Best-supported at the phenomenon level:* two-photon imaging shows robust, experience-dependent mismatch/prediction-error responses in L2/3 of mouse visual cortex, gated by SST/VIP interneurons, with thalamocortical contributions (Furutachi, Mrsic-Flogel, Hofer et al., Nature 2024). *Contested at the algorithm level:* the 2025 *Rethinking Predictive Processing* review (Annual Review of Neuroscience) and "Predictive coding: a distinction without a difference" argue error-like responses are consistent with several models, so strict PC's specific circuitry (subtractive error units, precise wiring) is not uniquely confirmed.

**2. Backprop-approximation family / NGRAD hypothesis** (Lillicrap, Santoro, Marris, Akerman & Hinton 2020, *Nature Reviews Neuroscience*): cortex approximates gradient descent using locally computed activity *differences* rather than explicit error transport. Sub-variants: feedback alignment (Lillicrap 2016), predictive coding approximates backprop (Whittington & Bogacz 2017; Millidge 2021), e-prop (Bellec 2020), dendritic microcircuits (Sacramento 2018), burst-dependent plasticity (Payeur, Guerguiev, Zenke, Richards & Naud, *Nat. Neuro.* 2021). *New, strong evidence:* "Vectorized instructive signals in cortical dendrites" (Nature, April 2026) reports the first biological evidence of *per-neuron* instructive/error signals in L5 apical dendrites during a BCI task, whose signs predict learning and whose optogenetic perturbation disrupts it — direct support for vectorized, dendrite-based credit assignment.

**3. Thousand Brains / sensorimotor reference-frame theory** (Hawkins/Numenta 2019; Thousand Brains Project, arXiv 2024): each column learns a full object model via grid-cell-like reference frames, perception by voting. *Influential but least experimentally constrained* — a framework/engineering program, not yet discriminated by physiology (its "old-brain/new-brain" framing is criticized).

**Settled:** repeated laminar microcircuit; feedforward+feedback hierarchy; superficial-layer prediction-error-like signals under learned expectation; three-factor, dendrite-gated local plasticity. **Open:** whether there's one objective; whether cortex is gradient-following in vivo; subtractive-PC vs vectorized-instructive error; columns as full models vs feature detectors.

**What discriminates them / moves toward "confirmed":** measure and causally perturb apical-dendritic signals during learning (done 2026 → favors vectorized error); test feedback symmetry — Kolen-Pollack-like weight tracking vs random DFA — via connectomics + physiology; distinguish subtractive vs divisive error units and SST/VIP roles; demonstrate synaptic change follows a gradient of a global objective.

**Three concrete things sapience could add next:**

1. **Typed PV/SOM/VIP interneurons wired as a PC microcircuit** (on your known-not-done list). This is exactly the live experimental frontier (SST/VIP-gated L2/3 error). It lets sapience reproduce mismatch-response phenomenology and measure the bits/byte cost of a genuine predictive-coding microcircuit.

2. **Instrument the feedback-alignment debate directly.** Log feedback↔forward weight alignment over training on your capability-vs-fidelity curve (KP vs DFA), and add a *vectorized instructive-signal* mode — a per-neuron dendritic teaching signal à la Payeur 2021 / Nature 2026 — as another toggleable constraint. This turns sapience into a self-experiment mirroring the current frontier.

3. **Close a minimal sensorimotor loop with a grid-like location code** (also known-not-done) to test the Thousand Brains claim inside the same harness — converting an untestable framework into a measurable bits/byte cost of adding action + a reference frame.

Sources:
- [Lillicrap, Santoro, Marris, Akerman & Hinton, "Backpropagation and the brain," Nat Rev Neurosci 2020](https://go.gale.com/ps/i.do?asid=34591c14&id=GALE%7CA624630406&it=r&p=AONE&sid=googleScholar&u=googlescholar&v=2.1)
- [Whittington & Bogacz, "On the relationship between predictive coding and backpropagation"](https://arxiv.org/pdf/2106.13082)
- [Keller & Mrsic-Flogel, "Predictive Processing: A Canonical Cortical Computation" (UCL copy)](https://discovery.ucl.ac.uk/10064516/3/Keller_Mrsic-Flogel.pdf)
- ["Rethinking Predictive Processing," Annual Review of Neuroscience](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-102124-031410)
- ["Predictive coding: A distinction without a difference," PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12221311/)
- [Payeur, Guerguiev, Zenke, Richards & Naud, "Burst-dependent synaptic plasticity...," Nat Neurosci 2021 (Zenke Lab)](https://zenkelab.org/2021/05/paper-burst-dependent-synaptic-plasticity-can-coordinate-learning-in-hierarchical-circuits/)
- ["Vectorized instructive signals in cortical dendrites," Nature, April 2026](https://www.nature.com/articles/s41586-026-10190-7)
- [Current Biology 2026, "Neuron-by-neuron error signals in the neocortex"](https://www.cell.com/current-biology/abstract/S0960-9822(26)00509-9?rss=yes)
- [Thousand Brains Project, arXiv 2024](https://arxiv.org/pdf/2412.18354)
- [Predictive Coding: a Theoretical and Experimental Review, arXiv](https://arxiv.org/pdf/2107.12979)

---

## Survey 1

## Embodiment / closed sensorimotor loop as the deepest "setup" gap

**The strongest claim in the literature is that the loop, not the synapse, is where grounding lives.** Held & Hein's 1963 kitten-carousel remains the cleanest evidence: two kittens receive *identical* visual input, but only the one whose movement *generated* that input develops normal visually-guided behavior. The passively-moved kitten, with the same photons and presumably the same plasticity rules, does not learn. Learning was gated by the *contingency* between self-produced action and sensory change, not by the learning rule ([Held & Hein 1963, review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248214/)). This is the empirical anchor for the whole "it's about the setup" thesis.

O'Regan & Noë (2001) formalize the object of learning as **sensorimotor contingencies** — lawful rules mapping actions to sensory change — and argue perception *is* mastery of those rules, not a representation computed from a static input ([SMC theory](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.624610/full)). Friston's active inference / free-energy framing makes the same point mechanistically: an agent minimizes prediction error (variational free energy) by *two* routes — changing beliefs (perception) **or** changing the world so it matches predictions (action) — and a genuine agent is defined by a *closed* perception–action loop where actions alter the environment that generates the next input ([Active Inference, Parr/Pezzulo/Friston 2022](https://mitpress.mit.edu/9780262045353/active-inference/); [Physical AI engineering perspective 2026](https://arxiv.org/pdf/2603.20927)). "Offline" systems that only consume a fixed input stream are, by this definition, not agents. Oudeyer and Schmidhuber add the *drive*: without extrinsic reward, an agent that maximizes **learning progress / prediction-error reduction** (curiosity) self-structures a developmental curriculum, seeking inputs at the edge of its competence ([Oudeyer](https://www.pyoudeyer.com/curiosity-and-information-seeking-in-cognitive-development/); [Schmidhuber formal creativity](https://people.idsia.ch/~juergen/ieeecreative.pdf)). Modern world-model agents (DreamerV2/V4, Hafner et al. 2020–2025) show this scales computationally: learn a generative model, act to reduce its uncertainty, learn from the consequences ([Dreamer 4](https://www.emergentmind.com/papers/2509.24527)).

**Settled:** self-generated action changes what is learnable from identical sensory statistics (Held/Hein); perception–action must be closed for grounding (SMC, active inference); curiosity/learning-progress can replace external reward as the loop's objective. **Open:** whether the *specific* free-energy math is necessary vs. one instance of a broader principle; how much embodiment must be physical vs. informational; whether language tokens can carry real sensorimotor contingencies or only simulate them.

**2–3 concrete additions for sapience** (it already reads bytes and talks to an LLM, so a minimal loop is cheap):

1. **Close the loop through the LLM as environment.** The brain's byte output → LLM → LLM's reply becomes the brain's *next* input. This alone makes outputs contingently reshape input — the minimal O'Regan/Held condition. Log the action→next-input map as explicit sensorimotor-contingency traces.

2. **Add a prediction-error / free-energy channel as the third factor.** The brain already has three-factor neuromodulation and e-prop eligibility. Gate plasticity on *surprise* = mismatch between the brain's predicted next byte and the byte actually returned by the loop. This turns the existing eligibility traces into active inference without new plasticity machinery.

3. **Intrinsic-motivation drive on learning progress.** Give the action-selection head a reward = *reduction* in byte-prediction error over a short window (Oudeyer/Schmidhuber). The brain then emits outputs that make its own future input maximally *learnable* — a curriculum it authors, closing action→perception→plasticity→action.

Sources: [Held/Hein review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248214/), [SMC theory](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.624610/full), [Active Inference (MIT Press)](https://mitpress.mit.edu/9780262045353/active-inference/), [Physical AI active inference](https://arxiv.org/pdf/2603.20927), [Oudeyer curiosity](https://www.pyoudeyer.com/curiosity-and-information-seeking-in-cognitive-development/), [Schmidhuber](https://people.idsia.ch/~juergen/ieeecreative.pdf), [Dreamer 4](https://www.emergentmind.com/papers/2509.24527).

---

## Survey 2

## Predictive coding & active inference vs. approximate-gradient learning

**The core claim and how it differs.** Rao & Ballard (1999) proposed cortex as a hierarchical generative model in which each level *predicts* the level below and only the residual **prediction error** propagates upward; Friston (2005, 2010) reframed this as variational free-energy minimization, adding *precision* (inverse-variance) weighting of errors and, in active inference, action that changes sensory input to fulfill predictions. The structural difference from approximate-gradient methods (backprop, feedback-alignment/DFA, e-prop) is *what the learning signal means*: PC's signal is a **local, layer-wise generative mismatch** (predict-your-input) computed at every level by dedicated error units, whereas DFA/e-prop propagate a **single global readout error** (target − output) back through fixed or random weights. PC therefore looks more faithful — everything is local, self-supervised, and needs no global target.

**Settled: the equivalence results, and their double edge.** Millidge, Tschantz & Buckley (2020) showed PC approximates backprop along arbitrary computation graphs; Song et al. (2020) "Z-IL" and Rosenbaum (2022) showed PC can compute **exactly** backprop's gradients under specific schedules. This is the strongest theoretical result — but it cuts both ways: the configurations that match backprop **reintroduce weight transport** (symmetric backward weights) and **separate error-carrying units**, the very biological implausibilities PC was meant to avoid (Millidge et al. 2022 review, arXiv:2107.12979). Random/Hebbian-trained backward weights (Kolen-Pollack) relax this but sacrifice exactness — the same trade sapience already exposes on its DFA-vs-KP toggle.

**Open: does cortex actually do PC?** Evidence is real but contested. Genuine prediction-error signals exist — stimulus-specific PE neurons in mouse auditory cortex (Audette & Schneider 2023, *J. Neurosci.*), expectation-violation responses in mouse V1 (Garrett et al.), and a thalamocortical disinhibitory circuit that Mrsic-Flogel and colleagues (2024, *Nature*) show is *required* to generate V1 prediction errors — supporting Keller & Mrsic-Flogel's (2018) mismatch framework. But the 2025 *Annual Reviews of Neuroscience* "Rethinking Predictive Processing" argues definitions are slippery and "expectation suppression" often has non-PC explanations (adaptation, attention). **Settled:** cortex carries mismatch/error signals and top-down predictions. **Open:** whether it implements *canonical* PC (explicit error units, precision-weighted variational inference) rather than looser predictive processing.

**Concrete next steps for sapience.**

1. **Add a PC objective as a toggle, distinct from delivery.** Sapience already has apical/basal compartments and burst error delivery — that is *how* errors arrive, orthogonal to *what* they encode. Repurpose the existing top-down feedback matrix as a **generative prediction pathway** (apical = top-down prediction, basal = feedforward drive, soma = local error), following Sacramento et al. (2018) dendritic microcircuits. Then measure PC's bits/byte on the capability-vs-fidelity curve *against* the global-readout DFA/e-prop signal — a direct test of the "local is more faithful" claim.

2. **Precision-weighting via the existing neuromodulator.** Make error gain state-dependent by routing PC precision through sapience's three-factor neuromodulation — the cleanest, cheapest PC ingredient to add.

3. **Typed interneurons to build signed error units.** Negative prediction errors need a PV/SOM/VIP microcircuit (Hertäg & Sprekeler 2020, *eLife* "Learning prediction error neurons in a canonical interneuron circuit"). Typed interneurons are already on sapience's "not-done" list — implementing them is the concrete substrate for separate positive/negative error populations, the missing piece for genuine (not backprop-equivalent) PC.

**Sources:**
- [Millidge, Tschantz & Buckley (2020), PC approximates backprop along arbitrary graphs](https://arxiv.org/pdf/2006.04182)
- [Song et al. (2020), PC can do exact backprop (Z-IL)](https://arxiv.org/pdf/2103.03725)
- [Rosenbaum (2022), On the relationship between PC and backprop, PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0266102)
- [Millidge et al. (2021/2022), Predictive Coding: A Theoretical and Experimental Review](https://arxiv.org/pdf/2107.12979)
- [Rethinking Predictive Processing, Annual Reviews of Neuroscience (2025)](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-102124-031410)
- [Mrsic-Flogel et al. (2024), Cooperative thalamocortical circuit for sensory prediction errors, Nature](https://www.nature.com/articles/s41586-024-07851-w)
- [Audette & Schneider (2023), Stimulus-Specific Prediction Error Neurons in Mouse Auditory Cortex, J. Neurosci.](https://www.jneurosci.org/content/43/43/7119)
- [Hertäg & Sprekeler (2020), Learning prediction error neurons in a canonical interneuron circuit, eLife](https://elifesciences.org/articles/57541)
- [Predictive Coding Networks: Tutorial and Survey (2024)](https://arxiv.org/pdf/2407.04117)

---

## Survey 3

## NGRAD / approximate-gradient family: how cortex likely assigns credit

**The unifying frame.** Lillicrap, Santoro, Marris, Akerman & Hinton (2020, *Nat. Rev. Neurosci.*, "Backpropagation and the brain") argue that exact backprop is biologically impossible — it requires weight transport, a separate error-delivery network, signed high-precision errors, and a frozen forward pass — but that a whole family of rules approximate its gradients using **differences in neural activity** driven by feedback. They named this class **NGRAD** (Neural Gradient Representation by Activity Differences). Feedback alignment (FA), target propagation, equilibrium propagation, predictive coding, and burst-prop are all NGRAD instances.

**What is settled.**
- *Weight transport is not required.* Random fixed feedback still learns (Lillicrap et al. 2016); at scale the forward/feedback **sign concordance** is what matters — sign-symmetry (Xiao et al. 2019) and weight mirrors (Akrout et al. 2019) nearly match backprop on ImageNet. This is robust.
- *Plain FA does not scale.* Bartunov et al. (2018) showed FA fails on hard visual tasks; this motivated feedback learning (weight mirror / Kolen–Pollack).
- *Cortex has the architectural substrate.* Layer-5 pyramidal neurons segregate **basal (feedforward)** from **apical (top-down/context)** compartments and signal via bursts — the substrate dendritic-error and burst-prop rules assume (Guerguiev, Lillicrap & Richards 2017; Sacramento, Costa, Bengio & Senn 2018).

**Strongest current hypotheses.**
- **Burst-dependent plasticity** (Payeur, Guerguiev, Zenke, Richards & Naud 2021, *Nat. Neurosci.*): the same axon multiplexes a feedforward event-rate and a feedback-driven **burst probability** that acts as a local, layer-wise error — no separate error network, no explicit subtraction. This is currently the most mechanistically concrete cortical proposal.
- **Prospective configuration** (Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz 2024, *Nat. Neurosci.*): the network first *relaxes* activity toward a target-consistent configuration, then applies Hebbian plasticity on the settled difference. It is more sample-efficient, scales better with depth, and matches neural/behavioral data better than feedforward gradient rules — the family that also contains predictive coding (Whittington & Bogacz 2017) and equilibrium propagation (Scellier & Bengio 2017; Laborieux et al. 2021).
- **Temporal credit** via eligibility traces: e-prop (Bellec et al. 2020, *Nat. Commun.*) truncates BPTT into local traces × a top-down learning signal.

**What is open.** No direct evidence that apical signals are *signed layer-wise gradients* used for plasticity — Larkum-lab in-vivo data show **unsigned, TD/salience-like** apical signals (Gillon et al.; biorxiv 2021), not backprop errors. The **loss/target source** (self-supervised, predictive, RL?) is unknown. Whether cortex actively **aligns feedback weights** is unconfirmed. And e-prop's truncation *discards recurrent sensitivity* — genuine long-range temporal credit remains unsolved.

**Three concrete things sapience could add next.**
1. **A short relaxation/inference micro-phase** (prospective configuration / EP style): sapience is forward-in-time only; let activity settle under top-down nudging for a few iterations *before* applying the e-prop update, then plot sample-efficiency on the existing capability-vs-fidelity curve. This is the biggest lever and complements e-prop's temporal traces with spatial relaxation.
2. **True burst multiplexing + short-term plasticity** (Payeur 2021): sapience already has apical/burst error delivery but explicitly lacks STP; add facilitating/depressing apical synapses so one channel carries both feedforward rate and a signed burst-error, removing the separate feedback path.
3. **Measure and learn feedback alignment:** sapience toggles DFA vs Kolen–Pollack — log the **cosine angle** between feedback and forward-transpose as a fidelity axis, and test dis-inhibitory sign control (Bhatia/Zenke, NeurIPS 2023) using its still-missing PV/SOM/VIP interneurons to gate apical error sign.

Sources:
- [Lillicrap et al. 2020, Backpropagation and the brain (Nat. Rev. Neurosci.)](https://ora.ox.ac.uk/objects/uuid:862189c1-0088-4f78-b17a-2748c2019209/download_file?safe_filename=Lillicrap_v6_2020.pdf&type_of_work=Journal+article)
- [Payeur, Guerguiev, Zenke, Richards, Naud 2021 (Nat. Neurosci.)](https://pubmed.ncbi.nlm.nih.gov/34728832/)
- [Song, Millidge, ... Bogacz 2024, Prospective configuration (Nat. Neurosci.)](https://www.nature.com/articles/s41593-023-01514-1)
- [Akrout et al. 2019, Weight mirrors / Deep learning without weight transport](https://arxiv.org/pdf/1904.05391)
- [Bartunov et al. 2018, Bio-plausible algorithms scaling](https://arxiv.org/pdf/1811.03567)
- [Laborieux et al. 2021, Scaling Equilibrium Propagation](https://arxiv.org/abs/2101.05536)
- [Ernoult et al. 2022, Scaling Difference Target Propagation](https://proceedings.mlr.press/v162/ernoult22a/ernoult22a.pdf)
- [Bellec et al. 2020, e-prop (semanticscholar)](https://www.semanticscholar.org/paper/Eligibility-traces-provide-a-data-inspired-to-time-Bellec-Scherr/c539630c4cae92f0e14eb0918a980313090a6351)
- [Unsigned TD errors in L5 dendrites during learning (biorxiv 2021)](https://www.biorxiv.org/content/10.1101/2021.12.28.474360.full.pdf)
- [Dis-inhibitory circuits control the sign of plasticity (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/ca22641c182b3b9608634edb4d09bc33-Paper-Conference.pdf)

---

## Survey 4

## Toward the true cortical algorithm: where sapience sits, and the next move

**(1) Position.** There is no confirmed single cortical algorithm — only two orthogonal live debates, *what* cortex optimizes (the objective) and *how* it assigns credit (the learning rule), over a substrate everyone agrees on: a repeated laminar microcircuit with feedforward+feedback hierarchy, superficial-layer prediction-error-like signals, and three-factor, dendrite-gated local plasticity (Harris & Shepherd 2015; Keller & Mrsic-Flogel 2018; Lillicrap et al. 2020). On the *learning-rule* axis sapience is at the frontier: it already runs forward-in-time e-prop eligibility (Bellec et al. 2020), a togglable random/learned feedback matrix (DFA vs Kolen–Pollack), apical/burst dendritic error delivery (Payeur et al. 2021), and three-factor neuromodulation — precisely the ingredients the NGRAD family posits. On the *objective* and *setup* axes it is mid-field: its top-down matrix delivers a single global readout error, not a local generative prediction, and it consumes a byte stream rather than closing a loop. Its realism axes (Dale's law, Fusi bounds, homeostasis) are partial.

**(2) The one highest-value move: typed PV/SOM/VIP interneurons that gate the apical compartment.** This is the connective tissue between three ingredients sapience already has but keeps separate — apical/burst error delivery, the feedback matrix, and neuromodulation — and it is exactly the live experimental frontier: SST/VIP-gated L2/3 mismatch responses (Furutachi, Mrsic-Flogel, Hofer et al., *Nature* 2024), disinhibitory control of the *sign* of plasticity (Bhatia & Zenke, NeurIPS 2023), and canonical interneuron circuits that learn to compute prediction-error neurons (Hertäg & Sprekeler, *eLife* 2020). Why it beats the alternatives: BTSP-proper and the realism axes mostly buy faithfulness, not new learning; and the two other high-value moves — a PC-vs-readout objective knob and a prospective-configuration relaxation micro-phase (Song et al. 2024) — actually *depend* on this circuit. You cannot build separate signed positive/negative error units, or make apical error sign state-dependent, without a typed disinhibitory microcircuit. It converts sapience from a system that delivers one global error into one that can represent local, signed, precision-weighted error — the difference between loose predictive processing and canonical PC, and the substrate for the vectorized per-neuron instructive signals just reported in L5 apical dendrites (*Nature*, April 2026).

**(3) Honesty: faithfulness is asymptotic.** There is no finish line where sapience "is" the cortical algorithm; each axis independently narrows a measurable gap, and the paper's real contribution is separating axes that move the capability curve from those that only add realism. **Group A (changes learning):** e-prop eligibility, feedback alignment (DFA↔KP), apical error delivery, neuromodulation, BTSP long eligibility, a PC objective, a relaxation micro-phase, and closing the sensorimotor loop. **Group C (realism at a bits/byte cost, little learning change):** Dale's law, Fusi bounds, stochastic spiking + metabolic cost, STDP-kernel detail, sharp-wave ripples, neurogenesis, dendritic morphology. Typed interneurons are decisive precisely because they sit on the A/C boundary — added for realism, but wired to gate error sign they become a group-A intervention. One deeper lever is orthogonal to all of this: Held & Hein (1963) show identical sensory statistics yield different learning depending on whether *action generated them* — grounding may live in the loop, not the synapse (O'Regan & Noë 2001; active inference, Parr, Pezzulo & Friston 2022). We flag embodiment as the largest but riskiest bet, and place it on the roadmap rather than the critical path.

**(4) Experiments to run next (capability vs fidelity, all on the existing bits/byte curve):**
- **Sign-gating:** add PV/SOM/VIP; route VIP-disinhibition to set apical error sign; compare signed local-PC error against global-readout DFA; ablate the gate and predict mismatch-response collapse (Furutachi et al. 2024, in silico).
- **Objective knob:** hold *delivery* fixed, swap error *meaning* (predict-your-input vs target−output); plot sample-efficiency vs depth (Millidge et al. 2020; Song et al. "Z-IL" 2020).
- **Alignment axis:** log cosine(feedback, Wᵀ) across training, KP vs DFA; correlate with capability (Akrout et al. 2019).
- **Vectorized-instructive mode:** a per-neuron dendritic teaching signal; flip signs (optogenetic analog) and confirm learning is disrupted (Payeur et al. 2021; *Nature* 2026).
- **Relaxation micro-phase:** a few settling iterations before each e-prop update; measure the sample-efficiency gain (Song et al. 2024).
- **Loop closure (the Held/Hein control):** byte-out → LLM → next-input with surprise-gated plasticity, benchmarked against an identical *replayed* stream.

---

