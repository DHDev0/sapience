# Memory-architecture research: replay buffer vs generative/generalized net (R&D)

## SYNTHESIS: the redesign

## Memory Architecture Redesign: From Stored Episodes to a Generative Net

### 1. Verdict

The user's hypothesis is correct, and all four surveys converge on it. **The raw replay buffer is biologically wrong.** Hippocampal replay was never a verbatim tape: it constructs never-experienced shortcuts and combinatorial trajectories (Gupta et al., *Neuron* 2010), pre-plays untraversed paths (Ólafsdóttir, *eLife* 2015), and sweeps toward *future* goals (Pfeiffer & Foster, *Nature* 2013). Recall is schema-driven *reconstruction*, not reproduction (Bartlett 1932; Schacter 1999). Under Complementary Learning Systems (McClelland, McNaughton & O'Reilly, *Psych. Rev.* 1995; Kumaran, Hassabis & McClelland, *TICS* 2016), the neocortex accumulates *generalized* structure in its weights as a side-effect of slow, interleaved, error-corrected learning; nothing stores raw episodes long-term. The ML instantiation is generative replay (Robins 1995; Shin et al., *NeurIPS* 2017; van de Ven, Siegelmann & Tolias, *Nat. Commun.* 2020): a generator dreams samples, the learner interleaves them. So "generalized-in-the-net + generative self-replay" is the faithful replacement for `EpisodicMemory`'s zstd byte deque (`brain/memory.py`).

### 2. The Missing Mechanism(s)

Sapience already has every ingredient except the wiring. It **can** dream (`SpikingBrain.generate/think/resonate`, `spiking_brain.py:532–597`), it **has** a pattern-separating hippocampal index with novelty gating (`SpikingHippocampus.separate/store/recall`, `spiking_modules.py:154`; `_index_episode`, `life.py:862`), SHY downscale (`life.py:934`, `p.mul_(1-2e-5)`), pruning (`spiking_brain.py:872`) and NREM/REM tone (`spiking_modules.py:287`). What is missing:

- **No seed→dream→learn loop.** Sleep trains on *raw text* (`_sleep_tick`: `chunk = self.memory.sample(1400)`, `life.py:927`). `generate()` is never a training source. This is the core gap.
- **No conditioning path from hippocampus into cortex.** Hippocampal `vals` are 256-bin byte-histogram fingerprints (`_fingerprint`, `life.py:795`) — perfect *gist* seeds, but nothing injects a recalled seed into generation.
- **Wake does no slow reorganization** beyond online e-prop: no multi-timescale synapse, no importance-gating, no awake micro-replay that consolidates.
- **SHY is uniform, not selective** down-selection; no phase-gated depress-familiar.

### 3. Buildable Redesign

**KEEP** (already faithful): the hippocampus as an index/seed generator; `generate/resonate`; SHY; `prune/develop`; the `_novelty` salience signal; NREM/REM cycling.

**REPLACE**: `memory.sample()` as the sleep *training source* → generative self-replay. Keep the byte store only as a cold audit log, never a learning input.

**Generative self-replay (sleep).** Rewrite `_sleep_tick` (`life.py:906`): each replay iteration (a) draws a prioritized seed — sample a stored hippocampal key by **need × gain** = recency × `_novelty`/surprise (Mattar & Daw, *Nat. Neurosci.* 2018), pattern-complete via `hippo.recall`; (b) conditions the cortex on that reconstructed gist (project the 256-vector as a soft logit bias / initial-state prompt into `generate`) and dreams a sequence — reconstructed *and*, in REM, adversarially mixed seeds (Deperrois et al., *eLife* 2022; Hoel, *Patterns* 2021); (c) hard-learns that self-generation with full e-prop. No raw bytes enter training. This is CLS division of labour: hippocampus reinstates, cortex integrates.

**Selective flush + prune.** Replace uniform `mul_(1-2e-5)` with **eligibility-weighted down-selection** (Tononi & Cirelli, *Neuron* 2014): scale each synapse by a function of its recent replay eligibility — protect replayed/strong, drive weak→0 — then `prune()`. Add **phase-gated plasticity** (Poe, *J. Neurosci.* 2017): potentiate novel/surprising seeds, LTD familiar ones, gated on prediction error. After a seed's gist is absorbed (recall similarity high), evict its hippocampal key — trace-transformation (Winocur & Moscovitch 2011), the real "flush + prune."

**Wake slow reorganization.** (i) Give each cortical weight a short Benna–Fusi/cascade chain (Benna & Fusi, *Nat. Neurosci.* 2016) — live e-prop enters the fast variable, a slow leak diffuses value into protected slow variables continuously. (ii) Track a running per-weight importance (Synaptic Intelligence, Zenke et al. 2017) from the e-prop trace and inversely scale learning rate — native metaplasticity, no Fisher recompute. (iii) During low-input quiescence, fire **awake micro-replay**: `resonate` already runs (`life.py:991`); let a fraction hard-learn, priority-ordered.

### 4. Division of Labor

**WAKE** = many small, attention-gated, importance-weighted online integrations + fast-variable overlay + prioritized awake micro-replay (gentle reshaping). **SLEEP** = seed→dream→strong consolidation → selective SHY down-selection → prune → key eviction (aggressive reorganization).

### 5. Step-by-Step Plan

1. Add `hippo.sample_key()` + a fingerprint→cortex conditioning hook in `generate`.
2. Rewrite `_sleep_tick` to the seed→dream→learn loop; demote `memory` to audit-only.
3. Priority scoring on stored keys (recency × novelty).
4. Eligibility-weighted downscale + key eviction in the SHY block.
5. Per-weight importance/cascade state in `_eprop_step`.
6. Phase-gated novel/familiar split under REM tone.

### 6. Measurement

Use the existing evals (`next_byte_acc`, `bits_per_byte`, `spiking_brain.py:601–620`). **Consolidation-without-buffer:** disable `memory` as a training input; show old-topic probe accuracy still rises across nights. **Forgetting-resistance:** train topic A, then B–E, measure A-probe accuracy retention — buffer-replay vs generative-replay vs no-replay; expect generative ≈ buffer, both ≫ none, with power-law (not exponential) decay from the cascade synapse. Track seed recall similarity and replay diversity (`generate_diag` entropy) to confirm reconstructive, non-verbatim replay.

Net: the missing mechanism is a **hippocampus-seeded conditional generator wired into the sleep loop**, plus multi-timescale wake plasticity — letting the raw byte buffer be deleted entirely.

---

## Survey 1

## RESEARCH — What sleep computationally DOES to the network (flush + prune + hard-learn)

**Two established, complementary theories — they are not rivals.**

**(1) Synaptic Homeostasis Hypothesis (SHY)** — Tononi & Cirelli (*Brain Res. Bull.* 2003; *Sleep Med. Rev.* 2006; *Neuron* 2014, "Sleep and the price of plasticity"; *Sleep and synaptic down-selection*, EJN 2019). Claim: wake is net **potentiating** — synapses strengthen proportionally to activity, raising total synaptic weight, energy cost, and saturating capacity. NREM slow-wave activity drives a **global renormalization**: synapses are **downscaled toward baseline**, but *selectively* — it is "down-selection," not uniform decay. Strong/consolidated circuits survive proportionally better; weak, spuriously-potentiated synapses are driven to zero and **pruned**. Net effect = improved SNR, restored capacity, energy savings. Settled: sleep produces net synaptic depression and slow-wave homeostasis (SWA rises with prior wake learning). Open/contested: whether downscaling is truly multiplicative-proportional vs. tag-selective, and how it coexists with potentiation.

**(2) Active Systems Consolidation** — Born & Wilhelm (*Psychol. Res.* 2012); Diekelmann & Born (*Nat. Rev. Neurosci.* 2010); Klinzing, Niethard & Born (*Neuron* 2019). Claim: NREM **redistributes** memories from hippocampus to cortex via **replay**, orchestrated by a nested oscillatory triple: neocortical **slow oscillations (0.5–2 Hz)** whose depolarized up-states open windows in which **thalamocortical spindles (12–15 Hz)** and **hippocampal sharp-wave ripples (150–200 Hz, carrying compressed replay)** are phase-locked (ripple → spindle trough → SO up-state). This coupling is the *gate* that times plasticity; a 2024 Bayesian meta-analysis (eLife) confirms SO–spindle coupling predicts consolidation. Settled: replay + SO-spindle-ripple coupling causally support consolidation (closed-loop TMS/auditory-cueing studies). Open: precise causal weight of each rhythm.

**(3) Sleep as directed forgetting / phase-gated LTD** — Poe (*J. Neurosci.* 2017, "Sleep Is for Forgetting"); Boyce et al. 2016 (REM theta). Timing relative to hippocampal **theta phase** decides sign of plasticity: novel-place cells fire at theta **peaks → LTP**; familiar representations fire at **troughs → LTD/depotentiation**. So REM theta actively **weakens the already-known** while protecting the new — a built-in forgetting mechanism, not passive decay.

**(4) Dreaming as generative rehearsal (not raw replay).** Hoel (*Patterns* 2021, "Overfitted brain hypothesis"): dreams are *corrupted/augmented* samples that fight overfitting and improve generalization — like noise augmentation in DNNs. Deperrois, Petrovici, Senn et al. (*eLife* 2022, "Perturbed and adversarial dreaming"): NREM = perturbed replay of episodic latents (robustness); REM = **adversarially generated, mixed** internal inputs (extract semantics). The brain rehearses *self-generated* content, never a stored data buffer.

### Concrete mechanisms sapience could adopt to REPLACE the raw replay buffer

1. **Generative replay instead of a byte buffer** (van de Ven, Tolias & Siegelmann, *Nat. Commun.* 2020). Sleep should sample from the **cortex's own generative dynamics** (sapience already has think/generate/dream) and re-learn on those, optionally conditioned on hippocampal DG keys. This is exactly the user's "generalized memory in the net" — kills the RAM deque + zstd SSD store entirely.

2. **Selective down-selection, not uniform SHY downscale.** Replace the current *multiplicative* global rescale with **synaptic-tag-weighted** downscaling: scale each synapse by a function of its recent replay/eligibility (protect replayed/strong, drive weak→0), then prune. This is SHY's "down-selection" and directly implements "flush + prune."

3. **Phase-gated potentiate-novel / depress-familiar (Poe).** During REM-tone, split plasticity by novelty: apply LTP to surprising/novel replayed patterns and **LTD to familiar** ones (gate on a familiarity/prediction-error signal), giving active reorganization rather than blanket learning. Pair with **SO-up-state-gated bursts**: only apply consolidation plasticity inside simulated up-state windows so replay is timed, not continuous.

Sources: [SHY 2014 Neuron](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/) · [Tononi & Cirelli 2006](https://pubmed.ncbi.nlm.nih.gov/16376591/) · [Sleep & synaptic down-selection 2019](https://pubmed.ncbi.nlm.nih.gov/30614089/) · [Born & Wilhelm 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3278619/) · [Diekelmann & Born 2010 / Klinzing-Niethard-Born Neuron 2019](https://www.cell.com/neuron/fulltext/S0896-6273(23)00201-5) · [SO–spindle meta-analysis eLife 2024](https://elifesciences.org/articles/101992) · [Poe, Sleep Is for Forgetting 2017](https://www.jneurosci.org/content/37/3/464) · [Remembering to Forget 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6425990/) · [Hoel, Overfitted Brain 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8134936/) · [Deperrois et al., Adversarial Dreaming eLife 2022](https://arxiv.org/abs/2109.04261) · [van de Ven et al., Brain-inspired replay Nat. Commun. 2020](https://www.nature.com/articles/s41467-020-17866-2.pdf)

---

## Survey 2

**Verdict: A raw‑episode buffer is not biologically real. Hippocampal replay is reconstructive/generative pattern completion, and generative self‑replay is the faithful replacement for sapience's byte-chunk buffer.**

**Settled findings.** Sleep/rest replay was discovered as reactivation of waking ensemble sequences (Wilson & McNaughton, *Science* 1994), but the "verbatim tape" reading is now contradicted by its own literature. Gupta et al. ("Hippocampal Replay Is Not a Simple Function of Experience," *Neuron* 2010) showed replay constructs *never‑experienced* trajectories and shortcuts the animal never ran, and does not simply reflect recency — it is combinatorial. Ólafsdóttir et al. (*eLife* 2015) found "preplay" of a visible‑but‑never‑traversed path. Pfeiffer & Foster ("Hippocampal place‑cell sequences depict future paths to remembered goals," *Nature* 2013) showed forward sweeps that generate goal‑directed *future* trajectories, not stored pasts. At the cognitive level this is old news: Bartlett (1932) established recall as schema‑driven *reconstruction*, not reproduction; Schacter's constructive‑memory framework and "seven sins" (*American Psychologist* 1999; Schacter & Addis 2007) argue the same machinery that reconstructs the past simulates the future, and its distortions are the cost of a *generative* system. The systems‑level frame is Complementary Learning Systems (McClelland, McNaughton & O'Reilly, *Psych. Review* 1995): a fast, pattern‑separated hippocampus teaches a slow, distributed neocortex by *reinstating* memories for interleaved learning — the cortex accumulates generalized structure; nothing stores raw episodes long‑term.

The ML lineage is the buffer‑free alternative that mirrors this: Robins ("Catastrophic forgetting, rehearsal and pseudorehearsal," *Connection Science* 1995) replayed self‑generated pseudopatterns instead of stored data; Shin et al. (Deep Generative Replay, *NeurIPS* 2017) cast the hippocampus as a generator and cortex as the learner; van de Ven, Siegelmann & Tolias ("Brain‑inspired replay," *Nature Communications* 2020) made this scale via *internal* replay of latent features (not pixels), replay‑through‑feedback, and conditional/context‑gated generation — matching biology far better than a sample buffer.

**Open / unsettled.** Replay is *not* pure noise or uniform: it is biased toward recent, rewarded, novel, and surprising experiences (Gupta 2010; later prioritized‑replay work), and awake vs. sleep replay may serve different roles. So "generative" does not mean "unconditioned" — the generator must be *seeded* and *prioritized*. Whether cortex needs the hippocampus as an explicit external generator, or can self‑generate, is genuinely open.

**Concrete mechanisms sapience could adopt (replacing the raw buffer):**

1. **Self‑generative sleep replay (pseudo‑rehearsal).** Delete the zstd byte‑chunk deque as a *training source*. During NREM, let the cortex *dream* sequences from its own dynamics (sapience already has think/generate/resonate) and e‑prop‑learn on those self‑samples — Robins/Shin pseudo‑rehearsal. The generalized memory then lives entirely in the net, exactly the user's hypothesis.

2. **Hippocampus‑seeded conditional replay (the CLS teacher).** Use the existing SpikingHippocampus as the *generator*, not a lookup cache: sample sparse DG keys, pattern‑complete to reconstructions, and replay those *reconstructed* codes into cortex (van de Ven "internal replay" of latent activity, not raw bytes; replay‑through‑feedback). This is the biological division of labour — hippocampus reinstates, cortex slowly integrates — and is inherently reconstructive.

3. **Prioritized + future‑directed generation.** Bias which memories get generated by recency × salience (reward/surprise/novelty), per Gupta 2010, and generate goal/future sweeps (Pfeiffer‑Foster), not only reconstructed pasts. Keep the SHY multiplicative downscale (Tononi & Cirelli) as the flush, and pair generative replay with your §10 pruning as the "hard flush + prune + hard‑learn" sleep the user describes.

Net: the missing mechanism is a **conditioned generator wired into the sleep loop**, letting you drop the raw replay buffer entirely.

Sources:
- [Gupta et al. 2010, Hippocampal Replay Is Not a Simple Function of Experience (Neuron)](http://redishlab.neuroscience.umn.edu/papers/2010_Gupta_Replay_Neuron.pdf)
- [Pfeiffer & Foster 2013, Hippocampal place-cell sequences depict future paths to remembered goals (Nature)](https://www.nature.com/articles/nature12112)
- [Schacter 1999, The seven sins of memory (PubMed)](https://pubmed.ncbi.nlm.nih.gov/10199218/) and [Constructive memory: past and future (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3341652/)
- [McClelland, McNaughton & O'Reilly 1995, Why there are Complementary Learning Systems (PDF)](http://wixtedlab.ucsd.edu/publications/Psych%20218/McClellandMcNaughtonOReilly95.pdf)
- [Robins 1995 pseudo-rehearsal (context via IJCAI 2019 review)](https://www.ijcai.org/Proceedings/2019/0463.pdf)
- [van de Ven, Siegelmann & Tolias 2020, Brain-inspired replay (slides)](https://gmvandeven.github.io/files/slides/CLAImeetUp_Oct2020.pdf) and [Replay in Deep Learning: Missing Biological Elements (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9074752/)
- [Human hippocampal ripples coordinate planning sequences / compositional generative replay (Nature Neuroscience 2026)](https://www.nature.com/articles/s41593-026-02291-3)

---

## Survey 3

## CLS + systems consolidation: the mechanism, and what sapience should adopt

**Strongest current understanding.** Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly, *Psychological Review*, 1995) argues the brain *needs* two systems because a single distributed net that learns fast suffers catastrophic interference: new facts overwrite old weights. The resolution is division of labor — a hippocampus that encodes each episode rapidly in a **sparse, pattern-separated code**, and a neocortex that learns **slowly** by **interleaving** many reactivated episodes so each is only a small gradient step. Because cortical codes are dense and overlapping, that interleaving forces the net to represent what is *shared* across episodes (latent statistical structure) while idiosyncratic detail averages out. This is the actual answer to "how does cortex build a generalized model instead of storing episodes": generalization is a *side effect* of slow, error-corrected, interleaved learning over overlapping representations — the gist accumulates in the weights, the episode does not.

The offline mechanism is **replay**: hippocampal sharp-wave ripples during NREM reactivate recent ensembles and broadcast them to cortex, coordinated with cortical slow-waves/spindles (settled empirically — e.g. large-SWR/PFC reactivation, Neuron 2025). Critically, only ~10–30% of ripples carry recent replay, and replay is **selective and prioritized**, not a uniform tape.

Kumaran, Hassabis & McClelland (*Trends Cogn. Sci.*, 2016) updated CLS on exactly the point in the user's hypothesis: the hippocampus is better modeled as a **generative model** whose replay can be *weighted* by goal/reward/surprise, not a verbatim buffer. Tse et al. (*Science*, 2007; PNAS 2011) showed cortex can consolidate in **one trial** when a **schema** already exists — so cortical learning is *not* obligatorily slow; it is slow only when building the scaffold, then fast for consistent additions.

**Settled vs. open.** Settled: two complementary systems; sparse-vs-distributed division; interleaving prevents interference; replay coordinates the transfer; schemas accelerate it. Open/debated: whether the hippocampus ever becomes *unnecessary* (Standard Consolidation) versus remaining required for episodic detail forever (**Multiple-Trace Theory**, Nadel & Moscovitch 1997; **Trace-Transformation Theory**, Winocur & Moscovitch 2011) — the latter reframes consolidation as **transformation/semanticization**: only a gist-like trace migrates to cortex while the detailed episodic trace stays hippocampal. Also open: how novelty gates cortical learning-rate.

**Three concrete mechanisms to replace the raw replay buffer.**

1. **Generative replay instead of stored bytes.** Sapience already dreams/generates. Wire *that* into sleep: seed generation from hippocampal recalls, then interleave-learn on the self-generated sequences — no raw-text deque. This is the ML-validated instantiation of CLS (Shin et al., NeurIPS 2017, "Deep Generative Replay"; van de Ven, Siegelmann & Tolias, *Nature Comms*, 2020, "brain-inspired replay"), and matches Kumaran 2016's generative-hippocampus view. The buffer becomes a *compressed generative index*, not a byte log — which is precisely the user's intuition, made rigorous.

2. **Schema-gated plasticity (novelty/prediction-error driven).** Compute per-input prediction error against the cortex. Low error (fits a schema) → high learning-rate, one-shot cortical assimilation (Tse). High error/novelty → keep it hippocampal and defer until enough similar traces accumulate. This replaces uniform low-plasticity replay with the biological fast/slow split.

3. **Prioritized, weighted replay + transform-then-flush.** Sample reactivations weighted by recency × surprise × reward (Kumaran 2016; Mattar & Daw, *Nat. Neurosci.*, 2018), and during the SHY downscale, *prune the episodic key once its gist is absorbed* (trace-transformation) — a genuine "flush + prune + hard-learn" cycle rather than eviction of raw chunks.

**Sources:**
- [McClelland, McNaughton & O'Reilly 1995, *Psych. Review*](https://www.researchgate.net/publication/15575602_Why_There_are_Complementary_Learning_Systems_in_the_Hippocampus_and_Neocortex_Insights_from_the_Successes_and_Failures_of_Connectionist_Models_of_Learning_and_Memory)
- [Kumaran, Hassabis & McClelland 2016, *Trends Cogn. Sci.* 20:512–534](http://stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/27315762/))
- [Tse et al. 2007, *Science* 316:76–82, "Schemas and Memory Consolidation"](https://www.science.org/doi/abs/10.1126/science.1135935)
- [Nadel & Moscovitch 1997 / Multiple-Trace Theory review](https://cenl.ucsd.edu/psych506A/papers/Multiple%20Trace%20Theory%20of%20Human%20Memory%20-%20Computational,%20Neuroimaging%20....pdf)
- [Winocur & Moscovitch — Trace Transformation / systems consolidation review](https://neuropsychologylab.psych.utoronto.ca/files/Systems%20consolidation,%20transformation%20and%20reorganization%20Multiple%20Trace%20Theory,%20Trace%20Transformation%20Theory%20and%20their%20Competitors.pdf)
- [Large sharp-wave ripples promote hippocampo-cortical reactivation, *Neuron* 2025](https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1)
- [van de Ven, Siegelmann & Tolias 2020, *Nature Communications* — brain-inspired replay](https://arxiv.org/pdf/2301.06030)

---

## Survey 4

## WAKE-side slow reorganization: what a faithful net needs

**Strongest current understanding.** Catastrophic forgetting is a structural property of shared-weight distributed nets: new learning overwrites the single weight set (French 1999, *Trends Cog Sci* 3:128–135). Biology solves this WITHOUT a raw episode store, via two complementary levers that map onto a wake/sleep division of labor.

*Lever 1 — plasticity that is gated, not uniform (the "reorganize-in-the-net" mechanism).* Instead of storing events, the brain distributes each memory across synapses whose plasticity itself changes (metaplasticity). Cascade synapses (Fusi, Drew & Abbott 2005, *Neuron* 45:599–611) and the Benna–Fusi complex synapse (Benna & Fusi 2016, *Nat Neurosci* 19:1697–1706) give each synapse a chain of coupled hidden variables on many timescales: recent events live in fast variables and are progressively pushed into slow, protected variables. This yields near-linear capacity scaling and power-law (not exponential) forgetting — a *generalized* memory accumulating in the weights, exactly the user's hypothesis. The ML analogue is importance-weighted regularization: EWC uses the Fisher information to stiffen weights important to past tasks (Kirkpatrick et al. 2017, *PNAS* 114:3521–3526); Synaptic Intelligence (Zenke, Poole & Ganguli 2017, *ICML*) computes that importance *online during wake*.

*Lever 2 — awake replay and post-encoding rest.* Reorganization is not sleep-only. Awake sharp-wave-ripple replay is actually *more* prevalent in the awake rest box than in quiescence (Karlsson & Frank 2009, *Nat Neurosci* 12:913–918), and hippocampal encoding patterns *persist into post-encoding waking rest* in proportion to later memory (Tambini & Davachi 2013, *PNAS* 110:19591–19596; review 2019, *Trends Cog Sci* 23:876–890). Crucially this replay is **prioritized, not random**: Mattar & Daw's normative theory (2018, *Nat Neurosci* 21:1609–1617) shows replay is ordered by utility = *need × gain* (recency/proximity × surprise/reward), which is what sapience's uniform buffer sampling lacks.

*Fast vs slow weights.* A short-timescale Hebbian overlay (Ba, Hinton et al. 2016, *Using Fast Weights to Attend to the Recent Past*, NeurIPS) captures the just-experienced context in-net and decays — the wake-side "working" trace that later gets distilled into slow weights.

**Settled vs open.** Settled: forgetting stems from shared weights; awake replay exists and correlates with consolidation; multi-timescale synaptic complexity provably extends retention. Open/contested: EWC's single quadratic penalty is empirically challenged (Huszár 2018, *PNAS*); whether awake replay is *causal* for consolidation vs. serving retrieval/planning; and the exact wake↔sleep split (complementary-learning-systems framing, McClelland 1995; Kumaran, Hassabis & McClelland 2016, *Trends Cog Sci*).

**Division of labor.** WAKE = many small, *gated, prioritized* integrative updates + a fast-weight overlay, gently reshaping slow weights online. SLEEP = the aggressive global reorganization + homeostatic downscale sapience already does.

**Three concrete mechanisms to replace the raw replay buffer:**

1. **Per-synapse multi-timescale state (Benna–Fusi/cascade).** Give each cortical weight a short chain of coupled variables (e.g. 3–4 timescales) instead of one scalar. Live e-prop updates enter the fast variable; a slow leak diffuses value into protected slow variables continuously during wake. This *is* "generalized events accumulating in the net" — no bytes stored.

2. **Online importance-gated plasticity (SI/online-EWC).** Track a running per-weight importance from the e-prop trace during wake and scale each weight's learning rate inversely — high-importance synapses resist change. Native metaplasticity, no buffer, no Fisher recompute.

3. **Generative, priority-ordered internal replay.** Sapience can already dream/generate; during low-input waking quiescence, trigger *self-generated* replay and learn on those sequences, ordered by a Mattar–Daw priority score (surprise × recency, tracked per event-code in the hippocampus module) rather than sampling raw chunks. This is the continual-learning "generative replay" line (Shin et al. 2017, NeurIPS; van de Ven, Siegelmann & Tolias 2020, *Nature Communications* — brain-inspired replay), which explicitly removes the stored-data buffer.

Together these let the raw byte deque be deleted: recent memory lives in fast synaptic variables and hippocampal codes, is reorganized online by gated plasticity + prioritized self-replay during wake, and hard-consolidated/pruned during sleep.

---

