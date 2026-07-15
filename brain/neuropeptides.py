"""§16 · SpikingNeuropeptides — the slow neuropeptide neuromodulation layer (companion to SpikingEndocrine).

A SECOND, slower neuromodulatory controller that sits ALONGSIDE the §16 endocrine (never a second cortisol) and
feeds the SAME three-factor gate M(t) + the §5 tone surface + the §2 BG reward, on a timescale an order of
magnitude slower than the cortisol pool (tau_p ≈ 800-1600 ticks vs endocrine tau_C = 200). Peptidergic
volume/slow transmission is the established SECOND neuromodulatory layer, distinct from fast monoamine tone: it
RECONFIGURES the same circuit on minutes-to-hours timescales rather than adding neurons (van den Pol 2012,
"Neuropeptide transmission in brain circuits", Neuron; Bargmann 2012, "Beyond the connectome") — exactly the
controller (not fifth-network) framing the §16 endocrine already uses.

Three bounded python-float pools, each ∈ [0,1], each a leaky integrator p ← clip(p + a·(x − p), 0, 1) whose fixed
point p* = mean(driver x) makes the pool a slow EMA of its driver (bounded + stable by construction):

  OXT  (oxytocin — affiliation / positive-valence buffering; Neumann & Landgraf 2012, Trends Neurosci). Rises
       with valence-positive signals (learning progress, dopamine, and a `social` cue = tool/teacher interaction),
       decays slowly. Anxiolytic: DAMPS the endocrine cortisol POOL (cortisol_relief → endo.C) and lifts valence.
  ORX  (orexin / hypocretin — arousal, wake-stability, novelty/reward-seeking; Sakurai 2007, Nat Rev Neurosci;
       de Lecea). Rises with novelty/reward-seeking, falls with satiation and in sleep. Biases toward EXPLORATION
       (ne_bias adds to §5 NE arousal) and sustains wakefulness (sleep_resist raises the wake threshold).
  CRH  (corticotropin-releasing factor — the stress/vigilance biaser; Bale & Vale 2004, Annu Rev Pharmacol).
       Rises with threat + surprise. UPSTREAM of the HPA axis: instead of storing its own cortisol it MULTIPLIES
       the THREAT driver that flows into endocrine.wake_tick (threat_gain) and biases valence negative + mildly
       suppresses plasticity. This is the explicit anti-duplication design — CRH gates cortisol's DRIVER, OXT
       relieves cortisol's POOL, neither re-implements the single cortisol integrator. One integrator per axis,
       biased at its INPUT (threat_gain) and OUTPUT (cortisol_relief), so the closed loop cannot double-count.

Device / dtype / FSDP: identical to SpikingEndocrine/SpikingDynamics — all state is a handful of python-float
scalars, NO tensors created, no device/dtype hard-coding. __init__ accepts device=/dtype= for API symmetry but
stores nothing on the device. Being a replicated controller of scalars (never per-neuron/per-synapse state) it is
trivially device-, dtype- and FSDP2-agnostic and O(1) at any width (hidden=128000 / 256k neurons costs the same
~1 KB): it only rescales the single scalar gate M(t), the scalar tone['ne'] and the scalar BG reward the loop
already applies uniformly, so it introduces no fan-in/width normalisation of its own and cannot reintroduce a
1/width starvation. Refs: runs/deeper_brain_integrated_design.md §16.
"""


class SpikingNeuropeptides:
    _KEYS = ("on", "tau_p", "k_op", "k_cp", "k_on", "k_ov", "k_ov2", "k_cv",
             "k_ct", "k_oc", "k_os", "social_gain")

    def __init__(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype                     # stored for API symmetry only — this controller holds NO tensors
        self.on = False                        # opt-in toggle (verify via the earns-keep A/B before defaulting on)
        # pools ∈ [0,1] (python floats → dtype/device/FSDP-trivially-safe, O(1) at any width)
        self.OXT = 0.3; self.ORX = 0.4; self.CRH = 0.2
        # slow baselines the sleep phase relaxes toward (constants, not tuned via set_params)
        self._orx_base = 0.4; self._oxt_base = 0.3
        # rates / gains (live-tunable). tau_p ≈ 10x slower than endocrine tau_C=200 → one HPA axis, no oscillation
        self.tau_p = 1200.0
        self.k_op = 0.30     # OXT → plasticity up (prosocial calm broadens learning)
        self.k_cp = 0.30     # CRH → plasticity down (stress narrows)
        self.k_on = 0.30     # ORX → NE arousal / exploration up
        self.k_ov = 0.10     # OXT → NE down (affiliation calms arousal)
        self.k_ov2 = 0.30    # OXT → BG reward up (approach / positive valence)
        self.k_cv = 0.30     # CRH → BG reward down (avoid / negative valence)
        self.k_ct = 0.50     # CRH → threat gain into endocrine (upstream HPA driver)
        self.k_oc = 0.02     # OXT → cortisol POOL relief (prosocial HPA buffer; small vs endocrine's own dynamics)
        self.k_os = 0.20     # ORX → sleep resistance (wakefulness)
        self.social_gain = 1.0  # weight of the tool/teacher `social` cue into the OXT driver

    # ---- drivers (each clipped ∈ [0,1] so the leaky fixed point stays bounded) ---- #
    @staticmethod
    def _clip01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def wake_tick(self, progress=0.0, novelty=0.5, surprise=0.0, threat=0.0, da=0.0, social=0.0):
        """One wake tick: advance the three pools by their leaky integrators. Each driver is clipped to [0,1] so
        the fixed point p* = mean(driver) ∈ [0,1] and the pool is a slow EMA of it (bounded, stable). Returns the
        current levels dict (metric surface)."""
        a = 1.0 / max(self.tau_p, 1e-6)                    # slow rate (guard live-tuned tau_p→0)
        # OXT driver: positive valence = learning progress + phasic dopamine + a social (tool/teacher) cue
        x_oxt = self._clip01(max(0.0, float(progress)) + max(0.0, float(da)) + self.social_gain * float(social))
        # ORX driver: novelty / reward-seeking, damped by satiation (learning progress met the drive)
        x_orx = self._clip01(float(novelty) + max(0.0, float(da)) - max(0.0, float(progress)))
        # CRH driver: threat + prediction-error/surprise (the HPA/vigilance biaser)
        x_crh = self._clip01(max(0.0, float(threat)) + max(0.0, float(surprise)))
        self.OXT = self._clip01(self.OXT + a * (x_oxt - self.OXT))
        self.ORX = self._clip01(self.ORX + a * (x_orx - self.ORX))
        self.CRH = self._clip01(self.CRH + a * (x_crh - self.CRH))
        return dict(OXT=self.OXT, ORX=self.ORX, CRH=self.CRH)

    def sleep_tick(self):
        """Sleep: orexin falls (orexin neurons silence in sleep, relaxing toward baseline), oxytocin recovers,
        CRH clears — mirrors the biological peptide dynamics and interconnects with endocrine.sleep_tick."""
        a = 1.0 / max(self.tau_p, 1e-6)
        self.ORX = self._clip01(self.ORX + 20.0 * a * (self._orx_base - self.ORX) - 0.01)  # decays + relaxes to base
        self.OXT = self._clip01(self.OXT + 20.0 * a * (self._oxt_base - self.OXT) + 0.005)  # gentle recovery
        self.CRH = self._clip01(self.CRH - 0.02)           # stress clears over the night

    # ---- modifiers exposed to the shared surface (the biological locus of peptide action) ---- #
    def plasticity_bias(self):
        """Multiplier on the SAME three-factor gate M(t) that carries ach·endo.plasticity_gain(). OXT broadens,
        CRH narrows. Bounded ∈ [0.5, 1.5] so it preserves the upstream fan-in normalisation (width-invariant)."""
        g = 1.0 + self.k_op * self.OXT - self.k_cp * self.CRH
        return 0.5 if g < 0.5 else (1.5 if g > 1.5 else g)

    def ne_bias(self):
        """Additive exploration/arousal term folded into the per-call tone['ne'] (ORX up, OXT down)."""
        return self.k_on * self.ORX - self.k_ov * self.OXT

    def valence_bias(self):
        """Added to the §2 BG reward (OXT → approach, CRH → avoid) — the primary earns-keep behavioural channel."""
        return self.k_ov2 * self.OXT - self.k_cv * self.CRH

    def threat_gain(self):
        """Multiplies the THREAT that flows INTO endocrine.wake_tick — CRH as the upstream HPA driver. ≥ 1."""
        return 1.0 + self.k_ct * self.CRH

    def cortisol_relief(self):
        """Subtracted from the endocrine's own cortisol POOL after its wake_tick — the prosocial OXT buffer."""
        return self.k_oc * self.OXT

    def sleep_resist(self):
        """Raises the wake threshold / resists sleep pressure (orexin = wakefulness). ≥ 0."""
        return self.k_os * self.ORX

    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on, oxytocin=round(self.OXT, 3), orexin=round(self.ORX, 3), crh=round(self.CRH, 3),
                    plasticity_bias=round(self.plasticity_bias(), 3), ne_bias=round(self.ne_bias(), 3),
                    valence_bias=round(self.valence_bias(), 3), cortisol_relief=round(self.cortisol_relief(), 3))
