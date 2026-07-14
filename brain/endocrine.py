"""§16 · SpikingEndocrine — the slow neuro-endocrine controller (P1).

Three hormone-like scalars that evolve over minutes / a wake-bout — an order of magnitude slower than the
fast §5 tones and the eligibility trace — sitting ABOVE them. It is NOT a fifth network: it is a controller
of the existing four tones + the self-adapting attention + the sleep-debt + the dopamine critic.

  drive D (energy, novelty)  — basic-need deficits, leaky integrators that FILL over a wake-bout and are MET
      by learning-progress / novelty / sleep. Meeting a need emits a homeostatic-RL reward (Keramati-Gutkin
      2014: reward = the DROP in total deficit) and sharpens focus (Aston-Jones-Cohen: satiation → low tonic
      NE → phasic focused mode).
  cortisol C  — the slow stress hormone (HPA axis). Rises with unmet drive + prediction-error/threat, decays
      over a wake-bout, and is RELIEVED by sleep. Inverted-U modulation of plasticity (Lupien 2009; Joëls):
      moderate acute C sharpens learning, chronic-high C impairs it.
  mood M  — a dopamine EMA that sets the 5-HT tone and DAMPS cortisol (resilience; Eldar-Niv).
  allostatic load AL  — chronic-stress cost: accrues while C is high, recovers only in low-C sleep, and caps
      the plasticity ceiling (McEwen allostatic load).

References: runs/drive_stress_research.md, runs/deeper_brain_integrated_design.md §16.
"""
import math


class SpikingEndocrine:
    _KEYS = ("on", "alpha_D", "tau_C", "k_thr", "k_pe", "k_need", "C_star", "C_sigma",
             "lam_mood", "al_thr", "drive_met", "novelty_met")

    def __init__(self, device=None):
        self.device = device
        self.on = False                        # opt-in toggle (verify before defaulting on)
        # state (all scalars, device-agnostic → dtype/device/FSDP-trivially-safe)
        self.D_energy = 0.3; self.D_novelty = 0.3     # deficits ∈ [0,1]
        self.C = 0.2; self.M = 0.5; self.AL = 0.0
        # rates (live-tunable)
        self.alpha_D = 0.003                   # deficit fill / tick (fills over ~hundreds of ticks = a wake-bout)
        self.drive_met = 1.5; self.novelty_met = 1.0  # satiation removes this FRACTION of the deficit (fast reset)
        self.tau_C = 200.0                     # cortisol decay timescale (ticks ≈ a wake-bout)
        self.C_max = 1.5                       # cortisol is a BOUNDED hormone (physiological ceiling)
        self.k_thr, self.k_pe, self.k_need = 0.05, 0.03, 0.02
        self.C_star, self.C_sigma = 0.35, 0.35 # inverted-U optimum + width (moderate stress = best learning)
        self.lam_mood = 0.02
        self.al_thr = 0.6                      # cortisol above which allostatic load accrues

    def wake_tick(self, surprise=0.0, threat=0.0, progress=0.0, novelty=0.5, da=0.0):
        """One wake tick. Fills drives (met by progress/novelty), updates cortisol (threat + prediction-error
        + unmet drive, decaying, damped by mood), mood (dopamine EMA), allostatic load. Returns r_home."""
        a = self.alpha_D
        D0 = self.D_energy + self.D_novelty
        # fill by a; a satiation event (progress / novelty) removes a FRACTION of the current deficit — the
        # cue-triggered fast reset (Chen-Knight 2015), not a slow trickle.
        self.D_energy = min(1.0, max(0.0, self.D_energy + a - self.drive_met * max(0.0, float(progress)) * self.D_energy))
        self.D_novelty = min(1.0, max(0.0, self.D_novelty + a - self.novelty_met * float(novelty) * self.D_novelty))
        D1 = self.D_energy + self.D_novelty
        r_home = max(0.0, D0 - D1)                                       # homeostatic-RL reward = deficit drop
        self.C = min(self.C_max, max(0.0, self.C + self.k_thr * float(threat) + self.k_pe * max(0.0, float(surprise))
                     + self.k_need * D1 - self.C / self.tau_C - 0.02 * max(0.0, self.M - 0.5)))   # BOUNDED hormone
        self.M = (1.0 - self.lam_mood) * self.M + self.lam_mood * (0.5 + float(da))
        if self.C > self.al_thr:
            self.AL = min(1.0, self.AL + 0.001 * (self.C - self.al_thr))
        return r_home

    def sleep_tick(self):
        """Sleep relieves cortisol + allostatic load and meets the drives (rest/consolidation)."""
        self.C = max(0.0, self.C - 0.02)       # sleep actively RELIEVES cortisol (clears the ceiling over a night)
        self.AL = max(0.0, self.AL - 0.003)    # allostatic load recovers only in low-C sleep
        self.D_energy = max(0.0, self.D_energy - 0.02); self.D_novelty = max(0.0, self.D_novelty - 0.02)

    def plasticity_gain(self):
        """Inverted-U cortisol modulation, capped by allostatic load. g(C) ∈ (0,1] (moderate C ⇒ ≈1)."""
        g = math.exp(-((self.C - self.C_star) ** 2) / (2.0 * self.C_sigma ** 2 + 1e-9))
        return g * (1.0 - 0.5 * self.AL)

    def ne_gain(self):
        """Deficit → arousal/exploration (high NE); satiation → focus (low NE). ∈ [0.5, 1.5]."""
        return 0.5 + 0.5 * (self.D_energy + self.D_novelty)

    def sleep_pressure(self):
        """Unmet stress adds to sleep pressure."""
        return 1.0 + 0.8 * self.C

    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            v = bool(v) if isinstance(cur, bool) else (float(v) if cur is not None else v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on, drive_energy=round(self.D_energy, 3), drive_novelty=round(self.D_novelty, 3),
                    cortisol=round(self.C, 3), mood=round(self.M, 3), allostatic=round(self.AL, 3),
                    plasticity_gain=round(self.plasticity_gain(), 3), ne_gain=round(self.ne_gain(), 3))
