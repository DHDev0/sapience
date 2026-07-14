"""§16 · SpikingDynamics — dynamic network states + oscillatory rhythm (P2).

Two facts the always-on loop ignores. (1) NOT ALL REGIONS ARE ACTIVE at once — each subsystem IGNITES only
when its salience beats a threshold set by a single ENTROPY knob β: low β = focused / modular / sparse (the
normal brain), high β = integrated / near-all-on (the psychedelic/REBUS regime, Carhart-Harris 2019). β tracks
arousal (NE tone) and attention, so a satiated, focused brain engages FEW systems and an aroused one engages
many. (2) PROCESSING FREQUENCY CHANGES WITH ATTENTION — focused attention runs a fast gamma window (short
eligibility integration), disengagement runs a slow alpha window (Fries communication-through-coherence 2015;
Jensen-Mazaheri 2010). Both are single scalars over the fixed physics tick, so nothing about the integrator
changes — only which modules run and over what window. Refs: runs/dynamics_oscillations_research.md.
"""
import math


class SpikingDynamics:
    _KEYS = ("on", "beta0", "kappa", "ignite_thr", "f_alpha", "f_gamma")

    def __init__(self, device=None):
        self.device = device
        self.on = False                        # opt-in toggle (verify before defaulting on)
        self.beta0 = 2.0                       # base entropy/gain (the normal↔psychedelic dial: raise → all-on)
        self.kappa = 1.0                       # attention's contribution to β
        self.ignite_thr = 0.5                  # a subsystem runs if its ignition gate > this
        # eligibility-decay eb endpoints (window τ≈1/(1-eb)): ALPHA = disengaged = LONG window = HIGH eb;
        # GAMMA = focused = SHORT window = LOW eb. (Fixed: eb is a retention factor, so short window = low eb.)
        self.f_alpha, self.f_gamma = 0.975, 0.90
        self._beta = self.beta0
        self._active = {}                      # last ignition mask (metric)

    def entropy(self, ne=1.0, attention=1.0):
        """Global entropy/temperature β — the normal↔LSD dial. Rises with arousal (NE) and attention, so a
        satiated/focused brain (low NE) is sparse+selective (low β) and an aroused one integrates (high β)."""
        return self.beta0 * float(ne) * (1.0 + self.kappa * (float(attention) - 1.0))

    def ignition(self, saliences, ne=1.0, attention=1.0):
        """Which subsystems ignite this cycle (global-workspace soft competition, Dehaene). gate_i =
        σ(β·(salience_i − mean)); a system runs iff gate_i > ignite_thr. Low β → few ignite; high β → many.
        The cortex always ignites (it is the learner); this gates the auxiliary systems."""
        b = self.entropy(ne, attention); self._beta = b
        vals = list(saliences.values()) or [0.0]
        mean = sum(vals) / len(vals)
        active = {}
        for k, s in saliences.items():
            g = 1.0 / (1.0 + math.exp(-b * (float(s) - mean)))
            active[k] = bool(g > self.ignite_thr)
        self._active = active
        return active

    def eligibility_beta(self, attention=1.0):
        """Attention → processing frequency: focused (high attention) → gamma (short window, decay → f_gamma),
        disengaged → alpha (long window, f_alpha). Returns the eligibility decay for the e-prop trace."""
        a = max(0.0, min(1.3, float(attention))) / 1.3
        return self.f_alpha + a * (self.f_gamma - self.f_alpha)

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
        return dict(on=self.on, beta=round(self._beta, 3), n_active=sum(self._active.values()) if self._active else 0,
                    active=self._active, eff_freq=round(self.eligibility_beta(), 3))
