"""§16 · SharpWaveRipple — NREM sharp-wave-ripple-gated consolidation.

A small scalar controller that models hippocampal sharp-wave-ripples (SWRs) as a point process nested in the
up-state of the <1 Hz slow oscillation, and GATES which sleep-replay reactivations are allowed to WRITE to
cortex during life._sleep_tick. Today the sleep loop commits every generative dream and every buffer chunk
UNCONDITIONALLY; with this gate on, commits are restricted to up-state-coincident, refractory-respecting,
NREM-only events whose density tracks endocrine sleep-pressure and sleep debt — the biological selectivity the
always-commit loop lacks.

Theory / governing equations (per replay ATTEMPT, not per physics second — see state()):
  phase:      phi <- (phi + 2*pi*f_so*dt) mod 2*pi          (slow-oscillation clock)
  up-state:   U   = 1[cos(phi) > up_thr]                    (ripples nest in SO up-states; Staresina 2015)
  gain:       g   = 1 + press_gain*(pressure-1) + debt_gain*tanh(debt/debt_scale)   (density ~ learning load;
                                                                                     Grosmark-Buzsaki 2016)
  emission:   p   = clip(p0 * g * U * (1 - refractory_active), 0, 1)  ; REM forces p=0 (rem_suppress)
  ripple:     fired ~ Bernoulli(p)   drawn from a REPLICATED random.Random(seed) so every FSDP rank agrees.
  commit iff a ripple is emitted for that reactivation; a fired event blocks the next `refractory` attempts.

Refs: Girardeau 2009; Ego-Stengel & Wilson 2010; Wilson & McNaughton 1994; Buzsaki 2015; Diekelmann & Born 2010;
      runs/deeper_brain_integrated_design.md §16; the §8-9 sleep/replay design in life._sleep_tick.

Device / dtype / FSDP: pure python-float scalars + one int refractory counter (identical to endocrine.py /
dynamics.py) — nothing is materialized as a tensor, so it is trivially device/dtype-agnostic and O(1) at ANY
width (256k neurons included). It is a small REPLICATED controller: it holds NO per-neuron/per-synapse state and
never copies a sharded param. The one FSDP correctness requirement — that the commit/skip boolean be IDENTICAL on
every rank (else ranks diverge on whether to run the sleep-replay optimizer step) — is met by drawing the
Bernoulli from a replicated random.Random(seed) advanced by a replicated attempt counter: every rank computes the
same boolean with zero communication. __init__ accepts device+dtype only to mirror the §16 pattern.
"""
import math
import random


class SharpWaveRipple:
    _KEYS = ("on", "f_so", "dt", "p0", "up_thr", "refractory",
             "press_gain", "debt_gain", "debt_scale", "rem_suppress", "seed")

    def __init__(self, device=None, dtype=None, seed=0):
        self.device = device
        self.dtype = dtype
        self.on = False                        # opt-in toggle (verify before defaulting on)
        # --- tunable params (live-settable) ---
        self.f_so = 0.8                        # slow-oscillation frequency (Hz; ~0.75-1 Hz SO)
        self.dt = 0.25                         # phase advance per replay ATTEMPT (2*pi*f_so*dt rad/attempt)
        self.p0 = 0.5                          # base ripple emission prob at the up-state peak
        self.up_thr = 0.0                      # cos(phi) above this = SO up-state (0.0 = the up half-cycle)
        self.refractory = 2.0                  # attempts an emission suppresses the next (ripple refractoriness)
        self.press_gain = 0.5                  # coupling of ripple density to endocrine sleep-pressure
        self.debt_gain = 0.3                   # coupling of ripple density to sleep debt
        self.debt_scale = 50.0                 # debt normaliser inside tanh (saturating)
        self.rem_suppress = True               # SWRs are a NREM phenomenon: REM ticks force p=0
        self.seed = int(seed)                  # replicated-RNG seed (FSDP determinism)
        # --- transient per-night state (scalars; not persisted) ---
        self.phi = 0.0                         # slow-oscillation phase
        self._refr = 0                         # refractory counter (attempts left suppressed)
        self._rate_ema = 0.0                   # EMA of ripple emissions per attempt (NREM SWR density)
        self.rate_lam = 0.05                   # EMA rate for ripple_rate (internal, not tuned)
        self._n_attempt = 0                    # attempted reactivations this night
        self._n_commit = 0                     # committed (ripple-coincident) reactivations this night
        self._n_ripple = 0                     # ripples emitted this night (== commits; secondary/debug)
        self._rng = random.Random(self.seed)   # replicated Bernoulli source (identical on every rank)

    # -------- the single job: gate one candidate reactivation -------- #
    def event(self, phase="nrem", pressure=1.0, debt=0.0):
        """Advance the SO phase and decide COMMIT (ripple-coincident) vs REHEARSE-only for ONE reactivation.
        Returns a python bool that is identical on every FSDP rank for a given (seed, attempt index)."""
        self._n_attempt += 1
        self.phi = (self.phi + 2.0 * math.pi * self.f_so * self.dt) % (2.0 * math.pi)
        rem = (str(phase).strip().lower() == "rem")
        up = 1.0 if math.cos(self.phi) > self.up_thr else 0.0
        refr_active = 1.0 if self._refr > 0 else 0.0
        if self._refr > 0:
            self._refr -= 1                                    # spend one attempt of refractoriness
        if rem and self.rem_suppress:
            p = 0.0                                            # hippocampal SWRs are essentially NREM-only
        else:
            gain = 1.0 + self.press_gain * (float(pressure) - 1.0) \
                + self.debt_gain * math.tanh(float(debt) / max(self.debt_scale, 1e-9))
            p = self.p0 * max(0.0, gain) * up * (1.0 - refr_active)
            p = min(1.0, max(0.0, p))
        fired = (self._rng.random() < p)
        if fired:
            self._refr = int(round(max(0.0, self.refractory)))  # a ripple silences the next `refractory` attempts
            self._n_ripple += 1
            self._n_commit += 1
        self._rate_ema = (1.0 - self.rate_lam) * self._rate_ema + self.rate_lam * (1.0 if fired else 0.0)
        return bool(fired)

    def reset_night(self):
        """Fresh ripple-rate + gated-commit counters (and SO phase / refractory) at the start of each night.
        The replicated RNG is NOT reseeded, so successive nights vary while staying rank-deterministic."""
        self.phi = 0.0; self._refr = 0; self._rate_ema = 0.0
        self._n_attempt = 0; self._n_commit = 0; self._n_ripple = 0

    def set_params(self, **kw):
        applied = {}
        reseed = False
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif k == "seed":
                v = int(float(v)); reseed = True
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        if reseed:
            self._rng = random.Random(self.seed)               # re-derive the replicated Bernoulli source
        return applied

    def state(self):
        # ripple_rate is per-ATTEMPT (the SO phase advances per reactivation attempt, not per second); it is
        # monotone in the true SWR density so the A/B comparison stays valid. gated_commit_fraction == 1.0 with
        # no attempts (matches the commit-everything baseline the off-path uses).
        gcf = (self._n_commit / self._n_attempt) if self._n_attempt > 0 else 1.0
        return dict(on=self.on, ripple_rate=round(self._rate_ema, 4),
                    gated_commit_fraction=round(gcf, 4), n_ripples=self._n_ripple,
                    phi=round(self.phi, 3), refractory_left=int(self._refr))
