"""§17 · SpikingSTP — Tsodyks-Markram short-term synaptic plasticity (facilitation/depression).

A fast, activity-dependent PREsynaptic gain on transmission. Each source neuron carries two fast state
variables per (batch, neuron): utilization u (facilitation / release probability) and available resources x
(vesicle pool / depression). Every physics tick resources recover toward 1 and utilization relaxes toward the
baseline release probability U; when the neuron spikes, facilitation jumps, the released efficacy
g = (u·x)/U is emitted, then resources deplete. g is normalized so at REST (u=U, x=1, no recent spike) g≡1 —
the DEFAULT-OFF path is byte-identical and no weight rescale is needed. A depressing regime (short tau_facil,
long tau_rec) drives g<1 under repeated spikes; a facilitating regime (small U, long tau_facil) drives g>1.

STP is presynaptic (axonal vesicle depletion, Zucker & Regehr 2002), so ONE (u,x) per source neuron governs
all its outgoing transmission. By default it gates the RECURRENT synapses — where presynaptic timing is
unambiguous (z_prev) and where short-term temporal / working memory lives (Mongillo, Barak & Tsodyks 2008,
"Synaptic theory of working memory"). An optional modulate_input bank extends it to the feedforward path.

It interconnects with the e-prop learner: the transmitted effective spike z·g (not the raw spike) is what
accumulates in the recurrent eligibility trace, so the per-synapse gradient is computed on the signal the
synapse ACTUALLY transmitted (a resource-aware eligibility). It composes with the two_compartment PV divisive
gain as an orthogonal PRE- vs POST-synaptic gain (STP depletes the spike before the matmul; PV normalizes the
summed drive after).

Follows the §16 SpikingEndocrine / SpikingDynamics module pattern: a small class with a _KEYS tuple,
set_params (with the string-'off' bool coercion), state() (metrics dict), device/dtype-agnostic scalars,
DEFAULT OFF. Refs: Tsodyks & Markram 1997 (PNAS); Markram-Wang-Tsodyks 1998; Mongillo 2008 (Science);
Bellec et al. 2020 (e-prop). See brain/endocrine.py + brain/dynamics.py for the pattern.
"""
import torch


class SpikingSTP:
    _KEYS = ("on", "tau_rec", "tau_facil", "U", "modulate_input")

    def __init__(self, device=None):
        self.device = device
        self.on = False                        # opt-in toggle (DEFAULT OFF → forward path byte-identical)
        # rates (live-tunable). tau_facil > tau_rec ⇒ net FACILITATING; tau_facil < tau_rec ⇒ net DEPRESSING.
        self.tau_rec = 20.0                    # resource-recovery ticks → the depression timescale
        self.tau_facil = 4.0                   # utilization-relax ticks → the facilitation timescale
        self.U = 0.5                           # baseline release probability (u/x rest point)
        self.modulate_input = False            # also gate the feedforward spike path (default OFF, half-tick lag)
        # transient per-layer state (lists of (B,hid) tensors, lazily (re)built by reset/_ensure)
        self._u = []                           # utilization (facilitation level)
        self._x = []                           # available resources (depletion level)
        self._eff_ema = 1.0                    # running mean transmitted efficacy over spikes (metric)
        # §16 interconnection hook: NE/ACh arousal scales the EFFECTIVE release probability U (ACh/NE
        # neuromodulation of STP, Tsodyks/Hasselmo). 1.0 == neutral (no modulation) → byte-identical to
        # the un-hooked path; life._learn_text may set it from endocrine.ne_gain()/diff_neuromod ne.
        self._ne_gain = 1.0

    # ---- transient state (re)allocation ------------------------------ #
    def reset(self, cells, B, device=None):
        """(Re)allocate per-layer (u,x) at rest (u=U, x=1) — called when a fresh forward begins or B/hid
        changes. Tensors live on each layer's own device (device/dtype-agnostic)."""
        self._u = []; self._x = []
        for c in cells:
            d = device or self.device
            try:
                d = next(c.parameters()).device
            except Exception:
                pass
            self._u.append(torch.full((B, c.hid), float(self.U), device=d))
            self._x.append(torch.ones(B, c.hid, device=d))

    def _ensure(self, l, z_pre):
        """Lazily (re)build layer l's (u,x) at rest if missing or shape/device mismatched (so transmit() is
        safe even if reset() was not called first)."""
        while len(self._u) <= l:
            self._u.append(None); self._x.append(None)
        u = self._u[l]
        if (u is None or u.shape != z_pre.shape or u.device != z_pre.device
                or u.dtype != z_pre.dtype):     # match dtype too → never a hard-coded float32
            self._u[l] = torch.full_like(z_pre, float(self.U))
            self._x[l] = torch.ones_like(z_pre)

    # ---- one physics tick of the Tsodyks-Markram update -------------- #
    def transmit(self, l, z_pre):
        """Advance layer l's (u,x) one tick and RETURN the presynaptic efficacy gain g=(u·x)/U (shape like
        z_pre). At rest g≡1. Gradient-free (a no_grad gating gain, safe for the BPTT/surrogate path)."""
        self._ensure(l, z_pre)
        with torch.no_grad():
            U = max(float(self.U) * float(getattr(self, "_ne_gain", 1.0)), 1e-6)
            u = self._u[l]; x = self._x[l]
            # passive recovery / relaxation toward rest
            x = x + (1.0 - x) / max(self.tau_rec, 1e-6)
            u = u + (U - u) / max(self.tau_facil, 1e-6)
            # RELEASE using the current relaxed state (BEFORE this spike's facilitation jump), so at
            # rest (u=U, x=1) g = (U·1)/U ≡ 1 exactly — the byte-identity guarantee that lets default-off
            # need no weight rescale. Facilitation from this spike is carried forward to the NEXT release.
            g = (u * x) / U                    # released efficacy (==1 at rest u=U,x=1)
            # spike-driven facilitation jump (u rises toward 1 where the neuron fired) for future releases
            u = u + U * (1.0 - u) * z_pre
            # resource depletion AFTER release (only where it spiked)
            x = x - u * x * z_pre
            self._u[l] = u; self._x[l] = x
            # mean transmitted efficacy over ACTIVE spikes → EMA metric
            zsum = float(z_pre.sum())
            if zsum > 0.0:
                eff = float((g * z_pre).sum() / zsum)
                self._eff_ema = 0.99 * self._eff_ema + 0.01 * eff
            return g

    # ---- metrics ----------------------------------------------------- #
    @property
    def mean_efficacy(self):
        return self._eff_ema

    @property
    def mean_u(self):
        try:
            return float(torch.cat([t.reshape(-1) for t in self._u]).mean())
        except Exception:
            return float(self.U)

    @property
    def mean_x(self):
        try:
            return float(torch.cat([t.reshape(-1) for t in self._x]).mean())
        except Exception:
            return 1.0

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
        return dict(on=self.on, tau_rec=self.tau_rec, tau_facil=self.tau_facil, U=self.U,
                    modulate_input=self.modulate_input, mean_efficacy=round(self._eff_ema, 4),
                    mean_u=round(self.mean_u, 4), mean_x=round(self.mean_x, 4))
