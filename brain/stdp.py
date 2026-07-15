"""§15.18 · SpikingSTDP — a pair-based, asymmetric spike-timing-dependent plasticity kernel.

The e-prop eligibility ε_i = β·ε_i + z_i is a per-PRE *rate* low-pass: it carries the recent firing
of the presynaptic neuron but NO post-synaptic timing. STDP supplies exactly what ε cannot represent —
the temporal COINCIDENCE of a pre spike and a post spike, and its SIGN (who fired first). This module is
the timing-refined sibling of ε: it adds a local, additive plasticity term that is blended (coefficient
`mix`) into the same g_rec / g_in gradient accumulators BEFORE the fan-in-normalised, dmax-clamped,
dopamine-gated `_upd` — so STDP inherits the cortex's width-compensation, its ±Δmax stability clamp and
its three-factor neuromodulator gate for free. mix=0 recovers pure e-prop exactly (the A/B baseline).

Governing online (all-to-all trace) rule, one physics tick t  — Pfister & Gerstner 2006, Morrison-
Diesmann-Gerstner 2008 (Biol. Cybern.):
    pre  trace   x_i(t) = λ₊·x_i(t-1) + z_i(t)          λ₊ = exp(-1/τ₊)
    post trace   y_j(t) = λ₋·y_j(t-1) + z_j(t)          λ₋ = exp(-1/τ₋)
    Δw_ji += A₊·z_j(t)·x_i(t-1)     (post fires now, pre fired recently → LTP, pre-before-post)
    Δw_ji -= A₋·z_i(t)·y_j(t-1)     (pre  fires now, post fired recently → LTD, post-before-pre)
Integrating one isolated pre/post pair with lag s recovers the canonical Bi-Poo 1998 window
    W(s) = +A₊·exp(-s/τ₊)  (s>0, pre→post, potentiation),  −A₋·exp(+s/τ₋)  (s<0, post→pre, depression).
The slight LTD bias A₋ > A₊ (Song-Miller-Abbott 2000) is the additive-model stability guarantor.

Blending BEFORE `_upd` is load-bearing: `_upd` divides the delta by the postsynaptic fan-in^p and clamps
to ±Δmax, so a wide post-neuron's larger summed STDP drive is compensated by its own fan-in count exactly
as the e-prop grad is — width-invariant, no fan-in/width starvation reintroduced. A raw w.add_() would
bypass the norm and blow up at hidden=128000. Gated by dopamine (the shared `scale` carries (0.5+da))
this becomes reward-modulated / three-factor R-STDP (Izhikevich 2007; Frémaux-Gerstner 2016) — which is
why it belongs on the fidelity↔capability curve with the rest of the faithfulness stack.

Device/dtype/FSDP: this class holds ONLY replicated python-float scalars + the device/dtype refs (exactly
like SpikingEndocrine / SpikingDynamics) — trivially device/dtype/FSDP-safe. It holds NO tensors as state.
All STDP tensors live transiently inside SpikingBrain._eprop_step, created from the cortex cells (traces
via torch.zeros(B, c.hid, device=dev) like eps_rec/eps_in; per-edge deltas via torch.zeros_like(g_rec[l]))
so they take the model dtype and the SAME sharding partition as the cortex tensors they ride on — the
edge deltas are the same shape/partition FSDP2 already reduces for the gradient accumulators. The kernel
here (`edge_delta`) is pure: it allocates only on the device of the tensors passed in and chunks over the
batch (EP_CHUNK cap), so it is O(nnz·B) with NO O(H²) anywhere and scales to hidden=128000 (nnz≈4.1M/layer,
~16 MB per edge-delta buffer, ~8 MB per B=16 trace — well under a GB for 2 layers).

Refs: Bi-Poo 1998; Song-Miller-Abbott 2000; Pfister-Gerstner 2006; Morrison-Diesmann-Gerstner 2008;
Izhikevich 2007 (R-STDP); Frémaux-Gerstner 2016 (three-factor).
"""
import math
import torch


class SpikingSTDP:
    _KEYS = ("on", "a_plus", "a_minus", "tau_plus", "tau_minus", "mix", "w_ceiling")
    EP_CHUNK = 1 << 26                              # cap on any transient (nnz, chunk) buffer — mirrors _EP_CHUNK

    def __init__(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype
        self.on = False                             # opt-in toggle (verify before defaulting on) — DEFAULT OFF
        # amplitudes: A₋ > A₊ (slight LTD bias) is the additive-model stability guarantor (Song-Abbott 2000).
        self.a_plus = 0.01
        self.a_minus = 0.0105
        # asymmetric window time-constants (in physics ticks): τ₋ > τ₊ (a wider depression window).
        self.tau_plus = 4.0
        self.tau_minus = 6.0
        self.mix = 0.5                              # blend into g_rec/g_in (0 = pure e-prop, the A/B baseline)
        self.w_ceiling = 1.0                        # optional soft LTP bound (Gütig 2003); enforced downstream too
        # last-step scalars set by the brain for state() (pure metrics — no tensors held here)
        self._ltp = 0.0
        self._ltd = 0.0
        self._mag = 0.0

    # ---- decay factors over the physics tick ------------------------- #
    def decay(self):
        """(λ₊, λ₋) = (exp(-1/τ₊), exp(-1/τ₋)) — the per-tick trace retention factors."""
        lam_p = math.exp(-1.0 / max(self.tau_plus, 1e-6))
        lam_m = math.exp(-1.0 / max(self.tau_minus, 1e-6))
        return lam_p, lam_m

    # ---- the pure edge-wise STDP kernel (device/dtype-agnostic, O(nnz·B)) ---- #
    def _pair(self, post_fac, pre_fac, row, col, cap):
        """Σ_b post_fac[b, row] · pre_fac[b, col]  → (nnz,).  Chunked over the batch so the (B, nnz)
        transient never exceeds `cap`; allocates only on post_fac.device — no O(H²)."""
        rw = row.long(); cl = col.long(); nnz = cl.numel()
        B = post_fac.shape[0]
        ch = max(1, min(B, cap // max(1, nnz)))
        out = torch.zeros(nnz, device=post_fac.device, dtype=torch.float32)
        for i in range(0, B, ch):
            out += (post_fac[i:i + ch].float()[:, rw] * pre_fac[i:i + ch].float()[:, cl]).sum(0)
        return out

    def edge_delta(self, z_post, z_pre, x_pre, y_post, row, col, w=None, cap=None):
        """STDP weight delta for ONE sparse matrix over its edge set. Edge e: pre = col[e] → post = row[e]
        (matching the cortex convention w[row, col]). All spike/trace tensors are (B, N_pop); the traces
        x_pre / y_post are the values from t-1 (advance them AFTER this call). Returns
            (delta_nnz, ltp_sum, ltd_sum)
          delta_nnz = A₊·(post-now × pre-trace) − A₋·(pre-now × post-trace)     [pre→post LTP, post→pre LTD]
        `w` (optional current per-edge weight) applies a soft LTP ceiling relu(1 − |w|/w_ceiling) (Gütig
        2003); default None → rely on the downstream ±Δmax clamp + bounded_synapses. NO O(H²)."""
        cap = self.EP_CHUNK if cap is None else cap
        ltp = self._pair(z_post, x_pre, row, col, cap)          # post fires now × pre fired recently  → LTP
        ltd = self._pair(y_post, z_pre, row, col, cap)          # post fired recently × pre fires now  → LTD
        pot = self.a_plus * ltp
        if w is not None:                                       # soft upper bound: LTP shrinks as w → ceiling
            wc = max(self.w_ceiling, 1e-6)
            pot = pot * (1.0 - (w.float().abs() / wc)).clamp_(min=0.0)
        dep = self.a_minus * ltd
        delta = pot - dep
        return delta, float(pot.clamp(min=0.0).sum()), float(dep.clamp(min=0.0).sum())

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
        eps = 1e-9
        tot = self._ltp + self._ltd + eps
        return dict(on=self.on, a_plus=round(self.a_plus, 5), a_minus=round(self.a_minus, 5),
                    tau_plus=round(self.tau_plus, 3), tau_minus=round(self.tau_minus, 3), mix=round(self.mix, 4),
                    ltp_ltd_balance=round(self._ltp / tot, 4),               # 0.5 = balanced; →1 = runaway LTP
                    stdp_net=round((self._ltp - self._ltd) / tot, 4),         # ∈[-1,1]
                    stdp_mag=round(self._mag, 8))                            # mean |blended STDP delta| / synapse
