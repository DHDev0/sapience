"""§PC · PredictiveCoding — hierarchical local predictive coding (Rao-Ballard / Friston free-energy).

A THIRD cortical learning rule (learn_rule='pc'), a peer of e-prop and surrogate-BPTT. It shares the cortex's
forward physics, head, top-down feedback matrices self._fb (= the PC generative/prediction weights) and the
width-invariant update tail; it DIFFERS in exactly three PC-defining ways, all fully LOCAL:

  (1) NO eligibility trace — the instantaneous β→0 limit §3.5 already names. The Hebbian weight rule is
      Δw_ji ∝ (Π_l·e_l·ψ)_j · pre_i(t) with pre = the instantaneous presynaptic activity (z_prev / layer_in),
      not the low-passed eligibility ε. _pc_step realises this by forcing the trace decay eb=0 so eps_rec/eps_in
      collapse to z_prev / layer_in exactly (see the wiring patch for spiking_brain._pc_step).
  (2) PRECISION weighting (Friston's free-energy hallmark) — each error unit is scaled by an inverse-variance
      running estimate Π_l,j = 1/(σ²_l,j + eps), mean-normalized to 1 so it adds NO width-dependent scale.
  (3) optional INFERENCE relaxation — the precision-weighted top-down error is fed back onto the membrane drive
      (infer_gain·Lsig), the SAME slot the two_compartment apical g_ap uses, so predictions influence activity.

Governing objective (Gaussian generative negative-log-evidence):
      F = Σ_l ½ e_l^T Π_l e_l − ½ log|Π_l|
Gradient descent gives three fully-local updates:
      weights     ΔW_l = −∂F/∂W = Π_l e_l · pre^T          (Hebbian error×activity, no weight transport)
      precision   Δlog Π ∝ (1/σ² − e²)                     (variance tracking → sig2 EMA below)
      inference   Δa_l  = −∂F/∂a_l                          (top-down error relaxes the representation → drive)
Whittington-Bogacz 2017 / Millidge 2022: supervised PC with the feedforward pass as amortized inference
approximates backprop when Π is uniform → PC reaches a BPTT-comparable objective while staying local, which is
why it belongs on the fidelity↔capability curve. Refs: Rao-Ballard 1999; Friston 2005; Bastos 2012.

This class holds ONLY the PC EXTRAS state (per-layer per-neuron error statistics + hyperparams). The rule itself
lives in spiking_brain._pc_step (a core method), which reuses e-prop's spmm/sddmm/edge_reduce and _upd.

DEVICE / DTYPE / FSDP: every per-neuron tensor (sig2/prec) is created in ensure() FROM each cortex cell's width
on self.device — identically to the cortex's own _fb / _thr_adapt aux state, so it shards along the SAME neuron
partition under FSDP2 and never device-materialises a full copy of a sharded param. The statistics are fp32 (the
deliberate, documented choice: e-prop's grads also .float(); variance tracking needs the precision) while the
model may run bf16/fp16 — the returned precision weight is cast to the working error dtype so matmuls autocast.
Scalar hyperparams (prec_tau, infer_gain, pc_lr_scale, eps) are replicated python floats — small controller state.

SCALE (hidden=128000, 2 layers, CSR fan-in 32): new state = 2 per-neuron fp32 vectors (sig2/prec) × 2 layers ≈
2×2×128000×4B ≈ 2.0 MB. NO O(N²) and NO O(nnz) per-synapse state — the weight gradient reuses the existing sparse
CSR sddmm/edge_reduce. Precision is mean-normalized to 1 (width-invariant) and the shared _upd divides by N_j^p,
so no fan-in/width starvation is reintroduced. DEFAULT OFF (self.on gates the PC EXTRAS; learn_rule='pc' selects
the rule) — present ≠ useful; the A/B decides.
"""
import torch


class PredictiveCoding:
    _KEYS = ("on", "precision", "prec_tau", "infer_gain", "pc_lr_scale", "phi", "eps")

    def __init__(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype                     # model dtype (matmuls follow it); statistics stay fp32 (see below)
        self.on = False                        # DEFAULT OFF — gates the PC EXTRAS (precision + inference relaxation)
        self.precision = True                  # inverse-variance weighting of error units (Friston); sub-toggle
        self.prec_tau = 200.0                  # variance-EMA timescale (ticks) → EMA rate 1/prec_tau
        self.infer_gain = 0.0                  # top-down error → membrane drive (inference relaxation); 0 = learn-only
        self.pc_lr_scale = 2000.0              # PC base rate (peer of eprop_lr_scale)
        self.phi = "id"                        # prediction nonlinearity (reserved; 'id' = linear generative model)
        self.eps = 1e-3                        # precision floor: Π = 1/(σ² + eps) (guards variance→0 blow-up)
        # per-layer per-neuron running statistics — one small O(hid) fp32 vector per layer, EMPTY until ensure()
        # builds them FROM the cortex layer widths (so they ride/shard with the cortex like _fb/_thr_adapt).
        self._sig2 = []                        # error variance EMA  σ²_l,j
        self._prec = []                        # last precision       Π_l,j = 1/(σ²+eps), mean-normalized
        self._err = []                         # last mean|e_l| scalar per hidden layer (metric)
        self._pred_err_out = 0.0               # last mean|err| at the output (free-energy proxy, metric)

    # ---- lazy per-layer state, grows/pads with the cortex (like _fb) ---- #
    @staticmethod
    def _pad(vec, h, fill):
        """Resize a per-neuron vector to width h. New entries get `fill` (None → the vector's own mean, so a
        grown neuron starts at precision ≈ mean-normalized 1 rather than a spurious 1/eps blow-up)."""
        old = int(vec.shape[0])
        if h <= old:
            return vec[:h].contiguous()
        if fill is None:
            f = float(vec.mean()) if old > 0 else 0.0
        else:
            f = float(fill)
        new = torch.full((h,), f, device=vec.device, dtype=vec.dtype)
        if old:
            new[:old] = vec
        return new

    def ensure(self, cells):
        """Lazily create / pad per-layer sig2/prec to each cortex cell's CURRENT width — called every step so a
        mid-life grow() (§10) that widens a layer keeps PC consistent and identity-preserving."""
        dev = self.device
        for l, c in enumerate(cells):
            h = int(c.hid)
            if l >= len(self._sig2):
                self._sig2.append(torch.zeros(h, device=dev, dtype=torch.float32))
                self._prec.append(torch.ones(h, device=dev, dtype=torch.float32))
            elif int(self._sig2[l].shape[0]) != h:
                self._sig2[l] = self._pad(self._sig2[l], h, fill=None)      # new neurons ← layer-mean variance
                self._prec[l] = self._pad(self._prec[l], h, fill=1.0)
        while len(self._err) < len(cells):
            self._err.append(0.0)

    # ---- the PC EXTRAS the module owns ---- #
    def precision_weight(self, l, e):
        """Update the per-neuron error-variance EMA from this batch and return the mean-normalized inverse-
        variance Π_l as a (hid,) weight to multiply the error. Returns a scalar 1 (no-op) when the PC extras are
        off — so plain local-delta PC costs nothing. e: (B, hid) prediction error err@B_l (any dtype/device).

        precision update (∂F/∂Π, variance tracking): σ²_l,j ← (1−a)σ²_l,j + a·mean_b e_l,j²,  a = 1/prec_tau.
        Π_l,j = 1/(σ²_l,j + eps), then Π ← Π / mean(Π): mean-normalized to 1 → width-invariant (adds no scale),
        bounded effective lr, and E[Π] ≈ 1 so a uniform-error layer reproduces plain PC exactly."""
        var = e.detach().float().pow(2).mean(0)            # (hid,) batch variance estimate (fp32)
        if l < len(self._sig2):
            a = 1.0 / max(float(self.prec_tau), 1.0)
            self._sig2[l].mul_(1.0 - a).add_(var, alpha=a)  # in-place EMA (fp32, sharded with the layer)
            s2 = self._sig2[l]
        else:
            s2 = var                                        # not yet ensure()d — transient, no persistence
        if not (self.on and self.precision):
            return e.new_ones(())                           # scalar 1 → uniform Π (plain delta-rule PC)
        prec = 1.0 / (s2 + float(self.eps))
        prec = prec / (prec.mean() + 1e-12)                 # mean-normalized → width-invariant, E[Π]≈1
        if l < len(self._prec):
            self._prec[l] = prec
        return prec.to(e.dtype)                             # (hid,) weight in the working dtype (autocast-safe)

    def record(self, l, e):
        """Record mean|e_l| for the REQUIRED per-layer prediction-error metric (pred_err_L{l})."""
        while len(self._err) <= l:
            self._err.append(0.0)
        self._err[l] = float(e.detach().abs().mean())

    def record_out(self, err):
        """Record mean|err| at the output — the top-level prediction error / free-energy proxy."""
        self._pred_err_out = float(err.detach().abs().mean())

    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                       # string 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif k == "phi":                                # 'phi' is a string tag, not a float
                v = str(v)
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        prec_mean = (sum(float(p.mean()) for p in self._prec) / len(self._prec)) if self._prec else 1.0
        return dict(on=self.on, precision=self.precision, infer_gain=round(float(self.infer_gain), 4),
                    prec_tau=round(float(self.prec_tau), 3), pc_lr_scale=round(float(self.pc_lr_scale), 3),
                    pred_err=[round(float(e), 5) for e in self._err],
                    pred_err_out=round(float(self._pred_err_out), 5),
                    mean_precision=round(float(prec_mean), 4))
