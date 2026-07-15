"""§15/§16 · SpikingInterneurons — PV / SOM / VIP spiking sub-populations for the two_compartment circuit.

A non-learned CONTROLLER (same discipline as endocrine.py / dynamics.py: _KEYS, set_params with the string-bool
'on' coercion, state(), DEFAULT OFF) that replaces the three MEAN-FIELD interneuron SCALARS in the cortex
two_compartment path with three SMALL SPIKING LIF sub-populations per cortical layer — PV, SOM, VIP.

Old mean-field (spiking_brain.py):
    drive /= (1.0 + pv_g * z_prev.mean(1))                     # PV divisive gain  (~391)
    vip = float(gate); som = som_b * z[l].mean(1)              # instantaneous scalar subtraction (~435-437)
    agate = (vip - som).clamp(min=0.0)

New spiking (drop-in, SAME (B,1) shapes and signs):
    I_pv  = k_pv · pop_act                       → PV pool → r_pv → denom = 1 + pv_g·r_pv
    I_vip = k_vip · gate  (ACh 'learn-now' tone) → VIP pool → r_vip
    I_som = som_b · pop_act − w_vip_som · r_vip   → SOM pool → r_som   (SOM is disinhibited by VIP)
    agate = (r_vip − r_som).clamp(min=0.0)

Each pool τ∈{pv,som,vip} in layer l runs the SAME LIF form as the cortex (spiking_brain.py:393):
    v_τ ← β_i·v_τ·(1−z_τ) + I_τ + h_τ ;  z_τ = 1[v_τ ≥ θ_τ] ;  r_τ = mean_k z_τ ∈ [0,1]
h_τ is a small fixed seeded per-neuron heterogeneity bias so the K pool neurons desynchronize (no degenerate
all-or-none pool). The population rate r_τ IS the mean-field observable, so in the K→∞ / smoothed-threshold
limit r_pv→z.mean, r_vip→gate, r_som→som_b·z.mean − w_vip_som·gate: OFF→ON is a strict, sign-preserving
generalization that ADDS (a) membrane inertia (a real interneuron time constant across the T window), (b) a
threshold/spiking nonlinearity, (c) the genuine VIP⊣SOM synapse w_vip_som the bare scalar subtraction faked.

Refs: runs/deeper_brain_integrated_design.md §15/§16; Bellec 2020 (e-prop); Naud-Richards 2018 (apical burst);
Karnani 2016 (VIP disinhibition); Atallah 2012 (PV divisive normalization); Pfeffer/Tremblay/Adesnik.

DEVICE/DTYPE/FSDP: every tensor is created with device=ref.device and dtype=ref.dtype captured in begin() from
the cortex membrane v[0] (and from self.device/self.dtype at init) — no hard-coded cpu/cuda/float32. The pools
are SMALL REPLICATED CONTROLLER state, NOT per-neuron sharded state: their inputs are population means (z.mean
over the hidden dim) and a scalar tone gate, both local reductions of full activations, so every FSDP rank
computes the identical (B,1) gain with no all-gather of a sharded param. WIDTH-INVARIANT: K_τ is a small constant
(default 16); the only width-coupled inputs are O(1) means, so a mid-life grow() that widens a layer changes only
the reduction, never the pool shapes. Memory at hidden=128000, 2 layers, B=64: 3 types × 2 layers × B × 16
floats ≈ 24 KB transient — negligible vs the multi-GB cortex; NO dense O(N²)/O(N) per-synapse state.
"""
import torch

_SEED_OFF = {"pv": 0, "som": 1, "vip": 2}


class SpikingInterneurons:
    _KEYS = ("on", "n_pv", "n_som", "n_vip", "beta_i", "thr_pv", "thr_som", "thr_vip",
             "k_pv", "k_vip", "w_vip_som", "het")

    def __init__(self, device=None, dtype=None, seed=4242):
        self.device = device
        self.dtype = dtype
        self.seed = int(seed)
        self.on = False                        # opt-in toggle (verify before defaulting on) — clean A/B
        # pool sizes (small constants → width-invariant, FSDP-replicated controller state)
        self.n_pv = 16; self.n_som = 16; self.n_vip = 16
        # dynamics knobs (live-tunable). beta_i FASTER than the cortex membrane (c.beta=0.9) so the gate
        # has a real-but-short interneuron time constant, not a stale one.
        self.beta_i = 0.8
        self.thr_pv = 0.5; self.thr_som = 0.5; self.thr_vip = 0.5
        # feedforward drive gains onto PV / VIP. With leak beta_i=0.8 the steady-state membrane is I/(1-beta_i)=5·I,
        # so a raw gate≈1.0 would saturate VIP (r=1); k_vip=0.1 puts the drive in the threshold band → GRADED firing
        # (calibrated so the pool sits in a live dynamic range, not pinned at r=1 — the reviewer's saturation flag).
        self.k_pv = 1.0; self.k_vip = 0.1
        self.w_vip_som = 0.1                    # VIP⊣SOM disinhibitory synapse (scaled to the graded VIP rate)
        self.het = 0.2                          # per-neuron heterogeneity magnitude (widened → graded, desync pool)
        # transient per-step state (rebuilt in begin())
        self._m = {}                            # (kind,l) -> (v,z) membrane+spike, shape (B,K)
        self._h = {}                            # (kind,K,device,dtype) -> heterogeneity bias (K,)  [cached]
        self._B = 1
        # metrics (last population rates)
        self._r_pv = 0.0; self._r_som = 0.0; self._r_vip = 0.0

    # ---- fixed seeded per-neuron heterogeneity (device/dtype-matched, cached) -----------------------------
    def _het(self, kind, K):
        key = (kind, K, self.device, self.dtype)
        h = self._h.get(key)
        if h is None:
            g = torch.Generator(device="cpu").manual_seed(self.seed + _SEED_OFF[kind])
            base = torch.rand(K, generator=g) - 0.5           # centered in [-0.5, 0.5]
            h = (self.het * base).to(device=self.device, dtype=self.dtype)
            self._h[key] = h
        return h

    def _lif(self, kind, l, I, thr, K):
        """One LIF step for pool (kind,l); membrane persists across the T window within a step. Returns the
        population firing rate r = mean_k z as (B,1). I is (B,1) (broadcast to (B,K)); h is (K,)."""
        key = (kind, l)
        m = self._m.get(key)
        if m is None:
            v = torch.zeros(self._B, K, device=self.device, dtype=self.dtype)
            z = torch.zeros(self._B, K, device=self.device, dtype=self.dtype)
        else:
            v, z = m
        h = self._het(kind, K)                                 # (K,) -> broadcasts to (1,K)
        v = self.beta_i * v * (1.0 - z) + I + h                # (B,K)
        z = (v >= thr).to(self.dtype)
        self._m[key] = (v, z)
        return z.mean(1, keepdim=True)                         # (B,1)

    # ---- life-loop API (called from SpikingBrain._eprop_step) --------------------------------------------
    def begin(self, B, ref):
        """Capture device/dtype from the cortex tensor `ref` (e.g. v[0]) and reset per-step membranes."""
        self.device = ref.device
        self.dtype = ref.dtype
        self._B = int(B)
        self._m = {}

    def pv(self, l, pop_act, pv_g):
        """PV feedforward divisive somatic gain. pop_act=z_prev.mean(1,keepdim=True) is (B,1).
        Returns the divisive DENOMINATOR (1 + pv_g·r_pv) as (B,1) — the drop-in for (1 + pv_g·z.mean)."""
        I = self.k_pv * pop_act                                # (B,1)
        r = self._lif("pv", l, I, self.thr_pv, self.n_pv)      # (B,1)
        self._r_pv = float(r.mean())
        return 1.0 + float(pv_g) * r                           # (B,1)

    def apical(self, l, z_pop, gate, som_b):
        """VIP-disinhibited SOM apical gate. z_pop=z[l].mean(1,keepdim=True) is (B,1); gate is the ACh
        'learn-now' tone (float or (B,1)); som_b is the som_baseline faith knob. Returns
        agate = (r_vip − r_som).clamp(min=0) as (B,1) — the drop-in for (vip − som).clamp(min=0)."""
        if torch.is_tensor(gate):
            gate = gate.to(device=self.device, dtype=self.dtype)
            if gate.dim() < 2:
                gate = gate.reshape(-1, 1)
        else:
            gate = torch.as_tensor(float(gate), device=self.device, dtype=self.dtype).reshape(1, 1)
        I_vip = self.k_vip * gate                              # (B,1) or (1,1) → broadcasts
        r_vip = self._lif("vip", l, I_vip, self.thr_vip, self.n_vip)     # (B,1)
        I_som = float(som_b) * z_pop - self.w_vip_som * r_vip  # SOM driven by pop activity, disinhibited by VIP
        r_som = self._lif("som", l, I_som, self.thr_som, self.n_som)     # (B,1)
        self._r_vip = float(r_vip.mean()); self._r_som = float(r_som.mean())
        return (r_vip - r_som).clamp(min=0.0)                  # (B,1)

    # ---- controller boilerplate (endocrine.py / dynamics.py pattern) ------------------------------------
    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif k in ("n_pv", "n_som", "n_vip"):              # pool sizes are ints (torch.zeros needs int dim)
                v = max(1, int(float(v)))
                self._h = {}                                    # size changed → drop cached heterogeneity
            elif cur is not None:
                v = float(v)
                if k == "het":
                    self._h = {}                                # magnitude changed → recompute biases
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on, rate_pv=round(self._r_pv, 4), rate_som=round(self._r_som, 4),
                    rate_vip=round(self._r_vip, 4), n_pv=self.n_pv, n_som=self.n_som, n_vip=self.n_vip)
