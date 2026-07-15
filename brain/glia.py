"""§17 · SpikingGlia — the slow astrocytic activation field (tripartite-synapse modulation).

A per-neuron ASTROCYTIC ACTIVATION field a_l ∈ R^{H_l} (one entry per postsynaptic cortical neuron,
per layer) that ensheathes each neuron's synapses (the tripartite synapse). It is a leaky integrator of
that neuron's LOCAL firing over seconds-to-minutes — an order of magnitude SLOWER than the eligibility
trace and the fast §5 / two_compartment SOM inhibition, one order FASTER than the §16 endocrine hormones.

Per physics tick the cortex hands glia a per-neuron rate r_l[j] = (Σ_{b,t} z_l[j])/(B·T)  (glia divides by
target_rate to get the dimensionless activity ratio), and the field relaxes toward it:

        a_l ← (1 − 1/τ_a)·a_l + (1/τ_a)·(r_l / target_rate)          (τ_a ≈ 300 ticks)

Because r_l is normalized by target_rate, a is a dimensionless activity ratio — a=1 ⇔ firing exactly at the
homeostatic target, so a>1 flags LOCAL representation runaway. The field is O(1) at ANY width (the
width-invariance that mirrors the fan-in-normalized cortex — it introduces no width-dependent divisor).

From the field glia derives THREE one-sided outputs (calm = no effect; only OVER-activity brakes):
  (1) per-postsynaptic-neuron metaplastic gain  p_l[j] = 1/(1 + k_p·relu(a_l[j] − 1))  — a chronically
      over-firing astrocyte domain DOWN-gates the plasticity of exactly its own synapses; a per-neuron
      FIELD that damps LOCALIZED runaway the global attention/cortisol scalars are blind to.
  (2) a slow scalar gliotransmission companion  G = 1 − k_g·relu(mean(a) − 1) ∈ (0,1]  — multiplied into
      the SAME life.py plasticity gate next to endocrine.plasticity_gain(): the region-wide slow brake.
  (3) a metabolic multiplier  m = 1 + k_m·mean(relu(a − 1)) ≥ 1  — scales the cortex §15.17
      metabolic_lambda (astrocyte-neuron lactate shuttle / ATP scarcity: sustained activation raises the
      spike-rate energy penalty that pushes incoming weights down).

During sleep the field clears toward baseline (glymphatic clearance): a_l ← a_l·(1 − ρ_clear) each
_sleep_tick — coupling glial "metabolic debt" to the wake/sleep cycle.

Theory: tripartite synapse + astrocytic gliotransmission (Perea/Navarrete/Araque 2009; Araque 1999;
Henneberger 2010 D-serine→LTP), astrocyte-neuron lactate shuttle (Pellerin & Magistretti 1994),
glymphatic clearance in sleep (Xie 2013). The update is a normalized metaplastic state variable
(Abraham-Bear 1996). The one-sided relu(a−1) form is deliberate (same lesson as the §16 endocrine A/B,
where a bidirectional inverted-U wrongly throttled calm learning): glia can only STABILIZE, never starve.

DEVICE/DTYPE/FSDP: every field tensor is created on self.device with self.dtype (baseline 1.0, neutral),
never hard-coded cpu/float32. The field is indexed by POSTSYNAPTIC neuron — the exact partition dimension
of the cortex per-neuron buffers (_thr_adapt, _ei_sign, in_bias), so it shards on the SAME partition as
the layer it rides under FSDP2 and is a buffer, not a Parameter. The reductions to G, m and the metrics
are all-reduce-means over the sharded field (small replicated controller scalars).

SCALE: field memory = Σ_l H_l = 2×128000 = 256000 floats ≈ 1 MB (fp32) / 0.5 MB (bf16), plus a transient
rate vector of the same size — strictly O(neurons), never O(N²). The plasticity gain is applied as a
per-row scale on the already-O(nnz) sparse gradient — no new dense synapse state. DEFAULT OFF.
"""
import torch


class SpikingGlia:
    _KEYS = ("on", "tau_a", "k_p", "k_g", "k_m", "rho_clear", "target_rate", "auto_target")

    def __init__(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        self.on = False                        # opt-in toggle (verify before defaulting on)
        # per-layer astrocytic field, lazily built to match cortex layer widths on first sense/ensure_width.
        # baseline 1.0 = "firing at the homeostatic target" → relu(a-1)=0 → every gain neutral.
        self.a = []                            # list[Tensor(H_l)] on self.device / self.dtype
        # rates (live-tunable) — timescale between the fast eligibility (~10s of ticks) and the endocrine
        # hormones (~hundreds of ticks): τ_a≈300 ticks = seconds-to-minutes.
        self.tau_a = 300.0
        self.k_p = 2.0                          # per-neuron metaplastic brake strength
        self.k_g = 0.5                          # region-wide gliotransmission brake strength
        self.k_m = 1.0                          # metabolic-scarcity multiplier strength
        self.rho_clear = 0.05                   # glymphatic clearance fraction per _sleep_tick
        # The astrocyte "over-activity" reference. A HARD-CODED 0.08 left global_gain/astro_pgain pinned at 1.0
        # forever whenever the net fires below it (measured mean ≈0.02-0.04 ⇒ mean_a≈0.27<1 ⇒ relu(·)=0 always) —
        # a dead metric. auto_target SELF-CALIBRATES the reference to a slow EMA of the population's OWN mean rate
        # (rate_ema_beta slower than tau_a), so the field hovers near 1.0 in whatever regime the cortex occupies
        # and relu(a−1) catches genuine excursions ABOVE that baseline. Scale/regime-invariant. Fixed target still
        # available via auto_target=False (uses target_rate).
        self.target_rate = 0.04                 # fixed-mode homeostatic reference (used only when auto_target=False)
        self.auto_target = True                 # self-calibrate the reference to the observed rate EMA
        self.rate_ema_beta = 0.002              # baseline EMA rate (τ≈500 ticks, slower than tau_a so a can excurse)
        self._rate_ema = None                   # running population mean-rate baseline (built on first sense)

    # ---- field maintenance ------------------------------------------- #
    def ensure_width(self, hids):
        """Build/extend the field to match the current per-layer widths [H_0, H_1, ...]. New (grown)
        neurons start at baseline 1.0 (neutral / unthrottled), mirroring _ensure_ei's grow-aware extension
        so a mid-life grow() keeps the field consistent and identity-safe. Idempotent."""
        if hids is None:
            return
        while len(self.a) < len(hids):
            self.a.append(torch.ones(int(hids[len(self.a)]), device=self.device, dtype=self.dtype))
        for l, h in enumerate(hids):
            h = int(h); cur = self.a[l].numel()
            if cur < h:                                                    # layer grew → pad with baseline
                pad = torch.ones(h - cur, device=self.a[l].device, dtype=self.a[l].dtype)
                self.a[l] = torch.cat([self.a[l], pad])
            elif cur > h:                                                  # (defensive) layer shrank
                self.a[l] = self.a[l][:h]

    def sense(self, rate_vec):
        """Slow-integrate this step's per-neuron rate. rate_vec = list of per-layer tensors
        r_l = (Σ_{b,t} z_l)/(B·T) (length H_l, on the cortex device/dtype). No-op if off / None / empty."""
        if not self.on or rate_vec is None:
            return
        hids = [int(r.numel()) for r in rate_vec]
        self.ensure_width(hids)
        # self-calibrating reference: slow EMA of the observed population mean firing rate
        tot = sum(float(r.detach().sum()) for r in rate_vec); cnt = sum(int(r.numel()) for r in rate_vec)
        mrate = tot / max(cnt, 1)
        self._rate_ema = mrate if self._rate_ema is None \
            else (1.0 - self.rate_ema_beta) * self._rate_ema + self.rate_ema_beta * mrate
        ref = self._rate_ema if self.auto_target else float(self.target_rate)
        # bound the self-calibrating reference to a band anchored on target_rate: it tracks the normal regime
        # (fixing the flat-at-1.0 gain) but CANNOT normalise away a genuine runaway (>4× target always registers
        # as over-activity, however sustained) — keeps both regime-robustness AND absolute over-firing detection.
        tr = min(max(ref, 0.25 * float(self.target_rate)), 4.0 * float(self.target_rate))
        tr = max(tr, 1e-6)
        inv = 1.0 / max(float(self.tau_a), 1.0)
        for l, r in enumerate(rate_vec):
            ratio = (r.detach().to(self.a[l].device, self.a[l].dtype)) / tr   # dimensionless activity ratio
            self.a[l] = (1.0 - inv) * self.a[l] + inv * ratio

    def sleep_tick(self):
        """Glymphatic clearance: relax the field toward baseline, clearing accumulated metabolic debt."""
        if not self.on:
            return
        keep = 1.0 - max(0.0, min(1.0, float(self.rho_clear)))
        for l in range(len(self.a)):
            self.a[l] = self.a[l] * keep

    # ---- the three one-sided outputs --------------------------------- #
    def pgain_per_layer(self, hids=None):
        """Per-postsynaptic-neuron metaplastic gain p_l[j] = 1/(1 + k_p·relu(a_l[j]−1)) ∈ (0,1]. Returns a
        list of per-layer tensors (rows = postsynaptic neurons) to scale the APPLIED e-prop update, or None
        if the field is not built yet (→ the cortex treats it as no-op = full plasticity)."""
        if not self.on:
            return None
        if hids is not None:
            self.ensure_width(hids)
        if not self.a:
            return None
        return [1.0 / (1.0 + float(self.k_p) * (a - 1.0).clamp(min=0.0)) for a in self.a]

    def global_gain(self):
        """Region-wide gliotransmission companion G = 1 − k_g·relu(mean(a)−1) ∈ (0,1] (floored). One-sided:
        calm activity leaves it at 1.0; only over-activity brakes. Multiplied into the life.py plasticity
        gate next to endocrine.plasticity_gain()."""
        if not self.on or not self.a:
            return 1.0
        over = max(0.0, self._mean_a() - 1.0)
        return max(0.05, 1.0 - float(self.k_g) * over)

    def metab_mult(self):
        """Metabolic multiplier m = 1 + k_m·mean(relu(a−1)) ≥ 1 that scales the cortex metabolic_lambda.
        Sustained astrocyte activation signals energy scarcity → a larger spike-rate penalty."""
        if not self.on or not self.a:
            return 1.0
        excess = torch.cat([(a - 1.0).clamp(min=0.0).float() for a in self.a]).mean()
        return min(10.0, 1.0 + float(self.k_m) * float(excess))

    # ---- helpers ----------------------------------------------------- #
    def _mean_a(self):
        return float(torch.cat([a.float() for a in self.a]).mean()) if self.a else 1.0

    def _overactive_frac(self):
        if not self.a:
            return 0.0
        cat = torch.cat([a.float() for a in self.a])
        return float((cat > 1.0).float().mean())

    def load_field(self, saved, hids):
        """Restore the field with a growth-invariant length guard: keep a saved a_l ONLY if its length
        matches the current layer width H_l, else fall back to the baseline-init field for that layer (so a
        GROWN brain still restores — mirrors the bg_M shape guard)."""
        self.a = []
        saved = saved or []
        for l, h in enumerate(hids):
            h = int(h)
            if l < len(saved) and saved[l] is not None and int(saved[l].numel()) == h:
                self.a.append(saved[l].to(device=self.device, dtype=self.dtype))
            else:
                self.a.append(torch.ones(h, device=self.device, dtype=self.dtype))

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
            setattr(self, k, v)
            applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on,
                    astro_activation=round(self._mean_a(), 4),
                    astro_overactive_frac=round(self._overactive_frac(), 4),
                    astro_pgain=round(1.0 / (1.0 + float(self.k_p) * max(0.0, self._mean_a() - 1.0)), 4),
                    astro_metab_mult=round(self.metab_mult(), 4),
                    glia_global_gain=round(self.global_gain(), 4),
                    auto_target=self.auto_target,                    # calibration mode (observable)
                    glia_ref_rate=round(float(self._rate_ema), 5) if self._rate_ema is not None else None,
                    k_g=self.k_g, k_p=self.k_p, target_rate=self.target_rate)
