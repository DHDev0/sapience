"""Dendritic NMDA plateau potentials — the regenerative apical nonlinearity controller.

The UNIFIED two-compartment circuit (spiking_brain.py _eprop_step) today treats the apical
compartment ap[l] as a purely LINEAR leaky integrator of the VIP/SOM-gated top-down error:

    ap = beta_ap*ap + agate*Lsig                       (linear, fast leak beta_ap≈0.9)

read into the soma as g_ap*ap and thresholded into the burst code brst. Real apical dendrites are
NOT linear: an NMDA-spike / Ca2+ plateau is an all-or-none, REGENERATIVE, LATCHED event that ignites
when apical glutamate drive coincides with a somatic (back-propagating) spike, then SUSTAINS
depolarisation for tens of ms on its own slow timescale — the biophysical substrate of the apical
burst and of behavioral-timescale credit (Schiller & Schiller 2000; Larkum 2013; Major/Larkum/Schiller
2013 Annu Rev Neurosci; Bittner & Magee 2017 BTSP).

This class is ONLY the slow replicated controller + the toggle (the endocrine/dynamics §16 pattern):
pure python-float scalars, DEFAULT OFF, live-tunable, device/dtype/FSDP-trivially-safe. The per-neuron
plateau tensors plat[l]/pclk[l] are ACTIVATIONS created inside _eprop_step on the cortex device+dtype
(they ride on ap[l], never a host-only copy of a sharded param), so the mechanism itself lives in the
hot loop. The equations it parameterises (per timestep tt, per layer l, tensors (B,hid)):

    thr_p = p_thr * (mean|ap| + 1e-9)                   # RELATIVE threshold → width-invariant
    trig  = 1[|ap| > thr_p] * z * 1[pclk<=0]            # NMDA coincidence (Mg2+ unblock) + refractory latch
    pclk  = where(trig, D_eff, pclk)                    # ignite an all-or-none window of D_eff ticks
    plat  = plat + trig * p_gain * ap                   # supralinear regenerative seed
    act   = 1[pclk>0]; plat = rho_p * plat * act        # SUSTAIN (slow tau_p) then CLEAR
    pclk  = clamp(pclk-1, min=0)
    apd   = ap + plat                                   # effective apical membrane → soma AND burst

rho_p = exp(-1/tau_p) with tau_p >> 1/(1-beta_ap): the plateau OUTLASTS the linear compartment (a true
plateau, not a fast EPSP). All thresholds are RELATIVE to the population apical scale — like the existing
burst_thr — so trigger rate and magnitude do NOT drift with hidden width (no fan-in/width starvation).

References: runs/deeper_brain_integrated_design.md (apical two-compartment), the existing burst code
(spiking_brain.py ~438-442), btsp_beta (behavioral-timescale eligibility).
"""


class DendriticPlateau:
    _KEYS = ("on", "p_thr", "p_gain", "rho_p", "dur", "btsp_couple")

    def __init__(self, device=None, dtype=None):
        # Controller carries ONLY replicated scalars → identical on every FSDP rank; the per-neuron
        # plateau state is allocated in _eprop_step on this device+dtype (which we record for reference).
        self.device = device
        self.dtype = dtype
        self.on = False                        # DEFAULT OFF (must EARN its keep via the delayed-credit A/B)
        self.p_thr = 1.2                       # NMDA relative-threshold (× mean|ap|); >1 ⇒ sparse, mean-relative
        self.p_gain = 0.5                      # supralinear regenerative seed gain (keep modest ≤0.5: runaway guard)
        self.rho_p = 0.95                      # plateau sustain factor = exp(-1/tau_p); tau_p≈20 ≫ apical 1/(1-0.9)=10
        self.dur = 8.0                         # all-or-none window length D (ticks); refractory against chatter
        self.btsp_couple = 0.5                 # plateau-rate → e-prop eligibility stretch (toward btsp_beta)
        self._last_rate = 0.0                  # last top-layer plateau_rate (metric write-back from the cortex)

    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' must DISABLE, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif cur is not None:
                v = float(v)
            if k == "dur":         v = max(1.0, v)             # ≥1 tick (survives the 5-HT patience shrink)
            elif k == "rho_p":     v = min(0.999, max(0.0, v)) # sustain factor <1 ⇒ bounded (runaway guard)
            elif k in ("p_thr", "p_gain", "btsp_couple"): v = max(0.0, v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on, p_thr=self.p_thr, p_gain=self.p_gain, rho_p=self.rho_p,
                    dur=self.dur, btsp_couple=self.btsp_couple,
                    plateau_rate=round(self._last_rate, 4))
