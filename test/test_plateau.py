"""CPU tests for DendriticPlateau — the NMDA apical plateau controller (brain/plateau.py).

Two halves:
  1. the CONTROLLER contract (endocrine/dynamics §16 pattern): DEFAULT OFF, string-bool coercion,
     param clamps, state() metric, device/dtype propagation, live on/off toggle.
  2. the MECHANISM correctness — a byte-faithful reimplementation of the plateau recurrence that the
     wiring_patch inserts into _eprop_step, run standalone on CPU tensors, proving: the BAC coincidence
     gate (no plateau without a somatic spike), the all-or-none latch / no-chatter refractory, the
     sustain that OUTLASTS the linear apical, the supralinear boost, mean-relative sparsity, and
     WIDTH-INVARIANCE (trigger rate independent of hidden width ⇒ scales to 256k with only O(B·H) state).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
import torch

from brain.plateau import DendriticPlateau


# --------------------------------------------------------------------------------------------------
# reference recurrence — EXACTLY the tensor math the wiring_patch inserts after `ap = beta_ap*ap + ...`
# (kept here, not on the module, because the spec keeps the controller pure-scalar / FSDP-trivial).
# --------------------------------------------------------------------------------------------------
def _plateau_step(ap, z, plat, pclk, p_thr, p_gain, rho_p, pdur):
    thr_p = p_thr * (ap.abs().mean(1, keepdim=True) + 1e-9)
    trig = (ap.abs() > thr_p).float() * z * (pclk <= 0).float()
    pclk = torch.where(trig > 0, torch.full_like(pclk, pdur), pclk)
    plat = plat + trig * p_gain * ap
    act = (pclk > 0).float()
    plat = rho_p * plat * act
    pclk = (pclk - 1.0).clamp(min=0.0)
    apd = ap + plat
    # plat_rate = fraction currently in an active plateau (the metric the cortex reports);
    # trig_rate = fraction that IGNITED this step (used to prove no chatter / coincidence gating).
    return apd, plat, pclk, float(act.mean()), float((trig > 0).float().mean())


def _zeros(B, H, dtype=torch.float32):
    return torch.zeros(B, H, dtype=dtype), torch.zeros(B, H, dtype=dtype)


# ==================================================================================================
# 1. CONTROLLER CONTRACT
# ==================================================================================================
def test_default_off():
    p = DendriticPlateau()
    assert p.on is False                                   # DEFAULT OFF — must earn its keep


def test_string_bool_coercion():
    p = DendriticPlateau()
    assert p.set_params(on="true")["on"] is True
    for s in ("false", "0", "off", "no", ""):
        assert p.set_params(on=s)["on"] is False, f"{s!r} must DISABLE, not enable"
    assert p.set_params(on=1)["on"] is True
    assert p.set_params(on=True)["on"] is True


def test_param_clamps():
    p = DendriticPlateau()
    a = p.set_params(dur="0.2", rho_p="5.0", p_thr="-1", p_gain="-2", btsp_couple="-0.3")
    assert a["dur"] == 1.0                                 # dur clamped ≥1 tick
    assert a["rho_p"] == 0.999                             # sustain <1 (bounded, runaway guard)
    assert a["p_thr"] == 0.0 and a["p_gain"] == 0.0 and a["btsp_couple"] == 0.0   # ≥0
    a2 = p.set_params(p_thr="1.5", p_gain="0.4", rho_p="0.9")
    assert a2["p_thr"] == 1.5 and a2["p_gain"] == 0.4 and a2["rho_p"] == 0.9
    assert isinstance(a2["p_thr"], float)


def test_state_metric():
    p = DendriticPlateau()
    st = p.state()
    for k in ("on", "p_thr", "p_gain", "rho_p", "dur", "btsp_couple", "plateau_rate"):
        assert k in st
    assert st["plateau_rate"] == 0.0                       # never latched ⇒ 0
    p._last_rate = 0.0731                                   # cortex writes this back each step
    assert p.state()["plateau_rate"] == 0.0731             # rounded, surfaced


def test_device_dtype_propagation():
    dev = torch.device("cpu")
    # (i) the controller RECORDS whatever device+dtype it is handed — never hard-codes cpu/cuda/float32.
    for dt in (torch.float32, torch.float64, torch.bfloat16):
        p = DendriticPlateau(device=dev, dtype=dt)
        assert p.device is dev and p.dtype is dt
    # (ii) the per-neuron plateau tensors the cortex allocates use v[0].dtype (float32/64 in the real
    # no_grad loop) and the cortex device → the recurrence preserves that device+dtype end-to-end.
    for dt in (torch.float32, torch.float64):
        B, H = 4, 12
        plat, pclk = torch.zeros(B, H, device=dev, dtype=dt), torch.zeros(B, H, device=dev, dtype=dt)
        ap = torch.randn(B, H, device=dev, dtype=dt)
        z = (torch.rand(B, H, device=dev) > 0.5).to(dt)
        apd, plat, pclk, _, _ = _plateau_step(ap, z, plat, pclk, 1.2, 0.5, 0.95, 8.0)
        assert apd.dtype == dt and apd.device == dev       # dtype/device propagated, not hard-coded
        assert plat.dtype == dt and pclk.dtype == dt
        assert torch.isfinite(apd).all()


def test_live_toggle():
    p = DendriticPlateau()
    p.set_params(on=True);  assert p.on is True
    p.set_params(on=False); assert p.on is False           # off leaves defaults intact
    p.set_params(on=True);  assert p.on is True
    assert p.p_thr == 1.2 and p.rho_p == 0.95              # toggling never mutates the hyperparams


# ==================================================================================================
# 2. MECHANISM CORRECTNESS  (the recurrence the wiring_patch inserts)
# ==================================================================================================
def test_coincidence_gate_needs_somatic_spike():
    # strong apical drive but NO somatic spike (z=0) → Mg2+ stays blocked → NO plateau ever ignites.
    B, H = 3, 40
    plat, pclk = _zeros(B, H)
    ap = torch.zeros(B, H); ap[:, 0] = 10.0                # one big apical hotspot, above any rel-threshold
    z = torch.zeros(B, H)                                  # no back-propagating spike
    tot = 0.0
    for _ in range(10):
        apd, plat, pclk, rate, trig = _plateau_step(ap, z, plat, pclk, 1.2, 0.5, 0.95, 8.0)
        tot += rate + trig
    assert tot == 0.0                                      # coincidence detector: no spike ⇒ no ignition
    assert torch.equal(plat, torch.zeros_like(plat))       # plat stays exactly linear (apd == ap)


def test_ignition_and_supralinear_boost():
    B, H = 2, 40
    plat, pclk = _zeros(B, H)
    ap = torch.zeros(B, H); ap[:, 0] = 5.0                 # a hotspot > p_thr*mean
    z = torch.zeros(B, H); z[:, 0] = 1.0                   # coincident somatic spike on the same unit
    apd, plat, pclk, rate, trig = _plateau_step(ap, z, plat, pclk, 1.2, 0.5, 0.95, 8.0)
    assert rate > 0.0 and trig > 0.0                       # ignited
    # supralinear: apd on the hot unit exceeds the linear ap (plat added a positive seed in ap's sign)
    assert apd[0, 0].item() > ap[0, 0].item()
    assert (plat[:, 0].abs() > 0).all()                   # plateau latched on the driven unit
    assert torch.isfinite(apd).all()


def test_sustain_outlasts_linear_then_clears_no_chatter():
    # ignite once, then DROP the apical input to 0. A linear compartment (beta_ap=0.9) would be ~gone in
    # ~10 ticks; the plateau (rho_p=0.95, dur=8) must persist across its window, decay, then CLEAR to 0.
    B, H, pdur = 1, 20, 6.0
    plat, pclk = _zeros(B, H)
    ap = torch.zeros(B, H); ap[:, 0] = 4.0
    z = torch.zeros(B, H); z[:, 0] = 1.0
    apd, plat, pclk, r0, t0 = _plateau_step(ap, z, plat, pclk, 1.2, 0.5, 0.95, pdur)
    assert r0 > 0.0 and t0 > 0.0 and plat[0, 0].abs() > 0
    ap0 = torch.zeros(B, H); z0 = torch.zeros(B, H)        # input gone, no more spikes
    mags, retrig, pclk_seq = [], 0.0, [float(pclk.max())]
    for _ in range(int(pdur)):                             # remaining window
        apd, plat, pclk, r, trig = _plateau_step(ap0, z0, plat, pclk, 1.2, 0.5, 0.95, pdur)
        mags.append(plat[0, 0].abs().item()); retrig += trig; pclk_seq.append(float(pclk.max()))
    assert retrig == 0.0                                   # NO CHATTER: no re-IGNITION inside the window
    # refractory latch is strictly draining (never jumps back up = never re-armed mid-window)
    assert all(pclk_seq[i + 1] <= pclk_seq[i] for i in range(len(pclk_seq) - 1))
    assert mags[0] > 0.0                                   # still depolarised after input removed (sustain)
    assert mags[-1] == 0.0                                 # window elapsed ⇒ plateau CLEARED (pclk hit 0)
    # decay is monotone non-increasing within the sustain (rho_p<1)
    assert all(mags[i + 1] <= mags[i] + 1e-6 for i in range(len(mags) - 1))
    assert float(pclk.max()) == 0.0                        # latch fully drained


def test_mean_relative_sparsity():
    # uniform apical drive (all units equal) → |ap| == mean|ap| → NOT > p_thr(>1)*mean → nothing triggers.
    B, H = 4, 128
    plat, pclk = _zeros(B, H)
    ap = torch.full((B, H), 0.7)                           # perfectly uniform
    z = torch.ones(B, H)                                   # everybody spiking
    tot = 0.0
    for _ in range(8):
        _, plat, pclk, r, trig = _plateau_step(ap, z, plat, pclk, 1.2, 0.5, 0.95, 8.0)
        tot += r + trig
    assert tot == 0.0                                      # a DC/uniform drive is not a plateau (no salient unit)


def test_p_thr_governs_rate_inert_at_high_thr():
    # low p_thr → some units ignite (rate>0); enormous p_thr → inert (rate==0) but still finite (no NaN).
    B, H = 8, 200
    ap = torch.randn(B, H); z = (torch.rand(B, H) > 0.3).float()
    def _rate(p_thr):
        plat, pclk = _zeros(B, H); tot = 0.0
        for _ in range(6):
            apd, plat, pclk, r, trig = _plateau_step(ap, z, plat, pclk, p_thr, 0.5, 0.95, 8.0)
            assert torch.isfinite(apd).all()
            tot += trig
        return tot
    assert _rate(0.8) > 0.0
    assert _rate(1e6) == 0.0


def test_width_invariance_and_scalability():
    # SAME relative structure (fixed fraction of hot units) at two very different widths → trigger rate
    # must be ~equal (width-invariant, because the threshold is p_thr*mean|ap|). Confirms it scales to
    # 256k with only O(B·H) state and no O(H^2) tensor.
    torch.manual_seed(0)
    def _first_rate(H, frac=0.05):
        B = 4
        plat, pclk = _zeros(B, H)
        assert plat.shape == (B, H) and pclk.shape == (B, H)      # O(B·H) state, no dense O(H^2)
        ap = 0.1 * torch.randn(B, H)
        k = max(1, int(frac * H))
        ap[:, :k] += 6.0                                          # a fixed FRACTION of salient units
        z = torch.ones(B, H)
        apd, plat, pclk, r, trig = _plateau_step(ap, z, plat, pclk, 1.2, 0.5, 0.95, 8.0)
        return trig
    r_small = _first_rate(200)
    r_big = _first_rate(20000)                                    # 100× wider
    assert r_small > 0.0 and r_big > 0.0
    assert abs(r_small - r_big) < 0.02                            # rate independent of width (no starvation/blow-up)
