"""CPU tests for SpikingInterneurons (PV/SOM/VIP spiking pools for the two_compartment circuit).

Covers the integration contract: on/off + string-bool coercion, drop-in (B,1) shapes & signs, rates live in
(0,1) with heterogeneity (not saturated), device/dtype propagation from a ref tensor, width-invariance (pool
shapes independent of cortex hidden width), composition with a real SpikingBrain (two_compartment) + its
cortex membrane tensors, and grow-safety (widen a layer → pools unchanged).
"""
import math
import torch
import pytest

from brain.interneurons import SpikingInterneurons


def _ref(dtype=torch.float32):
    return torch.zeros(4, 8, dtype=dtype)          # stand-in for a cortex membrane v[0] (B=4)


# ---- (1) toggle + string-bool coercion ------------------------------------------------------------------
def test_toggle_and_stringbool():
    it = SpikingInterneurons()
    assert it.on is False                          # DEFAULT OFF
    assert it.set_params(on=True)["on"] is True
    assert it.on is True
    for s in ("false", "0", "off", "no", ""):
        it.set_params(on=s); assert it.on is False, s
    for s in ("true", "1", "on", "yes"):
        it.set_params(on=s); assert it.on is True, s
    it.set_params(beta_i="0.7", n_pv="8")          # numeric knobs float; pool sizes int
    assert it.beta_i == 0.7 and isinstance(it.n_pv, int) and it.n_pv == 8


# ---- (2) shapes + agate>=0 + drop-in signs --------------------------------------------------------------
def test_shapes_and_agate_nonneg():
    it = SpikingInterneurons(); it.set_params(on=True)
    B = 4; it.begin(B, _ref())
    denom = it.pv(0, torch.rand(B, 1), pv_g=0.3)
    assert denom.shape == (B, 1)
    assert bool((denom >= 1.0).all())              # divisive denom 1 + pv_g*r_pv >= 1 (sign preserved)
    agate = it.apical(0, torch.rand(B, 1), gate=1.0, som_b=0.5)
    assert agate.shape == (B, 1) and bool((agate >= 0.0).all())
    agate2 = it.apical(0, torch.rand(B, 1), gate=torch.rand(B, 1), som_b=0.5)   # gate as (B,1) tensor
    assert agate2.shape == (B, 1) and bool((agate2 >= 0.0).all())


# ---- (3) rates in [0,1] and NOT saturated with heterogeneity --------------------------------------------
def test_rates_dynamic_range():
    it = SpikingInterneurons(); it.set_params(on=True, n_pv=64, n_som=64, n_vip=64, het=0.15)
    B = 8; it.begin(B, _ref())
    # drive near thr·(1−beta_i)=0.1 so the seeded per-neuron heterogeneity straddles threshold → a graded
    # population code (not the saturated 0/1 a supra-threshold constant input would give — that regime is
    # exactly what the A/B health-check flags; see risk #1).
    for _ in range(6):
        it.pv(0, torch.full((B, 1), 0.1), pv_g=0.3)
        it.apical(0, torch.full((B, 1), 0.1), gate=1.0, som_b=0.5)   # ACh 'learn-now' tone ≈1.0 (real regime);
        #                                                              k_vip=0.1 puts I_vip≈0.1 in the graded band
    st = it.state()
    for k in ("rate_pv", "rate_som", "rate_vip"):
        assert 0.0 <= st[k] <= 1.0, (k, st[k])
    assert 0.0 < st["rate_pv"] < 1.0 and 0.0 < st["rate_vip"] < 1.0, st   # genuine partial firing w/ heterogeneity


# ---- (4) device/dtype propagation from ref --------------------------------------------------------------
@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.bfloat16])
def test_dtype_propagation(dt):
    it = SpikingInterneurons(); it.set_params(on=True)
    B = 4; ref = _ref(dt); it.begin(B, ref)
    denom = it.pv(0, torch.rand(B, 1, dtype=dt), pv_g=0.3)
    agate = it.apical(0, torch.rand(B, 1, dtype=dt), gate=1.0, som_b=0.5)
    assert denom.dtype == dt and agate.dtype == dt
    assert denom.device == ref.device and agate.device == ref.device
    h = it._het("pv", it.n_pv)                      # heterogeneity materialized on ref device/dtype
    assert h.dtype == dt and h.device == ref.device


# ---- (5) width-invariance: pool tensors identical across cortex hidden width -----------------------------
def test_width_invariance():
    it = SpikingInterneurons(); it.set_params(on=True)
    B = 4; shapes = []
    for _hid in (64, 4096, 128000):                # width enters ONLY via the (B,1) population mean
        it.begin(B, torch.zeros(B, 1))
        pop = torch.rand(B, 1)                      # z.mean(1) over `hid` already reduced to (B,1)
        it.pv(0, pop, pv_g=0.3); it.apical(0, pop, gate=1.0, som_b=0.5)
        shapes.append(tuple(it._m[("pv", 0)][0].shape))
    assert len(set(shapes)) == 1 and shapes[0] == (B, it.n_pv)   # K constant, no dependence on hid


# ---- (6) composes with a real SpikingBrain (two_compartment) cortex tensors ------------------------------
def test_composes_with_spiking_brain():
    from brain.spiking_brain import SpikingBrain
    dev = torch.device("cpu")
    brain = SpikingBrain(dev, dtype=torch.float32, emb=16, hidden=32, layers=2, seq=8, seed=0)
    brain.set_faith(two_compartment=True)
    brain._ensure_feedback()                        # random-feedback matrices (normally lazy-inited by the loop)
    it = SpikingInterneurons(dev, dtype=torch.float32); it.set_params(on=True)
    brain.interneurons = it
    B, T = 3, 5
    x = torch.randint(0, brain.V, (B, T)); y = torch.randint(0, brain.V, (B, T))
    loss0 = brain._eprop_step(x, y, gate=1.0)       # unwired scalar path still finite (A/B baseline)
    assert math.isfinite(float(loss0))
    # drive the pools with REAL cortex membrane tensors, mirroring the wiring_patch call sites
    v0 = torch.zeros(B, brain.cells[0].hid)
    it.begin(B, ref=v0)
    z_prev = (torch.rand(B, brain.cells[0].hid) > 0.7).float()
    denom = it.pv(0, z_prev.mean(1, keepdim=True), pv_g=float(getattr(brain, "pv_gain", 0.3)))
    agate = it.apical(0, z_prev.mean(1, keepdim=True), gate=1.0, som_b=float(getattr(brain, "som_baseline", 0.5)))
    drive = torch.randn(B, brain.cells[0].hid)
    out = drive / denom                             # exactly the wiring-patch usage
    assert out.shape == drive.shape and torch.isfinite(out).all()
    assert agate.shape == (B, 1)
    assert set(("rate_pv", "rate_som", "rate_vip")).issubset(it.state().keys())


# ---- (7) grow-safety: widen a cortex layer, pools unchanged ---------------------------------------------
def test_grow_safety():
    from brain.spiking_brain import SpikingBrain
    dev = torch.device("cpu")
    brain = SpikingBrain(dev, dtype=torch.float32, emb=16, hidden=32, layers=2, seq=8, seed=0)
    it = SpikingInterneurons(dev, dtype=torch.float32); it.set_params(on=True)
    B = 3
    it.begin(B, torch.zeros(B, brain.cells[-1].hid))
    a0 = tuple(it.pv(1, torch.rand(B, 1), pv_g=0.3).shape)
    added = brain.grow(add=16)                       # widen the top layer mid-life
    assert added == 16
    it.begin(B, torch.zeros(B, brain.cells[-1].hid))
    z = (torch.rand(B, brain.cells[-1].hid) > 0.5).float()
    a1 = tuple(it.pv(1, z.mean(1, keepdim=True), pv_g=0.3).shape)   # z.mean STILL (B,1) → no resize
    assert a0 == a1 == (B, 1)


# ---- (8) live on/off toggle both directions (no restart) ------------------------------------------------
def test_live_toggle():
    it = SpikingInterneurons(); assert it.on is False
    it.set_params(on=True); assert it.on is True
    it.set_params(on=False); assert it.on is False
