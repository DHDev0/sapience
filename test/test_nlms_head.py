"""CPU tests for the §NLMS energy-normalized readout head (head_norm="energy").

The readout head reads the FULL hidden width, so the fan-in power-law ÷hidden^head_fanin_pow starves it at
scale (frozen at 128k) yet explodes at pow<0.5 — a width-dependent, narrow stable band. NLMS replaces the
divisor with the realized top-layer membrane energy ‖top_v‖²: since Δlogit_i = μ·err_i·‖v‖²/(‖v‖²+ε) → μ·err_i,
the ‖v‖² cancels ⇒ the per-step logit move is invariant to BOTH width N and sparsity ρ, so ONE head_lr_scale
transfers across widths. These tests lock in: byte-identical-when-off, the invariance itself (with a power-mode
non-invariance contrast so the test can't pass vacuously), locality/B-alignment, dtype/silent-layer safety, and
the live-toggle+clamp contract.
"""
import math
import torch
import pytest

from brain.spiking_brain import SpikingBrain


def _brain(hidden=512, layers=2, seed=0, dtype=torch.float32):
    b = SpikingBrain(torch.device("cpu"), dtype=dtype, emb=32, hidden=hidden, layers=layers,
                     seq=16, seed=seed, sparse=True, rec_fanin=32)
    b._ensure_feedback()
    return b


def _batch(V=256, B=4, T=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (B, T), generator=g)
    y = torch.randint(0, V, (B, T), generator=g)
    return x, y


def _dz_rms(b, x, y):
    """Δz_rms = std over classes of the logit move produced by ONE head update on a FIXED top_v."""
    with torch.no_grad():
        emb = b.E(x); st = [c.init_state(x.shape[0], x.device) for c in b.cells]; inp = emb
        for i, c in enumerate(b.cells):
            sp_, _, st[i] = c.run_seq(inp, st[i]); inp = sp_
        top_v0 = inp[:, -1, :].float()
        logits0 = b.head(top_v0)
    b._eprop_step(x, y, gate=1.0)
    with torch.no_grad():
        return float((b.head(top_v0) - logits0).std())


# ---- (1) byte-identical when OFF (default) --------------------------------------------------------------
def test_byte_identical_off():
    assert _brain().head_norm == "power"                       # DEFAULT is the current byte-identical rule
    x, y = _batch()
    b1, b2 = _brain(seed=0), _brain(seed=0)
    for _ in range(6):
        b1._eprop_step(x, y, gate=1.0); b2._eprop_step(x, y, gate=1.0)
    assert torch.equal(b1.head.weight, b2.head.weight)         # deterministic
    assert torch.equal(b1.E.weight, b2.E.weight)
    for a, c in zip(b1._fb, b2._fb):
        assert torch.equal(a, c)
    assert "head_dlogit" not in b1._diag                       # the energy branch NEVER ran (no v_energy, no EMA)
    assert float(b1._head_e_ema) == 0.0


# ---- (2) width-invariance (with power-mode NON-invariance contrast) --------------------------------------
def test_width_invariance():
    # The loss-relevant quantity is the per-step logit move. In energy mode head_dlogit = μ·mean|err| is width-FREE
    # (the ‖v‖² divisor cancels the ‖v‖² in the coherent re-read) → invariant across a 32x width span. The analogous
    # power-mode head weight step (head_update_mag ∝ 1/N^hpow) is width-SENSITIVE — the starvation. Asserting BOTH
    # makes the test non-vacuous: energy removed the width-coupling that power still carries.
    x, y = _batch()
    dlogit = {}; upd_pow = {}
    for h in (64, 512, 2048):                                  # 32x span
        be = _brain(hidden=h); be.set_faith(head_norm="energy", head_lr_scale=1.0, head_energy_eps=1e-2)
        for _ in range(25): be._eprop_step(x, y, gate=1.0)
        dlogit[h] = be._diag["head_dlogit"]
        bp = _brain(hidden=h); bp.set_faith(head_norm="power", head_fanin_pow=0.7)
        for _ in range(25): bp._eprop_step(x, y, gate=1.0)
        upd_pow[h] = bp._diag["head_update_mag"]
    r_e = max(dlogit.values()) / max(min(dlogit.values()), 1e-12)
    r_p = max(upd_pow.values()) / max(min(upd_pow.values()), 1e-12)
    assert r_e <= 1.3, ("energy head logit-move NOT width-invariant", dlogit, r_e)
    assert r_p >= 3.0, ("power head step unexpectedly width-invariant — contrast vacuous", upd_pow, r_p)


# ---- (3) sparsity-invariance: Δz_rms tracks ρ via ‖v‖² --------------------------------------------------
def test_sparsity_invariance():
    x, y = _batch()
    dz = {}
    for tr in (0.04, 0.12):                                    # ~3x different target firing rate → different ρ
        b = _brain(hidden=512)
        b.set_faith(head_norm="energy", head_lr_scale=0.3, head_energy_eps=1e-2,
                    homeostasis=True, target_rate=tr)
        for _ in range(20): b._eprop_step(x, y, gate=1.0)
        dz[tr] = _dz_rms(b, x, y)
    r = max(dz.values()) / max(min(dz.values()), 1e-12)
    assert r <= 2.0, ("energy head not sparsity-invariant", dz, r)   # ‖v‖² absorbs the ρ change


# ---- (4) Kolen-Pollack B stays aligned under the shared energy normalizer -------------------------------
def test_b_alignment_preserved():
    # top-B and the head get the IDENTICAL energy-normalized update, so B rotates toward W. The alignment is weak
    # per-step (energy-mode head weights are small-magnitude) but strictly TRENDS UP and never anti-aligns.
    x, y = _batch()
    b = _brain(hidden=512)
    b.set_faith(head_norm="energy", head_lr_scale=2.0, head_energy_eps=1e-2, feedback_mode="learned")
    for _ in range(40):                                        # warm past the noisy first steps
        b._eprop_step(x, y, gate=1.0)
    cos0 = b.weight_stats().get("fb_align_cos", 0.0)
    for _ in range(400):
        b._eprop_step(x, y, gate=1.0)
    cos1 = b.weight_stats().get("fb_align_cos", 0.0)
    assert cos1 > cos0, ("B not tracking the head under the shared energy normalizer", cos0, cos1)
    assert cos1 > -0.01                                        # never anti-aligns (KP, not adversarial)


# ---- (5) silent-layer guard: no NaN/Inf, step bounded by dmax -------------------------------------------
def test_silent_layer_guard():
    b = _brain(hidden=256)
    b.set_faith(head_norm="energy", head_lr_scale=1.0, head_energy_eps=1e-2)
    x = torch.zeros(4, 16, dtype=torch.long)                   # near-silent drive → meanE → 0
    y = torch.randint(0, 256, (4, 16))
    hw0 = b.head.weight.detach().clone()
    for _ in range(5):
        b._eprop_step(x, y, gate=1.0)
    assert torch.isfinite(b.head.weight).all()                 # ε floor + dmax clamp keep it finite
    assert float((b.head.weight.detach() - hw0).abs().max()) <= 0.02 * 6 + 1e-6   # per-step |Δw| ≤ dmax=0.02


# ---- (6) bf16 energy accumulates in fp32 ----------------------------------------------------------------
def test_bf16_energy_fp32():
    b = _brain(hidden=512, dtype=torch.bfloat16)
    b.set_faith(head_norm="energy", head_lr_scale=0.5, head_energy_eps=1e-2)
    x, y = _batch()
    for _ in range(5):
        b._eprop_step(x, y, gate=1.0)
    d = b._diag
    assert math.isfinite(d.get("head_energy", 0.0)) and d["head_energy"] > 0.0   # fp32 sum-of-squares didn't underflow
    assert torch.isfinite(b.head.weight.float()).all()


# ---- (7) live toggle + clamp contract -------------------------------------------------------------------
def test_live_toggle_and_clamp():
    b = _brain()
    assert b.set_faith(head_norm="energy")["head_norm"] == "energy"
    assert b.set_faith(head_norm="bogus") == {}                 # invalid value ignored (stays energy)
    assert b.head_norm == "energy"
    assert b.set_faith(head_lr_scale=-1.0)["head_lr_scale"] == 0.0   # clamped to >= 0
    assert b.set_faith(head_energy_eps=-5.0)["head_energy_eps"] == 0.0
    assert b.set_faith(head_norm="power")["head_norm"] == "power"    # toggles back to the byte-identical path
    x, y = _batch()
    b._eprop_step(x, y, gate=1.0)
    assert "head_dlogit" not in b._diag                         # power path again → no energy diagnostics
