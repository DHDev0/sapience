"""CPU tests for §17 LaminarMicrocircuit — the canonical Douglas–Martin/Bastos microcircuit.

Asserts: (correctness) the allowed-adjacency actually masks the forbidden lamina pairs and the
effective connectome is a strict subset of the flat pool; (a) device/dtype propagation; (b) live
toggle on/off restores the flat pool bit-identically; (c) width-scalable shapes (no dense O(N^2) on
the sparse path — every laminar tensor is O(nnz)+O(hid)); plus grow-safety (existing labels kept).
"""
import torch
import pytest

from brain.laminar import LaminarMicrocircuit, L4, L23, L56
from brain.spiking_brain import SpikingBrain

DEV = torch.device("cpu")


def _brain(hidden=192, dtype=torch.float32, sparse=True, layers=2):
    return SpikingBrain(DEV, dtype=dtype, emb=32, hidden=hidden, layers=layers,
                        cell="alif", sparse=sparse, rec_fanin=16, in_fanin=16,
                        syn_density=1.0, seed=0)


def test_defaults_and_off():
    lam = LaminarMicrocircuit(DEV)
    assert lam.on is False                                     # DEFAULT OFF (opt-in)
    st = lam.state()
    for k in ("on", "frac_L4", "frac_L23", "frac_L56", "allow_fb", "input_to_l23",
              "apical_l4_gain", "strict", "rate_L4", "rate_L23", "rate_L56", "n_forbidden_frac"):
        assert k in st


def test_string_bool_coercion():
    lam = LaminarMicrocircuit(DEV)
    assert lam.set_params(on="true")["on"] is True
    assert lam.set_params(on="off")["on"] is False            # 'off' must DISABLE, not enable
    assert lam.set_params(allow_fb="0")["allow_fb"] is False
    assert lam.set_params(strict="no")["strict"] is False
    assert lam.set_params(frac_L4="0.3")["frac_L4"] == pytest.approx(0.3)
    assert "bogus" not in lam.set_params(bogus=1)             # unknown key ignored


def test_allow_matrix_canonical():
    lam = LaminarMicrocircuit(DEV)
    A = lam._allow_matrix(DEV)
    # feedforward + recurrence allowed
    assert bool(A[L4, L4]) and bool(A[L4, L23])
    assert bool(A[L23, L23]) and bool(A[L23, L56]) and bool(A[L56, L56])
    # feedback allowed (allow_fb default True)
    assert bool(A[L56, L4]) and bool(A[L56, L23])
    # the two canonical FORBIDDEN pairs
    assert not bool(A[L23, L4]) and not bool(A[L4, L56])
    # allow_fb gates the feedback
    lam.allow_fb = False
    A2 = lam._allow_matrix(DEV)
    assert not bool(A2[L56, L4]) and not bool(A2[L56, L23])
    # strict off loosens the forbidden pairs
    lam.strict = False
    A3 = lam._allow_matrix(DEV)
    assert bool(A3[L23, L4]) and bool(A3[L4, L56])


def test_assign_lamina_fractions_and_grow_safe():
    lam = LaminarMicrocircuit(DEV)
    lam.frac_L4, lam.frac_L23, lam.frac_L56 = 0.25, 0.45, 0.30
    la = lam._assign_lamina(400, DEV)
    assert la.dtype == torch.int8
    assert int((la == L4).sum()) == 100 and int((la == L23).sum()) == 180 and int((la == L56).sum()) == 120
    # grow: existing labels preserved, new indices appended
    la2 = lam._assign_lamina(500, DEV, existing=la)
    assert torch.equal(la2[:400], la)                         # identity-safe: no relabel
    assert la2.numel() == 500


def test_rebuild_masks_forbidden_edges():
    b = _brain(hidden=192, sparse=True)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True)
    lam.rebuild(b)
    for c in b.cells:
        assert hasattr(c, "lamina") and c.lamina.numel() == c.hid
        # (c) width-scalable: per-edge mask has EXACTLY nnz entries (no dense H^2)
        assert c.lam_rec_mask.shape == c.rec_val.shape
        assert c.lam_rec_mask.dtype == torch.bool
        assert c.lam_rec_fanin.numel() == c.hid
        # every masked-in edge must satisfy the adjacency; every masked-out edge must violate it
        A = lam._allow_matrix(c.rec_val.device)
        pre = c.lamina.long()[c.rec_col.long()]; post = c.lamina.long()[c.rec_row.long()]
        assert torch.equal(c.lam_rec_mask, A[pre, post])
        # forbidden pair L23->L4 is actually absent from the effective connectome
        forbidden = (pre == L23) & (post == L4)
        assert not bool((c.lam_rec_mask & forbidden).any())
    assert 0.0 < lam._forbidden < 1.0                          # SOME edges masked, not all


def test_apical_gain_spares_l4():
    b = _brain(hidden=192, sparse=True, dtype=torch.float32)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True, apical_l4_gain=0.05)
    lam.rebuild(b)
    c = b.cells[-1]
    assert c.lam_apical_gain.dtype == b.head.weight.dtype      # (a) model dtype respected
    assert torch.allclose(c.lam_apical_gain[c.lamina == L4], torch.tensor(0.05))
    assert torch.allclose(c.lam_apical_gain[c.lamina == L23], torch.tensor(1.0))
    assert torch.allclose(c.lam_apical_gain[c.lamina == L56], torch.tensor(1.0))


def test_input_routed_to_l4():
    b = _brain(hidden=192, sparse=True)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True, input_to_l23=0.0)
    lam.rebuild(b)
    c0 = b.cells[0]                                            # layer 0 has a dense Win (sparse_in False)
    if getattr(c0, "sparse_in", False):
        post_in = c0.lamina.long()[c0.in_row.long()]
        assert bool((c0.lam_in_mask == (post_in == L4)).all())
    else:
        assert torch.equal((c0.lam_in_row > 0), (c0.lamina == L4))  # input only into L4 rows
    # deeper layer routes previous-layer output into its L4 too
    lam.set_params(input_to_l23=1.0); lam.rebuild(b)
    if getattr(c0, "sparse_in", False):
        post_in = c0.lamina.long()[c0.in_row.long()]
        assert bool((c0.lam_in_mask == ((post_in == L4) | (post_in == L23))).all())
    else:
        assert torch.equal((c0.lam_in_row > 0), (c0.lamina == L4) | (c0.lamina == L23))


def test_toggle_off_restores_flat_pool():
    b = _brain(hidden=192, sparse=True)
    rec0 = b.cells[-1].rec_val.detach().clone()               # values before laminar
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True); lam.rebuild(b)
    assert hasattr(b.cells[-1], "lam_rec_mask")
    # (b) live toggle OFF removes every derived attr; rec_val is UNTOUCHED (bit-identical flat pool)
    lam.set_params(on=False); lam.clear(b)
    for c in b.cells:
        for a in ("lamina", "lam_rec_mask", "lam_in_mask", "lam_in_row", "lam_rec_fanin",
                  "lam_apical_gain", "lam_rec_w"):
            assert not hasattr(c, a)
    assert torch.equal(b.cells[-1].rec_val.detach(), rec0)


def test_dtype_propagation_bf16():
    b = _brain(hidden=192, sparse=True, dtype=torch.bfloat16)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True); lam.rebuild(b)
    c = b.cells[-1]
    assert c.lam_apical_gain.dtype == b.head.weight.dtype      # follows the model dtype
    assert c.lam_rec_mask.dtype == torch.bool                  # masks are dtype-agnostic
    assert c.lam_rec_fanin.dtype == torch.float32              # counts kept in f32 for stable division


def test_measure_buckets_by_lamina():
    b = _brain(hidden=192, sparse=True)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True); lam.rebuild(b)
    st = lam.measure(b, "the quick brown fox jumps over the lazy dog " * 4)
    for k in ("rate_L4", "rate_L23", "rate_L56"):
        assert 0.0 <= st[k] <= 1.0


def test_grow_then_rebuild_extends_labels():
    b = _brain(hidden=192, sparse=True)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True); lam.rebuild(b)
    old = b.cells[-1].lamina.clone()
    b.grow(64)                                                 # widen the top layer (structural)
    lam.rebuild(b)                                             # must extend labels + masks
    c = b.cells[-1]
    assert c.lamina.numel() == c.hid
    assert torch.equal(c.lamina[:old.numel()], old)           # existing neurons keep their lamina
    assert c.lam_rec_mask.shape == c.rec_val.shape            # masks re-derived to new nnz


def test_dense_cell_path():
    b = _brain(hidden=64, sparse=False, layers=2)             # dense LIF cells (small test net)
    lam = LaminarMicrocircuit(DEV, dtype=b.head.weight.dtype)
    lam.set_params(on=True); lam.rebuild(b)
    c = b.cells[-1]
    assert c.lam_rec_w.shape == (c.hid, c.hid) and c.lam_rec_w.dtype == torch.bool
    # dense mask[out,in] = A[pre=in, post=out]
    A = lam._allow_matrix(DEV); lam_l = c.lamina.long()
    assert bool((c.lam_rec_w == A[lam_l.view(1, -1), lam_l.view(-1, 1)]).all())
