"""CPU tests for §15.18 SpikingSTDP — the pair-based asymmetric STDP kernel.

Covers: (1) the governing-equation WINDOW SIGN (pre-before-post → LTP+, post-before-pre → LTD−) and its
asymmetry; (2) device/dtype propagation (constructed with a dtype; bf16 input → correct-device output);
(3) live toggle on→off→on via set_params with the string-bool 'on' coercion; (4) width-scalability — the
kernel is O(nnz) edge-wise with NO dense O(H²) tensor (runs at a width whose H² would be huge); (5) the
sddmm/cortex index convention (edge e: pre=col[e] → post=row[e]); (6) decay() = exp(-1/τ); (7) state()
metrics (ltp_ltd_balance ∈ [0,1], stdp_net ∈ [-1,1], stdp_mag).

Run: cd sapience && CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" python -m pytest test/test_stdp.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.stdp import SpikingSTDP


def _one_edge():
    """A single recurrent edge pre=neuron0 → post=neuron1 over a 2-neuron population, B=1."""
    row = torch.tensor([1])          # post = neuron 1
    col = torch.tensor([0])          # pre  = neuron 0
    return row, col


def test_decay_matches_exp():
    s = SpikingSTDP()
    s.tau_plus, s.tau_minus = 4.0, 6.0
    lp, lm = s.decay()
    assert abs(lp - math.exp(-1 / 4.0)) < 1e-12
    assert abs(lm - math.exp(-1 / 6.0)) < 1e-12
    assert lm > lp                                   # longer τ₋ → wider (slower-decaying) depression window


def test_window_sign_pre_before_post_is_LTP():
    """Pre fired recently (trace>0), post fires NOW → potentiation: delta > 0, ltp_sum>0, ltd_sum≈0."""
    s = SpikingSTDP()
    row, col = _one_edge()
    B, N = 1, 2
    z = torch.zeros(B, N); z[0, 1] = 1.0             # post (neuron 1) spikes now
    x_pre = torch.zeros(B, N); x_pre[0, 0] = 0.8     # pre (neuron 0) fired recently → nonzero pre-trace
    y_post = torch.zeros(B, N)                       # post has no prior trace
    delta, ltp, ltd = s.edge_delta(z_post=z, z_pre=z, x_pre=x_pre, y_post=y_post, row=row, col=col)
    assert delta.shape == (1,)
    assert delta.item() > 0.0                        # net potentiation
    assert ltp > 0.0 and ltd == 0.0
    assert abs(delta.item() - s.a_plus * 0.8) < 1e-6


def test_window_sign_post_before_pre_is_LTD():
    """Post fired recently (trace>0), pre fires NOW → depression: delta < 0, ltd_sum>0, ltp_sum≈0."""
    s = SpikingSTDP()
    row, col = _one_edge()
    B, N = 1, 2
    z = torch.zeros(B, N); z[0, 0] = 1.0             # pre (neuron 0) spikes now
    x_pre = torch.zeros(B, N)                        # pre has no prior trace
    y_post = torch.zeros(B, N); y_post[0, 1] = 0.8   # post (neuron 1) fired recently → nonzero post-trace
    delta, ltp, ltd = s.edge_delta(z_post=z, z_pre=z, x_pre=x_pre, y_post=y_post, row=row, col=col)
    assert delta.item() < 0.0                        # net depression
    assert ltd > 0.0 and ltp == 0.0
    assert abs(delta.item() + s.a_minus * 0.8) < 1e-6


def test_asymmetry_ltd_bias():
    """Identical coincidence magnitude on both channels → net DEPRESSION (a_minus > a_plus stability bias)."""
    s = SpikingSTDP()
    row, col = _one_edge()
    B, N = 1, 2
    z = torch.zeros(B, N); z[0, 0] = 1.0; z[0, 1] = 1.0     # both fire now
    tr = torch.zeros(B, N); tr[0, 0] = 1.0; tr[0, 1] = 1.0  # both have equal recent trace
    delta, ltp, ltd = s.edge_delta(z_post=z, z_pre=z, x_pre=tr, y_post=tr, row=row, col=col)
    assert ltd > ltp                                 # a_minus·1 > a_plus·1
    assert delta.item() < 0.0


def test_input_matrix_convention():
    """Feedforward matrix: post population (hid) ≠ pre population (in_dim). Kernel handles distinct shapes."""
    s = SpikingSTDP()
    B, hid, in_dim = 1, 3, 4
    row = torch.tensor([2, 0])                       # post neurons (in hid-space)
    col = torch.tensor([3, 1])                       # pre neurons  (in in_dim-space)
    z_post = torch.zeros(B, hid); z_post[0, 2] = 1.0             # post 2 spikes now
    x_pre = torch.zeros(B, in_dim); x_pre[0, 3] = 0.5           # pre 3 fired recently
    z_pre = torch.zeros(B, in_dim)
    y_post = torch.zeros(B, hid)
    delta, ltp, ltd = s.edge_delta(z_post=z_post, z_pre=z_pre, x_pre=x_pre, y_post=y_post, row=row, col=col)
    assert delta.shape == (2,)
    assert delta[0].item() > 0.0 and abs(delta[1].item()) < 1e-9   # only edge0 (post2←pre3) potentiates


def test_soft_ceiling_shrinks_ltp():
    """Passing w near w_ceiling shrinks LTP toward 0 (Gütig soft bound)."""
    s = SpikingSTDP(); s.w_ceiling = 1.0
    row, col = _one_edge()
    B, N = 1, 2
    z = torch.zeros(B, N); z[0, 1] = 1.0
    x_pre = torch.zeros(B, N); x_pre[0, 0] = 1.0
    y_post = torch.zeros(B, N)
    d_free, _, _ = s.edge_delta(z, z, x_pre, y_post, row, col, w=torch.tensor([0.0]))
    d_bound, _, _ = s.edge_delta(z, z, x_pre, y_post, row, col, w=torch.tensor([0.95]))
    assert d_bound.item() < d_free.item()
    assert abs(d_bound.item() - s.a_plus * (1.0 - 0.95)) < 1e-6


def test_device_dtype_propagation():
    """Constructed with a dtype; bf16 inputs on cpu → float32 cpu delta (matmul-accumulate convention)."""
    s = SpikingSTDP(device=torch.device("cpu"), dtype=torch.bfloat16)
    assert s.dtype == torch.bfloat16 and s.device.type == "cpu"
    row, col = _one_edge()
    z = torch.zeros(1, 2, dtype=torch.bfloat16); z[0, 1] = 1.0
    x_pre = torch.zeros(1, 2, dtype=torch.bfloat16); x_pre[0, 0] = 1.0
    y_post = torch.zeros(1, 2, dtype=torch.bfloat16)
    delta, ltp, ltd = s.edge_delta(z, z, x_pre, y_post, row, col)
    assert delta.device.type == "cpu"
    assert delta.item() > 0.0                        # bf16 in → still potentiates


def test_batch_sum_and_chunking():
    """Batch>1 sums coincidences over B; forcing a tiny cap exercises the chunk loop → same result."""
    s = SpikingSTDP()
    row, col = _one_edge()
    B, N = 5, 2
    z = torch.zeros(B, N); z[:, 1] = 1.0             # post fires now in every batch element
    x_pre = torch.zeros(B, N); x_pre[:, 0] = 1.0     # pre trace on in every batch element
    y_post = torch.zeros(B, N)
    d_full, _, _ = s.edge_delta(z, z, x_pre, y_post, row, col)
    d_chunk, _, _ = s.edge_delta(z, z, x_pre, y_post, row, col, cap=1)   # cap=1 → chunk size 1
    assert abs(d_full.item() - s.a_plus * B) < 1e-5  # summed over B
    assert abs(d_full.item() - d_chunk.item()) < 1e-6


def test_width_scalable_no_dense_H2():
    """O(nnz) edge-wise at a width whose H² would be enormous (H=40000 → H²=1.6e9 floats = 6.4 GB dense).
    The kernel only ever allocates (nnz,) and (chunk, nnz) — this must run in a few MB."""
    s = SpikingSTDP()
    H = 40000
    nnz = 32 * 1000                                  # 32k sparse edges (fan-in 32 over a slice)
    torch.manual_seed(0)
    row = torch.randint(0, H, (nnz,))
    col = torch.randint(0, H, (nnz,))
    B = 4
    z = (torch.rand(B, H) < 0.05).float()
    xr = torch.rand(B, H) * 0.5
    delta, ltp, ltd = s.edge_delta(z, z, xr, xr, row, col)
    assert delta.shape == (nnz,)
    assert torch.isfinite(delta).all()


def test_live_toggle_and_string_bool_coercion():
    """on→off→on with NO restart; string 'false'/'off'/'0' disable, 'true'/'1' enable (endocrine coercion)."""
    s = SpikingSTDP()
    assert s.on is False                             # DEFAULT OFF
    assert s.set_params(on=True)["on"] is True
    assert s.on is True
    assert s.set_params(on="false")["on"] is False
    assert s.set_params(on="off")["on"] is False
    assert s.set_params(on="0")["on"] is False
    assert s.set_params(on="true")["on"] is True
    assert s.on is True
    # numeric params float-coerced; unknown keys ignored
    ap = s.set_params(a_plus="0.02", mix=0.3, tau_plus=8, bogus=1.0)
    assert ap["a_plus"] == 0.02 and isinstance(s.a_plus, float)
    assert ap["mix"] == 0.3 and ap["tau_plus"] == 8.0
    assert "bogus" not in ap


def test_state_metrics_ranges():
    s = SpikingSTDP()
    s._ltp, s._ltd, s._mag = 3.0, 1.0, 1e-4
    st = s.state()
    assert 0.0 <= st["ltp_ltd_balance"] <= 1.0
    assert -1.0 <= st["stdp_net"] <= 1.0
    assert abs(st["ltp_ltd_balance"] - 0.75) < 1e-3   # 3/(3+1)
    assert abs(st["stdp_net"] - 0.5) < 1e-3           # (3-1)/(3+1)
    assert st["stdp_mag"] == 1e-4
    assert set(st) >= {"on", "a_plus", "a_minus", "tau_plus", "tau_minus", "mix",
                       "ltp_ltd_balance", "stdp_net", "stdp_mag"}
    # empty run (no spikes) → balance defined, no NaN
    s2 = SpikingSTDP()
    st2 = s2.state()
    assert math.isfinite(st2["ltp_ltd_balance"]) and math.isfinite(st2["stdp_net"])
