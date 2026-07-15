"""§PC · CPU test for PredictiveCoding — the module-level contract (correctness + device/dtype + toggle + scale).

The FULL rule (learn_pc / _pc_step) is a core edit to spiking_brain.py delivered as the wiring_patch; this test
exercises the self-contained PredictiveCoding module exactly (its ensure/precision_weight/record/state/set_params
surface), asserting: correctness of the precision math, device+dtype propagation, live on/off toggling, and
width-scalability (O(hid) only, grows/pads with the cortex — no dense O(N²))."""
import torch
from brain.predictive_coding import PredictiveCoding


class _Cell:                                   # minimal stand-in for a cortex layer (only .hid is read)
    def __init__(self, hid):
        self.hid = hid


def test_defaults_off_and_keys():
    pc = PredictiveCoding(device=torch.device("cpu"))
    assert pc.on is False                       # DEFAULT OFF (present ≠ useful; A/B decides)
    assert pc.precision is True
    assert set(pc._KEYS) == {"on", "precision", "prec_tau", "infer_gain", "pc_lr_scale", "phi", "eps"}
    st = pc.state()
    assert st["on"] is False and st["mean_precision"] == 1.0 and st["pred_err"] == []


def test_ensure_shapes_and_device_dtype():
    dev = torch.device("cpu")
    pc = PredictiveCoding(device=dev, dtype=torch.float64)
    cells = [_Cell(8), _Cell(16)]
    pc.ensure(cells)
    # per-neuron state matches each cell's width, lives on self.device, and is fp32 (statistics dtype)
    assert pc._sig2[0].shape == (8,) and pc._sig2[1].shape == (16,)
    assert pc._prec[0].shape == (8,) and pc._prec[1].shape == (16,)
    for t in pc._sig2 + pc._prec:
        assert t.device == dev and t.dtype == torch.float32
    assert len(pc._err) == 2
    # 1-D only → O(hid), never O(hid²): width-scalable to 128000
    assert all(t.dim() == 1 for t in pc._sig2 + pc._prec)


def test_precision_weight_dtype_and_mean_normalized():
    dev = torch.device("cpu")
    pc = PredictiveCoding(device=dev, dtype=torch.float32)
    pc.set_params(on=True, precision=True)
    cells = [_Cell(32)]
    pc.ensure(cells)
    torch.manual_seed(0)
    # heteroscedastic error: some units chronically noisy → precision should be non-uniform
    e = torch.randn(64, 32, dtype=torch.float32) * torch.linspace(0.1, 3.0, 32)
    for _ in range(50):                         # warm the variance EMA
        w = pc.precision_weight(0, e + 0.1 * torch.randn(64, 32))
    assert w.shape == (32,)
    assert w.device == e.device and w.dtype == e.dtype            # (a) dtype/device propagation
    # mean-normalized to ~1 (width-invariant) and noisy units DOWN-weighted vs clean units
    assert abs(float(w.mean()) - 1.0) < 1e-4
    assert float(w[0]) > float(w[-1])                             # unit 0 (σ=0.1) > unit 31 (σ=3.0)
    assert float(pc.state()["mean_precision"]) > 0.0


def test_toggle_off_is_uniform_noop():
    dev = torch.device("cpu")
    pc = PredictiveCoding(device=dev)
    pc.ensure([_Cell(10)])
    e = torch.randn(8, 10)
    pc.set_params(on=True)
    w_on = pc.precision_weight(0, e)
    assert w_on.shape == (10,)                                    # per-neuron precision when ON
    # (b) live toggle OFF → scalar 1 (uniform Π), so error is unchanged (plain local-delta PC)
    pc.set_params(on=False)
    w_off = pc.precision_weight(0, e)
    assert w_off.numel() == 1 and float(w_off) == 1.0
    assert torch.allclose((w_off * e), e)
    # sub-toggle: on but precision off → also uniform
    pc.set_params(on=True, precision=False)
    assert float(pc.precision_weight(0, e)) == 1.0


def test_string_bool_coercion():
    pc = PredictiveCoding(device=torch.device("cpu"))
    assert pc.set_params(on="true")["on"] is True
    assert pc.set_params(on="off")["on"] is False                # 'off' must DISABLE, not enable
    assert pc.set_params(on="0")["on"] is False
    assert pc.set_params(precision="no")["precision"] is False
    # numeric + string-tag params
    assert pc.set_params(infer_gain="0.3")["infer_gain"] == 0.3
    assert pc.set_params(pc_lr_scale="1500")["pc_lr_scale"] == 1500.0
    assert pc.set_params(phi="relu")["phi"] == "relu"


def test_grow_pads_precision_no_dense():
    """(c) width-scalable: a mid-life grow() widens a layer; ensure() pads sig2/prec, staying O(hid)."""
    dev = torch.device("cpu")
    pc = PredictiveCoding(device=dev)
    cells = [_Cell(8), _Cell(8)]
    pc.ensure(cells)
    pc.set_params(on=True)
    e = torch.randn(16, 8)
    pc.precision_weight(1, e); pc.record(1, e)                   # warm layer-1 stats
    old_mean = float(pc._sig2[1].mean())
    cells[1] = _Cell(24)                                          # grow layer 1: 8 → 24 neurons
    pc.ensure(cells)
    assert pc._sig2[1].shape == (24,) and pc._prec[1].shape == (24,)
    # preserved old entries; new neurons padded with the layer mean (not a 1/eps blow-up)
    assert abs(float(pc._sig2[1][:8].mean()) - old_mean) < 1e-6
    assert abs(float(pc._sig2[1][8:].mean()) - old_mean) < 1e-5
    # precision still finite + mean-normalized after growth
    w = pc.precision_weight(1, torch.randn(16, 24))
    assert w.shape == (24,) and torch.isfinite(w).all() and abs(float(w.mean()) - 1.0) < 1e-4


def test_metrics_surface():
    pc = PredictiveCoding(device=torch.device("cpu"))
    pc.ensure([_Cell(12), _Cell(12)])
    pc.set_params(on=True)
    pc.record(0, torch.randn(8, 12)); pc.record(1, torch.randn(8, 12))
    pc.record_out(torch.randn(8, 256))
    st = pc.state()
    assert len(st["pred_err"]) == 2 and all(v >= 0 for v in st["pred_err"])
    assert st["pred_err_out"] >= 0.0
    assert st["mean_precision"] > 0.0 and st["on"] is True


def test_precision_lowers_under_precision_on_heteroscedastic():
    """Precision EARNS its keep: on heteroscedastic error, precision-weighting concentrates weight on the
    reliable (low-variance) units — mean stays 1 but the weighted error energy on noisy units drops."""
    dev = torch.device("cpu")
    pc = PredictiveCoding(device=dev)
    pc.set_params(on=True, precision=True, prec_tau=20.0)
    pc.ensure([_Cell(4)])
    torch.manual_seed(1)
    scale = torch.tensor([0.1, 0.1, 5.0, 5.0])                   # 2 clean units, 2 chronically-noisy units
    for _ in range(200):
        pc.precision_weight(0, torch.randn(32, 4) * scale)
    w = pc.precision_weight(0, torch.randn(32, 4) * scale)
    assert float(w[0]) > 3 * float(w[2])                        # clean units weighted >> noisy units
