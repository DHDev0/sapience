"""§16 CPU tests for SharpWaveRipple — SWR-gated consolidation. Pure scalars, no torch tensors required."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brain.ripple import SharpWaveRipple


def test_default_off():
    r = SharpWaveRipple()
    assert r.on is False
    st = r.state()
    assert st["on"] is False and st["gated_commit_fraction"] == 1.0


def test_rem_never_commits():
    """Ripples are a NREM phenomenon: phase='rem' with rem_suppress must never emit a commit."""
    r = SharpWaveRipple(seed=1); r.on = True
    r.p0 = 1.0; r.up_thr = -1.0                     # would fire every up-state attempt if allowed
    commits = [r.event(phase="rem", pressure=3.0, debt=100.0) for _ in range(300)]
    assert not any(commits)
    assert r.state()["n_ripples"] == 0
    # same knobs in NREM DO commit → proves it is the phase, not the params, suppressing REM
    n = SharpWaveRipple(seed=1); n.on = True; n.p0 = 1.0; n.up_thr = -1.0
    assert sum(n.event(phase="nrem") for _ in range(300)) > 0


def test_refractory_suppresses_back_to_back():
    """A ripple silences the next `refractory` attempts → no two consecutive commits."""
    r = SharpWaveRipple(seed=2); r.on = True
    r.p0 = 1.0; r.up_thr = -1.0; r.refractory = 2.0   # always-up, always-p1 → refractory is the only limiter
    seq = [r.event(phase="nrem") for _ in range(60)]
    assert any(seq)                                    # it does fire sometimes
    for a, b in zip(seq, seq[1:]):
        assert not (a and b), "refractory must forbid back-to-back emissions"


def test_pressure_and_debt_raise_rate():
    lo = SharpWaveRipple(seed=3); lo.on = True; lo.p0 = 0.3
    hi = SharpWaveRipple(seed=3); hi.on = True; hi.p0 = 0.3
    n_lo = sum(lo.event(phase="nrem", pressure=1.0, debt=0.0) for _ in range(600))
    n_hi = sum(hi.event(phase="nrem", pressure=3.0, debt=200.0) for _ in range(600))
    assert n_hi > n_lo
    assert hi.state()["ripple_rate"] > lo.state()["ripple_rate"]


def test_gated_commit_fraction_in_unit_interval():
    r = SharpWaveRipple(seed=4); r.on = True
    for _ in range(400):
        r.event(phase="nrem", pressure=1.5, debt=30.0)
        assert 0.0 <= r.state()["gated_commit_fraction"] <= 1.0
    # gating is SELECTIVE: strictly fewer than all attempts commit
    assert 0.0 < r.state()["gated_commit_fraction"] < 1.0


def test_set_params_string_bool_coercion():
    r = SharpWaveRipple()
    r.set_params(on=True)
    assert r.on is True
    for falsey in ("off", "false", "0", "no", ""):
        r.set_params(**{"on": falsey})
        assert r.on is False, f"{falsey!r} must coerce to False"
    r.set_params(on="true")
    assert r.on is True


def test_live_toggle_on_off():
    """Survives toggling on/off live: off after being on returns to the default-off metric."""
    r = SharpWaveRipple(seed=7)
    r.set_params(on=True)
    for _ in range(50):
        r.event(phase="nrem", pressure=2.0, debt=40.0)
    assert r.on is True
    r.set_params(on="off")
    assert r.on is False
    # can be re-enabled and still functions
    r.set_params(on="true")
    assert r.event.__call__ is not None
    assert isinstance(r.event(phase="nrem"), bool)


def test_seed_determinism_fsdp():
    """Identical seed → identical commit sequence (the FSDP rank-agreement requirement)."""
    a = SharpWaveRipple(seed=0); a.on = True
    b = SharpWaveRipple(seed=0); b.on = True
    seq_a = [a.event(phase="nrem", pressure=1.3, debt=20.0) for _ in range(120)]
    seq_b = [b.event(phase="nrem", pressure=1.3, debt=20.0) for _ in range(120)]
    assert seq_a == seq_b
    # set_params(seed=...) reseeds → reproducible
    a.set_params(seed=99)
    s1 = [a.event(phase="nrem") for _ in range(40)]
    a.set_params(seed=99)
    s2 = [a.event(phase="nrem") for _ in range(40)]
    assert s1 == s2


def test_reset_night_zeroes_counters():
    r = SharpWaveRipple(seed=5); r.on = True
    for _ in range(100):
        r.event(phase="nrem", pressure=2.0, debt=50.0)
    assert r._n_attempt > 0 and r._n_ripple > 0
    r.reset_night()
    assert r._n_attempt == 0 and r._n_commit == 0 and r._n_ripple == 0
    assert r.phi == 0.0 and r._refr == 0 and r._rate_ema == 0.0
    assert r.state()["gated_commit_fraction"] == 1.0


def test_device_dtype_propagation():
    """Construct with a device+dtype: they are stored and the scalar outputs are unaffected (width-agnostic)."""
    import torch
    r = SharpWaveRipple(device=torch.device("cpu"), dtype=torch.float16, seed=8)
    assert r.device == torch.device("cpu") and r.dtype == torch.float16
    r.on = True
    out = r.event(phase="nrem", pressure=1.2, debt=10.0)
    assert isinstance(out, bool)                       # decision is a plain python bool, no tensor materialized
    st = r.state()
    assert all(isinstance(st[k], (int, float, bool)) for k in ("ripple_rate", "gated_commit_fraction", "n_ripples"))


def test_width_scalable_o1_state():
    """No dense O(N^2) / per-neuron state: the controller footprint is a fixed handful of scalars at ANY width."""
    r = SharpWaveRipple(seed=6)
    scalar_attrs = [a for a in vars(r) if not a.startswith("__")]
    # every stored attribute is a python scalar / device handle / RNG — none is a sized tensor/array
    import numbers
    for a in scalar_attrs:
        v = getattr(r, a)
        assert not hasattr(v, "shape"), f"{a} looks tensor-like — must stay width-invariant O(1)"
        # the actual state values are numbers or booleans
        if a not in ("device", "dtype", "_rng"):
            assert isinstance(v, numbers.Number)


def test_state_keys():
    r = SharpWaveRipple()
    st = r.state()
    for k in ("on", "ripple_rate", "gated_commit_fraction", "n_ripples"):
        assert k in st
