"""spatial — entorhinal GRID cells + hippocampal PLACE cells + analog PATH INTEGRATION.

Covers the integration-contract obligations for a §16-pattern module:
  (a) the place code CARRIES position — ridge decode R² rises well above chance;
  (b) k-WTA sparsity — place_active_frac ≈ k_place/place_n;
  (c) loop-closure pattern-completion BOUNDS path-integration drift on blind steps;
  (d) set_params string-bool 'on' coercion + a structural place_n rebuild changes place_dim;
  (e) state_dict → load_state_dict is shape-invariant across a live place_n change;
  (f) encode_place_bytes emits a SPACE_NERVE-wrapped 0-255 proprioceptive frame;
  + device/dtype propagation (construct with float64, outputs follow), live on/off toggling is a
    no-op-safe switch, and the code widths are WIDTH-INVARIANT (no dense O(N^2), independent of cortex hidden).
"""
import torch

from brain.spatial import SpikingSpatial, GridWorld, SPACE_NERVE, SPACE_FRAME_END


def _visit_grid(sp, size):
    """Sense every cell so the metric decode buffer + landmark map fill deterministically."""
    for x in range(size):
        for y in range(size):
            sp.observe((float(x), float(y)), sensed=True)


def test_default_off_and_state_surface():
    sp = SpikingSpatial(seed=1)
    assert sp.on is False                                   # DEFAULT OFF (opt-in)
    st = sp.state()
    for k in ("on", "spatial_decode_acc", "pi_drift", "nav_return", "nav_steps_to_goal",
              "place_active_frac", "place_novelty", "grid_dim", "place_dim"):
        assert k in st                                     # every metric surfaces to /api/state


def test_decode_acc_rises_above_chance():
    sp = SpikingSpatial(seed=2)
    assert sp.decode_acc() == 0.0                          # empty buffer → chance
    _visit_grid(sp, sp.world_size)
    acc = sp.decode_acc()
    # a periodic multi-scale grid→place code linearly carries continuous position far above chance (R²≈0)
    assert acc > 0.5, acc


def test_place_code_is_sparse_kwta():
    sp = SpikingSpatial(seed=3)
    sp.place_code(4.0, 7.0)
    frac = sp._last_place_frac
    # k-WTA (de Almeida) → exactly k_place winners: active fraction ≈ k_place/place_n
    assert abs(frac - sp.k_place / sp.place_n) < 1e-6, (frac, sp.k_place / sp.place_n)


def test_loop_closure_bounds_drift():
    sp = SpikingSpatial(seed=4)
    sp.loop_thr = 0.0                                      # trust the single stored landmark
    true = (5.0, 5.0)
    # lay ONE landmark at the true position (sensed step stores place↔pos)
    sp.observe(true, sensed=True)
    # simulate accumulated path-integration drift
    sp.pos_hat = [9.0, 9.0]
    drift_before = (sp.pos_hat[0] - true[0]) ** 2 + (sp.pos_hat[1] - true[1]) ** 2
    snapped, sim = sp.snap_loop_close()
    drift_after = (sp.pos_hat[0] - true[0]) ** 2 + (sp.pos_hat[1] - true[1]) ** 2
    assert snapped is True                                 # loop closure fired
    assert drift_after < drift_before                      # attractor pattern-completion pulled the estimate home

    # and blind path integration WITHOUT loop closure leaves drift uncorrected
    sp.loop_close = False
    sp.observe(true, sensed=True); sp.pos_hat = [9.0, 9.0]
    sp.observe(true, action=1, sensed=False)              # blind step, no snap
    assert sp._pi_drift > 2.0                             # drift stays large with loop closure OFF


def test_set_params_toggle_and_structural_rebuild():
    sp = SpikingSpatial(seed=5)
    ap = sp.set_params(on=True, vel_gain=0.5)
    assert ap["on"] is True and abs(sp.vel_gain - 0.5) < 1e-9
    assert sp.set_params(on="off")["on"] is False          # 'off' disables (string-bool coercion)
    assert sp.set_params(on="0")["on"] is False            # '0' disables
    assert sp.set_params(on="true")["on"] is True          # 'true' enables
    # a structural place_n change rebuilds the banks and changes the code width
    d0 = sp.place_dim
    ap = sp.set_params(place_n=128)
    assert ap.get("_rebuilt") is True
    assert sp.place_dim == 128 and sp.place_dim != d0
    assert tuple(sp.W_gp.shape) == (128, sp.grid_dim)      # readout resized, no O(hidden) blow-up


def test_state_dict_round_trip_shape_invariant():
    sp = SpikingSpatial(seed=6)
    sp.set_params(on=True)
    _visit_grid(sp, sp.world_size)
    sp.record_episode(3.0, 12)
    sd = sp.state_dict()
    # restore into a DIFFERENTLY-shaped fresh instance → shape-invariant guard keeps it from crashing
    other = SpikingSpatial(seed=999)
    other.set_params(place_n=128)                          # different width than the snapshot
    other.load_state_dict(sd)
    assert other.on is True                                # live params restored
    assert other.place_dim == sp.place_dim                 # structural params restored → banks rebuilt to match
    assert tuple(other.W_gp.shape) == tuple(sp.W_gp.shape)
    assert torch.allclose(other.W_gp, sp.W_gp)             # matching-shape readout restored byte-exact
    assert abs(other.nav_return - sp.nav_return) < 1e-9


def test_encode_place_bytes_is_space_nerve_frame():
    sp = SpikingSpatial(seed=7)
    place = sp.place_code(2.0, 3.0)
    frame = sp.encode_place_bytes(place)
    assert frame[:len(SPACE_NERVE)] == list(SPACE_NERVE)   # 'space' nerve marker prefix
    assert frame[-len(SPACE_FRAME_END):] == list(SPACE_FRAME_END)
    body = frame[len(SPACE_NERVE):-len(SPACE_FRAME_END)]
    assert len(body) == sp.place_dim                       # one byte-level per place cell
    assert all(0 <= b <= 255 for b in frame)               # valid byte-code
    bytes(frame).decode("latin1")                          # decodes into the universal byte stream (no crash)


def test_device_dtype_propagation():
    sp = SpikingSpatial(seed=8, dtype=torch.float64)
    assert sp.dtype == torch.float64
    assert sp.W_gp.dtype == torch.float64
    assert sp._kvecs[0].dtype == torch.float64
    g = sp.grid_code(1.0, 2.0)
    p = sp.place_code(1.0, 2.0)
    assert g.dtype == torch.float64 and p.dtype == torch.float64   # outputs follow the requested dtype
    # the metric decode must run without a dtype mismatch (ones/eye/Y all follow X.dtype)
    _visit_grid(sp, sp.world_size)
    assert sp.decode_acc() > 0.5


def test_width_invariant_no_dense_state():
    # grid/place widths are set by n_modules/n_phase/place_n — NEVER by any cortex hidden width.
    a = SpikingSpatial(seed=10)
    assert a.grid_dim == a.n_modules * a.n_phase            # 5×9 = 45, cortex-width-independent
    assert a.place_dim == a.place_n
    # the ONLY matrix is W_gp = place_n × grid_dim (a few thousand floats) — no O(N^2) / O(hidden) state
    assert a.W_gp.numel() == a.place_n * a.grid_dim
    assert a.W_gp.numel() < 100_000                         # ~2.9k floats — fits in KB at hidden=128000


def test_hippocampus_interconnection():
    """The place vector is a spatial fingerprint stored/recalled through the modern-Hopfield hippocampus;
    place_novelty = 1 − recall similarity feeds the CLS novelty drive (no crash, novelty in [0,1])."""
    from brain.spiking_modules import SpikingHippocampus
    sp = SpikingSpatial(seed=11)
    hippo = SpikingHippocampus(sp.place_dim, torch.device("cpu"), seed=0)
    sp.observe((5.0, 5.0), sensed=True, hippo=hippo)       # stores the place fingerprint
    assert hippo.keys.shape[0] >= 1                        # the hippocampus received the spatial memory
    sp.pos_hat = [5.0, 5.0]
    sp.snap_loop_close(hippo=hippo)                        # recall → sets place_novelty
    assert 0.0 <= sp._place_novelty <= 1.0


def test_toggle_off_is_noop_safe():
    sp = SpikingSpatial(seed=12)
    sp.set_params(on=True); sp.observe((1.0, 1.0), sensed=True)
    sp.set_params(on=False)                                # toggling off must not corrupt state
    # code + metrics still computable after an off toggle (the loop just stops calling it)
    p = sp.place_code(1.0, 1.0)
    assert p.shape[0] == sp.place_dim
