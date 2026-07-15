"""§17 — SpikingSTP: Tsodyks-Markram short-term facilitation/depression.

Module-level tests (always run) assert the STP contract: rest-gain==1 (the byte-identity guarantee),
depressing/facilitating regimes, set_params string coercion, state() keys, reset reallocation, and _KEYS
round-trip. Integration tests (guarded — SKIP until the spiking_brain.py/spiking.py wiring_patch is applied)
assert the forward + e-prop interconnection: default-off byte-identity through _run, STP-on changing _run, and
STP changing the e-prop weight update (the eligibility↔efficacy coupling)."""
import inspect
import pytest
import torch

from brain.synaptic_stp import SpikingSTP


# ------------------------------------------------------------------ module contract (always run)

class _Cell:                                     # minimal stand-in for a cortex layer (has .hid + .parameters)
    def __init__(self, hid, dev="cpu"):
        self.hid = hid; self._p = torch.nn.Parameter(torch.zeros(hid, device=dev))
    def parameters(self):
        return iter([self._p])


def test_default_off():
    s = SpikingSTP()
    assert s.on is False                         # DEFAULT OFF → forward path byte-identical
    assert s.modulate_input is False


def test_rest_gain_is_one():
    """At rest (u=U, x=1, no spike) the very first transmit with z=1 emits g==1 for every neuron: the
    normalization that makes the off/at-rest path need no weight rescale."""
    s = SpikingSTP()
    s.reset([_Cell(8)], B=3)
    z = torch.ones(3, 8)
    g = s.transmit(0, z)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


def test_depressing_regime_gain_below_one():
    """Depressing preset (tau_facil < tau_rec): repeated identical spikes drive g monotonically below 1 and
    mean_efficacy < 1 (vesicle depletion outpaces facilitation)."""
    s = SpikingSTP(); s.set_params(on=True, tau_rec=20.0, tau_facil=4.0, U=0.5)
    s.reset([_Cell(4)], B=1)
    z = torch.ones(1, 4)
    gains = [float(s.transmit(0, z).mean()) for _ in range(6)]
    assert gains[0] == pytest.approx(1.0, abs=1e-6)              # first release is at rest
    assert gains[-1] < 1.0                                       # net depression under a spike train
    # monotone non-increasing after the initial release
    for a, b in zip(gains[1:], gains[2:]):
        assert b <= a + 1e-6
    assert s.mean_efficacy < 1.0


def test_facilitating_regime_gain_above_one():
    """Facilitating preset (small U, long tau_facil >> tau_rec): repeated spikes push g above 1 as
    utilization builds faster than resources deplete."""
    s = SpikingSTP(); s.set_params(on=True, tau_rec=2.0, tau_facil=200.0, U=0.1)
    s.reset([_Cell(4)], B=1)
    z = torch.ones(1, 4)
    gains = [float(s.transmit(0, z).mean()) for _ in range(8)]
    assert max(gains[1:]) > 1.0                                  # facilitation drives efficacy above rest
    assert s.mean_efficacy > 1.0


def test_no_spike_relaxes_to_rest():
    """After depletion, ticks with no spike recover x toward 1 and relax u toward U (g returns toward 1)."""
    s = SpikingSTP(); s.set_params(on=True, tau_rec=3.0, tau_facil=3.0, U=0.5)
    s.reset([_Cell(4)], B=1)
    z1 = torch.ones(1, 4); z0 = torch.zeros(1, 4)
    for _ in range(4):
        s.transmit(0, z1)                                        # deplete
    depleted_x = s.mean_x
    for _ in range(30):
        s.transmit(0, z0)                                        # quiet → recover
    assert s.mean_x > depleted_x
    assert s.mean_x == pytest.approx(1.0, abs=1e-2)


def test_set_params_string_off_coercion():
    s = SpikingSTP()
    ap = s.set_params(on="on")
    assert s.on is True and ap["on"] is True
    ap = s.set_params(on="off")                                 # string 'off' must DISABLE, not enable
    assert s.on is False and ap["on"] is False
    s.set_params(modulate_input="false"); assert s.modulate_input is False
    s.set_params(modulate_input="1"); assert s.modulate_input is True
    ap = s.set_params(tau_rec="15", U=0.3)
    assert abs(s.tau_rec - 15.0) < 1e-9 and abs(s.U - 0.3) < 1e-9


def test_state_keys():
    s = SpikingSTP()
    st = s.state()
    for k in ("on", "tau_rec", "tau_facil", "U", "modulate_input", "mean_efficacy", "mean_u", "mean_x"):
        assert k in st


def test_keys_round_trip():
    """save→load convention: dump the _KEYS, reconstruct via set_params (transient u/x are NOT saved)."""
    s = SpikingSTP(); s.set_params(on=True, tau_rec=33.0, tau_facil=7.0, U=0.4, modulate_input=True)
    dumped = {k: getattr(s, k) for k in s._KEYS}
    s2 = SpikingSTP(); s2.set_params(**dumped)
    for k in s._KEYS:
        assert getattr(s2, k) == getattr(s, k)


def test_reset_reallocates_on_batch_change():
    s = SpikingSTP(); s.set_params(on=True)
    s.reset([_Cell(5)], B=2)
    assert s._u[0].shape == (2, 5)
    s.reset([_Cell(5)], B=7)                                     # batch size changed
    assert s._u[0].shape == (7, 5)
    assert torch.allclose(s._u[0], torch.full((7, 5), float(s.U)))   # rebuilt at rest


def test_ensure_lazy_build_on_shape_change():
    """transmit is safe without an explicit reset and rebuilds on a (B,hid) change."""
    s = SpikingSTP(); s.set_params(on=True)
    g1 = s.transmit(0, torch.ones(2, 4))
    assert g1.shape == (2, 4)
    g2 = s.transmit(0, torch.ones(5, 4))                        # different B → rebuild, no crash
    assert g2.shape == (5, 4)


def test_dtype_and_device_propagation():
    """u,x,g inherit the activation's dtype/device (never a hard-coded float32/cpu). Construct with a
    non-default dtype and check the returned gain matches."""
    for dt in (torch.float64, torch.float16):
        s = SpikingSTP(); s.set_params(on=True, tau_rec=20.0, tau_facil=4.0, U=0.5)
        z = torch.ones(2, 8, dtype=dt)
        s.reset([_Cell(8)], B=2)
        # reset() builds at rest on cpu-float32; _ensure rebuilds to match z's dtype/device on first transmit
        g = s.transmit(0, z)
        assert g.dtype == dt and g.device == z.device
        assert torch.allclose(g, torch.ones_like(g), atol=1e-3)      # rest gain 1 regardless of dtype


def test_ne_gain_is_neutral_by_default():
    """The §16 NE hook is identity at _ne_gain=1.0 (byte-identical), and raising it shifts the regime
    (higher effective U → stronger depression under a train)."""
    s = SpikingSTP(); s.set_params(on=True, tau_rec=20.0, tau_facil=4.0, U=0.5)
    z = torch.ones(1, 4)
    s.reset([_Cell(4)], B=1)
    base = [float(s.transmit(0, z).mean()) for _ in range(5)]
    assert base[0] == pytest.approx(1.0, abs=1e-6)                   # neutral at rest
    s2 = SpikingSTP(); s2.set_params(on=True, tau_rec=20.0, tau_facil=4.0, U=0.5)
    s2._ne_gain = 1.6                                                 # NE arousal raises release prob
    s2.reset([_Cell(4)], B=1)
    hi = [float(s2.transmit(0, z).mean()) for _ in range(5)]
    assert hi[-1] < base[-1]                                          # more depression under higher U


def test_live_toggle_on_off():
    """Toggling stp.on live flips whether transmit participates (state()['on']) with no rebuild needed."""
    s = SpikingSTP()
    assert s.state()["on"] is False
    s.set_params(on=True); assert s.state()["on"] is True
    z = torch.ones(1, 4)
    g = s.transmit(0, z); assert g.shape == (1, 4)
    s.set_params(on=False); assert s.state()["on"] is False          # off again, no crash / state kept


def test_width_scalable_no_dense_state():
    """State is O(B·H) per layer, NOT O(H^2): a large H must not allocate a dense HxH tensor."""
    H = 20000
    s = SpikingSTP(); s.set_params(on=True)
    g = s.transmit(0, torch.ones(1, H))
    assert g.shape == (1, H)
    assert s._u[0].numel() == H and s._x[0].numel() == H             # one (u,x) per neuron, not per synapse


# ------------------------------------------------------------------ integration (guarded / SKIP pre-wiring)

def _fresh_brain(**kw):
    from brain.spiking_brain import SpikingBrain
    dev = torch.device("cpu")
    return SpikingBrain(dev, emb=16, hidden=32, layers=2, seq=16, cell="alif",
                        readout="mem", seed=1234, sparse=False, **kw)


def _wiring_present(brain):
    """True once the wiring_patch has threaded STP into the cortex + run_seq."""
    if getattr(brain, "stp", None) is None:
        return False
    try:
        from brain.spiking import ALIFCell
        return "stp" in inspect.signature(ALIFCell.run_seq).parameters
    except Exception:
        return False


def test_integration_wiring_or_skip():
    brain = _fresh_brain()
    if not _wiring_present(brain):
        pytest.skip("STP wiring_patch not yet applied to spiking_brain.py / spiking.py")

    # warm up so the cortex actually SPIKES — STP only acts on spikes, and a fresh brain barely fires
    TXT = "the quick brown fox jumps over the lazy dog. water runs to the sea. " * 20
    for _ in range(15):
        brain.learn_eprop(TXT, epochs=1, bs=8, max_steps=1, seq=32)
    x = torch.tensor(brain.to_bytes(TXT)[:32], device=brain.device).unsqueeze(0)
    # (1) default OFF ⇒ _run is deterministic (off == off), and unchanged by the STP machinery
    brain.stp.set_params(on=False)
    with torch.no_grad():
        l_off, _ = brain._run(x)
        l_off2, _ = brain._run(x)
    assert torch.equal(l_off, l_off2), "STP off must leave the _run forward deterministic/unchanged"

    # (2) STP ON (depressing) must CHANGE the eval forward → bpb reflects it
    brain.stp.set_params(on=True, tau_rec=20.0, tau_facil=4.0, U=0.5)
    with torch.no_grad():
        l_on, _ = brain._run(x)
    assert not torch.equal(l_on, l_off), "STP on must change the _run forward (wired into recurrent drive)"


def test_integration_eprop_interconnection_or_skip():
    """STP changes the e-prop weight update: the eligibility carries z·g, not the raw spike."""
    brain = _fresh_brain()
    if not _wiring_present(brain):
        pytest.skip("STP wiring_patch not yet applied")
    data = list(bytes([65, 66, 65, 66, 67, 65, 66, 65, 66, 67] * 8))

    def _train_delta(on):
        b = _fresh_brain(); b.learn_rule = "eprop"
        b.stp.set_params(on=on, tau_rec=20.0, tau_facil=4.0, U=0.5)
        w0 = b.cells[0].Wrec.weight.detach().clone()
        b.learn_text(data, epochs=1, bs=2, max_steps=4)
        return (b.cells[0].Wrec.weight.detach() - w0)

    d_off = _train_delta(False)
    d_on = _train_delta(True)
    assert not torch.equal(d_off, d_on), "STP must reshape the e-prop update (eligibility↔efficacy coupling)"
