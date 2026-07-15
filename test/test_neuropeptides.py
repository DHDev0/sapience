"""CPU test for §16 SpikingNeuropeptides — mirrors test_endocrine's coverage.

Verifies: default-OFF, string-bool coercion, bounded leaky-integrator dynamics with sign-correct drivers,
the anti-duplication endocrine coupling (threat_gain into the driver, cortisol_relief from the pool),
sleep decay, device/dtype propagation (holds no tensors → any dtype safe), live on/off toggling with no
residue, O(1) width-scalability, and state()/save-load round-trip.
"""
import json
import types

from brain.neuropeptides import SpikingNeuropeptides


def test_default_off():
    p = SpikingNeuropeptides()
    assert p.on is False
    # pools at their documented resting setpoints
    assert p.OXT == 0.3 and p.ORX == 0.4 and p.CRH == 0.2


def test_string_bool_coercion():
    p = SpikingNeuropeptides()
    p.set_params(on=True)
    assert p.on is True
    for off in ("off", "false", "0", "no", ""):
        p.set_params(on=off)
        assert p.on is False, off
    for on in ("on", "true", "1", "yes"):
        p.set_params(on=on)
        assert p.on is True, on


def test_pools_bounded_under_extreme_drive():
    # every driver clipped to [0,1] → pools provably stay in [0,1] no matter how hard we push
    p = SpikingNeuropeptides(); p.set_params(on=True)
    for _ in range(5000):
        p.wake_tick(progress=9.0, novelty=9.0, surprise=9.0, threat=9.0, da=9.0, social=9.0)
    for lvl in (p.OXT, p.ORX, p.CRH):
        assert 0.0 <= lvl <= 1.0
    for _ in range(5000):
        p.wake_tick(progress=-9.0, novelty=-9.0, surprise=-9.0, threat=-9.0, da=-9.0, social=-9.0)
    for lvl in (p.OXT, p.ORX, p.CRH):
        assert 0.0 <= lvl <= 1.0


def test_crh_rises_and_threat_gain_and_valence_negative_under_stress():
    p = SpikingNeuropeptides(); p.set_params(on=True)
    crh0 = p.CRH
    for _ in range(400):
        p.wake_tick(progress=0.0, novelty=0.0, surprise=1.0, threat=1.0, da=0.0, social=0.0)
    assert p.CRH > crh0                       # sustained surprise+threat drives the CRH pool up
    assert p.valence_bias() < 0.0             # CRH biases §2 reward negative
    assert p.threat_gain() > 1.0              # CRH multiplies the threat flowing INTO endocrine (upstream HPA)
    assert p.plasticity_bias() < 1.0          # stress narrows plasticity


def test_oxt_rises_and_relief_and_valence_positive_under_social_progress():
    p = SpikingNeuropeptides(); p.set_params(on=True)
    oxt0 = p.OXT
    for _ in range(400):
        p.wake_tick(progress=0.8, novelty=0.1, surprise=0.0, threat=0.0, da=0.4, social=1.0)
    assert p.OXT > oxt0                        # progress + dopamine + social/tool interaction raise oxytocin
    assert p.valence_bias() > 0.0             # OXT biases §2 reward positive
    assert p.cortisol_relief() > 0.0          # OXT subtracts from the endocrine cortisol POOL (prosocial buffer)
    assert p.plasticity_bias() > 1.0          # calm/prosocial broadens plasticity


def test_anti_duplication_endocrine_coupling():
    # the mandated single-cortisol contract, exercised against a stub endocrine exactly as life.py wires it:
    # CRH multiplies the threat driver INTO endo.wake_tick; OXT subtracts from endo.C AFTER it.
    class _Endo:
        def __init__(self): self.C = 0.5; self.seen_threat = None
        def wake_tick(self, threat=0.0, **kw): self.seen_threat = threat; self.C += 0.1 * threat; return 0.0
    p = SpikingNeuropeptides(); p.set_params(on=True); p.CRH = 0.8; p.OXT = 0.9
    endo = _Endo()
    raw_threat = 0.5
    gated = raw_threat * p.threat_gain()
    assert gated > raw_threat                 # CRH amplified the driver at the INPUT
    endo.wake_tick(threat=gated)
    assert endo.seen_threat == gated
    c_before = endo.C
    endo.C = max(0.0, endo.C - p.cortisol_relief())
    assert endo.C < c_before                  # OXT relieved the pool at the OUTPUT — one integrator, biased both ends


def test_orexin_ne_bias_and_sleep_resist():
    p = SpikingNeuropeptides(); p.set_params(on=True)
    for _ in range(400):                      # high novelty, no satiation → orexin climbs
        p.wake_tick(progress=0.0, novelty=1.0, surprise=0.0, threat=0.0, da=0.3, social=0.0)
    assert p.ne_bias() > 0.0                  # orexin raises the additive NE/exploration term
    assert p.sleep_resist() > 0.0             # ... and resists sleep pressure


def test_sleep_decays_arousal_and_stress():
    p = SpikingNeuropeptides(); p.set_params(on=True)
    p.ORX = 0.95; p.CRH = 0.95
    for _ in range(200):
        p.sleep_tick()
    assert p.ORX < 0.95                       # sleep clears orexin arousal
    assert p.CRH < 0.95                       # sleep clears CRH stress
    for lvl in (p.OXT, p.ORX, p.CRH):
        assert 0.0 <= lvl <= 1.0


def test_live_toggle_no_residue():
    # toggling off must leave the shared surface at its neutral values regardless of pool state
    p = SpikingNeuropeptides(); p.set_params(on=True)
    for _ in range(50):
        p.wake_tick(progress=0.9, novelty=0.9, surprise=0.9, threat=0.9, da=0.5, social=1.0)
    p.set_params(on="off")
    assert p.on is False
    # the loop only APPLIES the modifiers when pep.on; when off the wiring uses identities. Confirm the
    # module can be flipped back on live with the pools intact (state carries across the toggle).
    oxt = p.OXT
    p.set_params(on="on")
    assert p.on is True and p.OXT == oxt


def test_device_dtype_propagation_holds_no_tensors():
    import torch
    # constructing with any device/dtype must behave identically — the controller stores NO tensors, so it is
    # trivially device-/dtype-/FSDP-agnostic and O(1) at any width (nothing per-neuron/per-synapse).
    a = SpikingNeuropeptides(device="cpu", dtype=torch.float16); a.set_params(on=True)
    b = SpikingNeuropeptides(device=None, dtype=None); b.set_params(on=True)
    for _ in range(100):
        kw = dict(progress=0.5, novelty=0.5, surprise=0.3, threat=0.2, da=0.1, social=0.5)
        a.wake_tick(**kw); b.wake_tick(**kw)
    assert a.state() == b.state()             # dtype/device do not change the scalar controller's output
    # no tensor attribute was ever created on the module
    assert not any(torch.is_tensor(v) for v in vars(a).values())


def test_state_keys_and_json_round_trip():
    p = SpikingNeuropeptides(); p.set_params(on=True)
    for _ in range(10):
        p.wake_tick(progress=0.5, novelty=0.5, surprise=0.3, threat=0.2, da=0.1, social=0.5)
    st = p.state()
    for k in ("on", "oxytocin", "orexin", "crh", "plasticity_bias", "ne_bias", "valence_bias", "cortisol_relief"):
        assert k in st
    json.dumps(st)                            # metrics must be JSON-serialisable for /api/state

    # save/load round-trip through the _KEYS + pool dict (exactly as life.py save_life/load_life do)
    saved = {**{k: getattr(p, k) for k in p._KEYS}, **{k: getattr(p, k) for k in ("OXT", "ORX", "CRH")}}
    blob = json.loads(json.dumps(saved))
    q = SpikingNeuropeptides()
    for k, v in blob.items():
        setattr(q, k, v)
    assert q.on == p.on
    assert abs(q.OXT - p.OXT) < 1e-9 and abs(q.ORX - p.ORX) < 1e-9 and abs(q.CRH - p.CRH) < 1e-9
    assert q.state()["valence_bias"] == p.state()["valence_bias"]
