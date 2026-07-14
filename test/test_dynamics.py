"""§16 P2 — SpikingDynamics: selective ignition + attention→frequency."""
from brain.dynamics import SpikingDynamics


def test_selective_ignition_not_all_active():
    d = SpikingDynamics(); d.on = True; d.beta0 = 4.0
    sal = {"bg": 0.9, "hippo": 0.1, "cereb": 0.1}                 # one clearly salient, two not
    act = d.ignition(sal, ne=1.0, attention=1.0)
    assert act["bg"] and not act["hippo"]                         # the salient system ignites, the flat ones don't
    assert sum(act.values()) < len(act)                          # NOT all-on


def test_entropy_tracks_arousal():
    d = SpikingDynamics(); d.on = True
    low = d.entropy(ne=0.5, attention=1.0); high = d.entropy(ne=1.5, attention=1.0)
    assert high > low                                            # arousal raises β (toward the all-on regime)
    # high arousal → more systems ignite than low arousal
    sal = {"a": 0.6, "b": 0.5, "c": 0.4}
    n_low = sum(d.ignition(sal, ne=0.3, attention=1.0).values())
    n_high = sum(d.ignition(sal, ne=3.0, attention=1.0).values())
    assert n_high >= n_low


def test_attention_sets_processing_frequency():
    d = SpikingDynamics(); d.on = True
    focused = d.eligibility_beta(attention=1.3)                  # focused → gamma → SHORT window → LOW eb
    disengaged = d.eligibility_beta(attention=0.3)               # disengaged → alpha → LONG window → HIGH eb
    assert focused < disengaged                                  # focus SHORTENS the eligibility window (lower eb)
    assert d.f_gamma <= focused <= disengaged <= d.f_alpha + 1e-9


def test_set_params_and_state():
    d = SpikingDynamics()
    ap = d.set_params(on=True, beta0=3.0, ignite_thr=0.4)
    assert ap["on"] is True and abs(d.beta0 - 3.0) < 1e-9
    d.ignition({"x": 0.5, "y": 0.2}, ne=1.0, attention=1.0)
    st = d.state()
    for k in ("on", "beta", "n_active", "eff_freq"):
        assert k in st
