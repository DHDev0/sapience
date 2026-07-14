"""§16 P1 — the SpikingEndocrine slow drive/cortisol/mood controller."""
from brain.endocrine import SpikingEndocrine


def test_satiation_emits_reward_and_focuses():
    e = SpikingEndocrine(); e.on = True
    for _ in range(150): e.wake_tick(progress=0.0, novelty=0.0)     # starve
    ne_starved, d_starved = e.ne_gain(), e.D_energy + e.D_novelty
    r = e.wake_tick(progress=0.5, novelty=1.0)                      # satiate
    assert d_starved > 0.5                                          # starvation builds the deficit
    assert r > 0.0                                                  # meeting a need is rewarding (homeostatic RL)
    assert e.ne_gain() < ne_starved                                # satiation lowers NE-gain → focus


def test_cortisol_gates_plasticity():
    e = SpikingEndocrine()
    g = {}
    for C in (0.0, 0.35, 1.0):
        e.C, e.AL = C, 0.0; g[C] = e.plasticity_gain()
    assert g[0.0] >= 0.95 and g[0.35] >= 0.95                       # calm→optimal = FULL plasticity (unthrottled)
    assert g[1.0] < g[0.35]                                         # only CHRONIC-high cortisol impairs (Lupien)


def test_chronic_stress_impairs_then_sleep_recovers():
    e = SpikingEndocrine(); e.on = True
    for _ in range(400): e.wake_tick(threat=0.9, surprise=0.5)      # chronic stress
    assert e.C <= e.C_max + 1e-6                                    # cortisol stays BOUNDED (no blow-up)
    assert e.AL > 0.05 and e.plasticity_gain() < 0.6               # allostatic load impairs plasticity
    g_stressed = e.plasticity_gain()
    for _ in range(400): e.sleep_tick()                            # recovery sleep
    assert e.C < 0.2 and e.plasticity_gain() > g_stressed          # sleep relieves + restores


def test_set_params_toggle_and_state():
    e = SpikingEndocrine()
    ap = e.set_params(on=True, C_star=0.4, alpha_D=0.002)
    assert ap["on"] is True and abs(e.C_star - 0.4) < 1e-9
    st = e.state()
    for k in ("cortisol", "drive_energy", "mood", "allostatic", "plasticity_gain", "ne_gain"):
        assert k in st                                             # every metric surfaces
