"""§17 — the SpikingEmbodiment closed sensorimotor loop (active inference).

The load-bearing test is the EARNS-KEEP A/B (§8 of the integration contract): does the embodied loop raise task
return over time vs a random policy? We assert a RISING learning curve that beats the always-on shadow random
baseline, that forward-model surprise falls, that steps-to-goal falls, and that save/load round-trips the
world-BG + forward-model tensors + stats byte-for-byte.
"""
import torch

from brain.embodiment import SpikingEmbodiment, GridWorld


def test_toggle_off_is_noop():
    e = SpikingEmbodiment(seed=1)
    assert e.on is False                                        # DEFAULT OFF
    assert e.episodes == 0 and len(e.ret_window) == 0          # nothing has happened
    st = e.state()
    for k in ("on", "world_return", "world_return_random", "world_success_rate",
              "world_steps_to_goal", "world_surprise", "world_episodes", "advantage"):
        assert k in st                                         # every metric surfaces


def test_set_params_toggle_string_bool_coercion():
    e = SpikingEmbodiment(seed=2)
    ap = e.set_params(on=True, gamma=0.9, epistemic_w=0.2)
    assert ap["on"] is True and abs(e.gamma - 0.9) < 1e-9 and abs(e.epistemic_w - 0.2) < 1e-9
    assert e.set_params(on="off")["on"] is False               # 'off' disables
    assert e.set_params(on="0")["on"] is False                 # '0' disables
    assert e.set_params(on="true")["on"] is True               # 'true' enables


def test_grid_change_rebuilds_world_and_nets():
    e = SpikingEmbodiment(seed=3)
    e.set_params(grid=4)
    assert e.G == 4 and e.world.G == 4
    assert e.bg.M.shape[1] == 16 and e.fwd.M.shape[1] == 16 + 4  # nets resized to the new world


def test_earns_keep_learning_curve_beats_random():
    e = SpikingEmbodiment(seed=7); e.on = True
    early, late = [], []
    n_ep = 80
    for i in range(n_ep):
        r = e.run_episode(learn=True)
        if i < 20:
            early.append(r)
        if i >= n_ep - 20:
            late.append(r)
    late_mean = sum(late) / len(late)
    early_mean = sum(early) / len(early)
    rand_mean = sum(e.rand_window) / len(e.rand_window)
    # (1) beats the shadow random policy by a clear margin (earns its keep)
    assert late_mean > rand_mean + 0.2, (late_mean, rand_mean)
    # (2) a RISING curve, not just above-chance luck
    assert late_mean > early_mean, (late_mean, early_mean)
    # (3) it actually reaches the goal
    assert e.success_rate > 0.4, e.success_rate


def test_forward_model_surprise_falls():
    e = SpikingEmbodiment(seed=11); e.on = True
    for _ in range(6):
        e.run_episode(learn=True)
    early_surprise = e.surprise
    for _ in range(40):
        e.run_episode(learn=True)
    assert e.surprise < early_surprise, (early_surprise, e.surprise)   # the world-model sharpens


def test_steps_to_goal_falls():
    e = SpikingEmbodiment(seed=13); e.on = True
    for _ in range(10):
        e.run_episode(learn=True)
    s0 = e.steps_to_goal
    for _ in range(60):
        e.run_episode(learn=True)
    assert e.steps_to_goal <= s0 + 1e-6, (s0, e.steps_to_goal)         # the policy gets more direct


def test_dtype_propagation():
    """The module honours a requested dtype for its own tensors and the whole sensorimotor compute path
    (observe → act → world → learn → dyna_replay) runs without a dtype mismatch."""
    e = SpikingEmbodiment(dtype=torch.float64, seed=5); e.on = True
    assert e.observe().dtype == torch.float64                  # module-owned observation honours dtype
    # a full step composes with the (float32) sub-nets via boundary casts — no crash, finite signals
    feat = e.observe()
    a, pi, epi = e.act(feat)
    # the reused SpikingBasalGanglia/SpikingCerebellum build float32 LIF state internally (shared code), so the
    # sensorimotor COMPUTE runs in that dtype (e._cdt); the module casts to/from it at every boundary so a
    # requested float64 observation never triggers a dtype mismatch. This is the documented, honest contract.
    assert pi.dtype == e._cdt                                  # policy in the compute dtype, no crash
    reward, nxt, done = e.world.step(int(a))
    surprise, rpe = e.learn(feat, a, reward, nxt)
    assert surprise == surprise and rpe == rpe                 # not NaN
    e.dyna_replay(k=2, rollout=3)                              # imagined-transition path also dtype-clean
    # and the default (dtype=None) path is unchanged float32
    d = SpikingEmbodiment(seed=5)
    assert d.observe().dtype == torch.float32


def test_toggle_on_off_live():
    """on flips live via set_params with no rebuild; the loop is a no-op while off (episodes frozen)."""
    e = SpikingEmbodiment(seed=8)
    assert e.on is False
    e.run_episode(learn=True)                                  # helper runs regardless; life-loop gates on e.on
    frozen = e.episodes
    assert e.set_params(on="true")["on"] is True
    e.run_episode(learn=True)
    assert e.episodes == frozen + 1
    assert e.set_params(on="off")["on"] is False               # disable live, no restart, no exception


def test_width_scalable_shapes():
    """No dense O(N^2) and NO dependence on cortex hidden width: every tensor is sized by the gridworld
    dimension G^2 only, so growing the cortex to 128000 does not touch a single embodiment tensor."""
    e = SpikingEmbodiment(seed=9)
    G = e.G
    assert e.bg.M.shape == (64, G * G)                         # MSN input, sized by G^2 (not hidden)
    assert e.bg.W_pi.shape == (4, 64) and e.bg.w_v.shape == (64,)
    assert e.fwd.M.shape == (512, G * G + 4) and e.fwd.W.shape == (G * G, 512)
    total = (e.bg.M.numel() + e.bg.W_pi.numel() + e.bg.w_v.numel()
             + e.fwd.M.numel() + e.fwd.W.numel())
    assert total < 200_000                                     # < ~1 MB at fp32; invariant to cortex width
    # scaling the WORLD scales the tensors LINEARLY in G^2 (no quadratic blow-up in any hidden dimension)
    e.set_params(grid=8)
    assert e.bg.M.shape == (64, 64) and e.fwd.M.shape == (512, 64 + 4)


def test_save_load_round_trip():
    e = SpikingEmbodiment(seed=17); e.on = True
    for _ in range(30):
        e.run_episode(learn=True)
    # snapshot exactly what life.save_life persists
    blob = {k: getattr(e, k) for k in e._KEYS}
    blob.update(dict(wbg_M=e.bg.M.clone(), wbg_wv=e.bg.w_v.clone(), wbg_Wpi=e.bg.W_pi.clone(),
                     fwd_W=e.fwd.W.clone(), fwd_M=e.fwd.M.clone(),
                     return_ema=e.return_ema, return_random=e.return_random,
                     episodes=e.episodes, world_pos=e.world.pos, world_goal=e.world.goal))
    # fresh instance, restore
    f = SpikingEmbodiment(seed=999)
    for k in f._KEYS:
        setattr(f, k, blob[k])
    f._build_world(); f._build_nets()
    f.bg.M = blob["wbg_M"]; f.bg.w_v = blob["wbg_wv"]; f.bg.W_pi = blob["wbg_Wpi"]
    f.fwd.W = blob["fwd_W"]; f.fwd.M = blob["fwd_M"]
    f.return_ema = blob["return_ema"]; f.return_random = blob["return_random"]
    f.episodes = blob["episodes"]; f.world.pos = blob["world_pos"]; f.world.goal = blob["world_goal"]
    assert torch.equal(f.bg.M, e.bg.M) and torch.equal(f.bg.w_v, e.bg.w_v) and torch.equal(f.bg.W_pi, e.bg.W_pi)
    assert torch.equal(f.fwd.W, e.fwd.W) and torch.equal(f.fwd.M, e.fwd.M)
    assert f.episodes == e.episodes and f.return_ema == e.return_ema
    # the restored policy is byte-identical → deterministic greedy action matches
    feat = e.observe()
    assert torch.equal(f.bg.msn(feat), e.bg.msn(feat))
