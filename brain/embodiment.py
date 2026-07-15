"""§17 · SpikingEmbodiment — the brain's FIRST closed sensorimotor loop (active inference).

Until now the brain only READS a passive byte-stream; it never ACTS with consequence. This adds a NEW life
mode running BESIDE the text-reading loop: the brain lives in a tiny gridworld and closes the loop
observation → BG actor → ACTION → world → new observation + reward → learning.

It reuses the EXACT spiking machinery already in spiking_modules.py — nothing new is invented:
  · a dedicated SpikingBasalGanglia(G*G → 4)  = the world ACTOR/CRITIC (dopamine RPE δ=r+γV′−V trains both),
    the same striatal LIF medium-spiny code that already picks curiosity topics, now picking physical actions;
  · a dedicated SpikingCerebellum(G*G+4 → G*G) = the active-inference FORWARD MODEL (Golgi-gated granule
    code + climbing-fibre delta rule) predicting the next observation from (obs, action).

Active inference / free-energy principle (Friston 2010; Friston et al. 2015 'Active inference and epistemic
value'): the policy minimises EXPECTED FREE ENERGY = pragmatic value (reach the rewarded goal) + epistemic
value (resolve uncertainty / info gain). The epistemic term = forward-model prediction error = intrinsic
curiosity (Pathak et al. 2017 ICM; Schmidhuber compression progress), so the agent explores surprising states
early and becomes goal-directed as its world-model sharpens.

INTERCONNECTION (not a bolt-on RL box): the forward-model surprise is the SAME surprise the endocrine layer
consumes; the world reward drives the SAME dopamine tone the cortex three-factor gate reads; obs+action+reward
are byte-encoded as a new 'world' nerve so the cortex PERCEIVES its own embodiment in the one unified byte-code
alongside text/vision/time. Model-based sleep replay of imagined transitions is Dyna (Sutton 1991) / hippocampal
replay, wired into the existing generative-replay path.

DEFAULT OFF — opt-in via set_net('embodiment', {'on': true}); the A/B (learned policy vs the shadow random
policy the loop runs every episode) must show it EARNS ITS KEEP before it is trusted on.
"""
from __future__ import annotations
import random
from collections import deque

import torch

from .spiking_modules import SpikingBasalGanglia, SpikingCerebellum


class GridWorld:
    """A tiny tabular gridworld: G×G cells, start at (0,0), goal at the far corner, an optional hazard in the
    centre, 4 actions (up/down/left/right, wall-clamped). Observation = one-hot cell (G*G). Reward = a small
    step cost + potential-based shaping toward the goal (dense, learnable in seconds on CPU) + a goal bonus and
    a hazard penalty. Deterministic start so steps-to-goal is a clean learning-curve metric."""

    def __init__(self, G=5, seed=0, step_cost=0.01, shape_w=0.1, goal_reward=1.0,
                 hazard=True, max_steps=40, device=None, dtype=None):
        self.G = int(G); self.device = device; self.dtype = dtype
        self.step_cost = float(step_cost); self.shape_w = float(shape_w)
        self.goal_reward = float(goal_reward); self.max_steps = int(max_steps)
        self.rng = random.Random(seed)
        self.goal = (self.G - 1, self.G - 1)
        self.hazard = (self.G // 2, self.G // 2) if hazard else None
        self.pos = (0, 0); self.t = 0
        self.reset()

    def reset(self):
        self.pos = (0, 0); self.t = 0
        return self

    def _dist(self, p):
        return abs(p[0] - self.goal[0]) + abs(p[1] - self.goal[1])

    @property
    def n_states(self):
        return self.G * self.G

    def observe(self):
        idx = self.pos[0] * self.G + self.pos[1]
        f = torch.zeros(1, self.G * self.G, device=self.device, dtype=self.dtype)
        f[0, idx] = 1.0
        return f

    def step(self, action):
        """Apply an action, return (reward, next_feat, done)."""
        r, c = self.pos
        d0 = self._dist(self.pos)
        a = int(action)
        if a == 0:   r = max(0, r - 1)
        elif a == 1: r = min(self.G - 1, r + 1)
        elif a == 2: c = max(0, c - 1)
        elif a == 3: c = min(self.G - 1, c + 1)
        self.pos = (r, c); self.t += 1
        d1 = self._dist(self.pos)
        reward = -self.step_cost + self.shape_w * (d0 - d1)      # pragmatic gradient toward the goal
        done = False
        if self.pos == self.goal:
            reward += self.goal_reward; done = True
        elif self.hazard is not None and self.pos == self.hazard:
            reward -= 0.5                                        # a state to avoid
        if self.t >= self.max_steps:
            done = True
        return reward, self.observe(), done


class SpikingEmbodiment:
    """§17 embodied active-inference loop — a self-contained controller (endocrine.py pattern): _KEYS,
    set_params (string-bool 'on' coercion), state(), device/dtype-agnostic, DEFAULT OFF."""

    _KEYS = ("on", "grid", "max_steps", "gamma", "explore_temp", "epistemic_w", "step_cost", "cadence")

    def __init__(self, device=None, dtype=None, seed=0):
        self.device = device
        self.dtype = dtype                           # module-owned tensors honour this dtype (observations, buffers)
        self.on = False                              # opt-in toggle (verify it earns its keep before defaulting on)
        self._seed = int(seed)
        # tunable knobs
        self.grid = 5                                # G×G tabular world
        self.max_steps = 40                          # episode horizon
        self.gamma = 0.95                            # TD discount
        self.explore_temp = 1.0                      # softmax temperature base (scaled by NE + sleep-pressure)
        self.epistemic_w = 0.3                       # weight on the curiosity / info-gain bonus (capped, anneal-able)
        self.step_cost = 0.01                        # per-step cost in the world
        self.cadence = 0.5                           # min seconds between embodied steps (throttle; never starves thought)
        self.fwd_eta = 0.3                           # forward-model delta-rule rate
        self.G = self.grid
        # the reused SpikingBasalGanglia/SpikingCerebellum build float32 LIF state internally (shared code); we
        # therefore run the sensorimotor COMPUTE in their weight dtype (self._cdt, set in _build_nets) and cast
        # module-owned tensors to/from it at every boundary, so self.dtype is honoured for observations while
        # nothing ever hits a dtype mismatch.
        self._build_world(); self._build_nets()
        # learning-curve statistics
        self.episodes = 0
        self.return_ema = 0.0; self.return_random = 0.0
        self.success_rate = 0.0; self.steps_to_goal = float(self.max_steps)
        self.surprise = 0.0
        self.ret_window = deque(maxlen=100)          # learned-policy episode returns
        self.rand_window = deque(maxlen=100)         # shadow random-policy returns (the live A/B baseline)
        self._shadow_rng = random.Random(self._seed + 777)

    # ---- construction ---------------------------------------------------- #
    def _build_world(self):
        self.G = int(self.grid)
        self.world = GridWorld(self.G, seed=self._seed, step_cost=self.step_cost,
                               max_steps=int(self.max_steps), device=self.device, dtype=self.dtype)

    def _build_nets(self):
        # the world ACTOR/CRITIC — a real SpikingBasalGanglia (SEPARATE from life.self.bg topic policy)
        self.bg = SpikingBasalGanglia(self.G * self.G, 4, self.device,
                                      alpha_v=0.1, alpha_pi=0.2, seed=self._seed)
        # the active-inference FORWARD MODEL — a real SpikingCerebellum (SEPARATE from life.self.cerebellum)
        self.fwd = SpikingCerebellum(self.G * self.G + 4, self.G * self.G, self.device,
                                     n_granule=512, seed=self._seed)
        self._cdt = self.bg.M.dtype                  # the sensorimotor compute dtype (sub-net weight dtype)

    # ---- the sensorimotor primitives ------------------------------------- #
    def observe(self):
        return self.world.observe()

    def _epistemic(self, feat):
        """Per-action epistemic bonus: the magnitude of the forward-model's PREDICTED transition. An untrained
        model (W≈0) predicts ≈0 → a uniform bonus (harmless); as it sharpens it distinguishes state-changing
        actions from wall-bumps, biasing exploration toward informative moves. Shape (B,4)."""
        B = feat.shape[0]; fd = self.fwd.M.dtype
        fc = feat.to(self.device, dtype=self._cdt)                # feat in compute dtype (for the norm)
        ff = feat.to(self.fwd.device, dtype=fd)                   # feat in the forward-model's dtype
        bonus = torch.zeros(B, 4, device=self.device, dtype=self._cdt)
        for a in range(4):
            oh = torch.zeros(B, 4, device=self.fwd.device, dtype=fd); oh[:, a] = 1.0
            inp = torch.cat([ff, oh], 1)
            pred = self.fwd.predict(inp).to(self.device, dtype=self._cdt)
            bonus[:, a] = (pred - fc).norm(dim=1)
        return bonus

    @torch.no_grad()
    def act(self, feat, ne=1.0, sleep_pressure=0.0):
        """BG softmax policy at temperature = explore_temp·NE·(1+sleep_pressure), plus the epistemic bonus.
        Aroused/tired (high NE / high sleep-pressure) → hotter → more exploratory. Returns (action, pi, bonus)."""
        r = self.bg.msn(feat.to(self.bg.device, dtype=self.bg.M.dtype))
        logits = (r @ self.bg.W_pi.t()).to(self.device, dtype=self._cdt)   # (B,4)
        temp = max(1e-3, float(self.explore_temp) * float(ne) * (1.0 + max(0.0, float(sleep_pressure))))
        epi = self._epistemic(feat)                              # (B,4) in self._cdt on self.device
        logits = logits / temp + float(self.epistemic_w) * epi
        pi = torch.softmax(logits.clamp(-30, 30), 1)
        a = torch.multinomial(pi, 1).squeeze(1)
        epi_bonus = float(epi.gather(1, a.view(-1, 1)).mean())
        return a, pi, epi_bonus

    @torch.no_grad()
    def learn(self, feat, action, reward, next_feat, r_home=0.0):
        """Close the loop: (1) forward-model surprise = prediction error on the ACTUAL transition (measured
        BEFORE the update — the epistemic/curiosity signal + the endocrine surprise); (2) train the forward
        model toward next_feat; (3) train the world-BG on total = pragmatic (world reward + met-need r_home) +
        epistemic (capped curiosity bonus). Returns (surprise, rpe)."""
        if not torch.is_tensor(action):
            action = torch.as_tensor([int(action)], device=self.device)
        B = feat.shape[0]; fd = self.fwd.M.dtype; bd = self.bg.M.dtype
        nfc = next_feat.to(self.device, dtype=self._cdt)         # next-obs in compute dtype (for surprise)
        ff = feat.to(self.fwd.device, dtype=fd)
        oh = torch.zeros(B, 4, device=self.fwd.device, dtype=fd)
        oh[torch.arange(B, device=self.fwd.device), action.to(self.fwd.device).long()] = 1.0
        inp = torch.cat([ff, oh], 1)
        pred = self.fwd.predict(inp).to(self.device, dtype=self._cdt)
        surprise = float(((pred - nfc) ** 2).mean())
        self.fwd.train_step(inp, next_feat.to(self.fwd.device, dtype=fd), eta=self.fwd_eta)
        epi = min(surprise, 1.0)                                  # bounded epistemic reward (guards reward-hacking)
        total = float(reward) + float(r_home) + float(self.epistemic_w) * epi
        rpe = self.bg.train_step(feat.to(self.bg.device, dtype=bd), action.to(self.bg.device),
                                 torch.tensor([total], device=self.bg.device, dtype=bd),
                                 phi_next=next_feat.to(self.bg.device, dtype=bd), gamma=float(self.gamma))
        self.surprise = 0.99 * self.surprise + 0.01 * surprise
        return surprise, float(rpe)

    # ---- episode drivers (used by the test + the shadow A/B) ------------- #
    def _finish_episode(self, ret, steps, reached):
        self.episodes += 1
        self.ret_window.append(float(ret))
        self.return_ema = float(ret) if self.episodes == 1 else 0.9 * self.return_ema + 0.1 * float(ret)
        self.success_rate = 0.95 * self.success_rate + 0.05 * (1.0 if reached else 0.0)
        if reached:
            self.steps_to_goal = 0.9 * self.steps_to_goal + 0.1 * float(steps)
        rr = self._random_episode_return()                       # the always-on shadow baseline (self-proving A/B)
        self.rand_window.append(rr)
        self.return_random = rr if self.episodes == 1 else 0.9 * self.return_random + 0.1 * rr

    def _random_episode_return(self):
        """Run a uniform-random policy over an identical fresh world — the live A/B baseline."""
        w = GridWorld(self.G, seed=self._shadow_rng.randint(0, 1 << 30), step_cost=self.step_cost,
                      max_steps=int(self.max_steps), device=self.device, dtype=self.dtype)
        ret = 0.0
        for _ in range(int(self.max_steps)):
            reward, _, done = w.step(self._shadow_rng.randint(0, 3))
            ret += reward
            if done:
                break
        return ret

    def run_episode(self, learn=True, ne=1.0, sleep_pressure=0.0):
        """One full episode with the BG active-inference policy; updates the learning-curve stats. Returns the
        episode return. (The life loop instead drives step-wise via observe/act/world.step/learn so world steps
        interleave with thought; this helper is the self-contained A/B path.)"""
        self.world.reset()
        ret = 0.0; steps = 0; reached = False
        for _ in range(int(self.max_steps)):
            feat = self.world.observe()
            a, pi, epi = self.act(feat, ne=ne, sleep_pressure=sleep_pressure)
            reward, next_feat, done = self.world.step(int(a))
            if learn:
                self.learn(feat, a, reward, next_feat)
            ret += reward; steps += 1
            if done:
                reached = (self.world.pos == self.world.goal)
                break
        self._finish_episode(ret, steps, reached)
        return ret

    @torch.no_grad()
    def dyna_replay(self, k=8, rollout=6):
        """Model-based (Dyna, Sutton 1991) sleep replay: roll the forward model out from random states and
        consolidate the world-BG on the IMAGINED transitions — buffer-free, mirroring the cortex's generative
        replay. Interconnects sleep + the forward model. Returns the number of imagined transitions replayed."""
        n = 0; fd = self.fwd.M.dtype; bd = self.bg.M.dtype
        for _ in range(int(k)):
            idx = random.randrange(self.G * self.G)
            feat = torch.zeros(1, self.G * self.G, device=self.device, dtype=self._cdt); feat[0, idx] = 1.0
            for _ in range(int(rollout)):
                a, pi, epi = self.act(feat)
                oh = torch.zeros(1, 4, device=self.fwd.device, dtype=fd); oh[0, int(a)] = 1.0
                inp = torch.cat([feat.to(self.fwd.device, dtype=fd), oh], 1)
                nxt = self.fwd.predict(inp).to(self.device, dtype=self._cdt)
                nxt = torch.softmax(nxt.clamp(-30, 30) * 4.0, 1)        # sharpen the imagined next-state
                # imagined pragmatic reward from the (differentiable-free) model: shaping toward the goal cell
                gr = self.G * self.G - 1
                r_img = float(nxt[0, gr]) - 0.01
                self.bg.train_step(feat.to(self.bg.device, dtype=bd), torch.as_tensor([int(a)], device=self.bg.device),
                                   torch.tensor([r_img], device=self.bg.device, dtype=bd),
                                   phi_next=nxt.to(self.bg.device, dtype=bd), gamma=float(self.gamma))
                feat = nxt; n += 1
        return n

    # ---- controller plumbing (endocrine.py pattern) --------------------- #
    def set_params(self, **kw):
        applied = {}
        structural = False
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif k in ("grid", "max_steps"):
                v = max(2, int(float(v))) if k == "grid" else max(2, int(float(v)))
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
            if k in ("grid", "max_steps", "step_cost"):
                structural = True
        if structural:                                         # world dims changed → rebuild (grid also rebuilds nets)
            rebuild_nets = ("grid" in applied)
            self._build_world()
            if rebuild_nets:
                self._build_nets()
        return applied

    def state(self):
        return dict(on=self.on,
                    world_return=round(self.return_ema, 3),
                    world_return_random=round(self.return_random, 3),
                    world_success_rate=round(self.success_rate, 3),
                    world_steps_to_goal=round(self.steps_to_goal, 2),
                    world_surprise=round(self.surprise, 5),
                    world_episodes=self.episodes,
                    advantage=round(self.return_ema - self.return_random, 3))
