"""spatial.py — entorhinal GRID cells + hippocampal PLACE cells for an embodied agent.

The first real EMBODIMENT paradigm in the brain: a small deterministic GridWorld body, a bank of
entorhinal grid modules that tile 2D space with a hexagonal periodic code (Hafting-Fyhn-Moser 2005;
Stensola 2012), a hippocampal place layer built by a fixed sparse grid→place readout + k-WTA
(Solstad-Moser-Einevoll 2006; de Almeida k-WTA 2009), and ANALOG PATH INTEGRATION — the emitted
movement action (velocity) advances an internal position estimate that drives the grid phase, whose
drift is bounded by loop-closure recall of stored landmarks (Burak-Fiete 2009; McNaughton 2006).

How it TALKS to the rest of the brain (interconnection, not a bolt-on):
  · the place-cell vector is a SPATIAL FINGERPRINT stored/recalled through the existing modern-Hopfield
    SpikingHippocampus (loop-closure = attractor pattern-completion → snap the estimate; place novelty
    = 1−recall-sim feeds the CLS novelty signal);
  · the grid+place code is the STATE FEATURE φ for a navigation policy learned by the SAME
    SpikingBasalGanglia actor/critic (dopamine RPE) that already picks topics — a 4-action nav BG;
  · the place code is encoded into byte-levels under a new 'space' nerve marker and injected as a
    proprioceptive sensory frame the cortex perceives in the universal byte-code (senses.py/motor.py);
  · reaching a NEW place feeds SpikingEndocrine.wake_tick(novelty=…) so spatial exploration meets the
    novelty drive, and the homeostatic deficit-drop reward is added to the nav reward;
  · the whole embodiment step is gated by SpikingDynamics ignition so a default-OFF brain pays nothing.

§16-pattern module: _KEYS / set_params / state(), device/dtype-agnostic, DEFAULT OFF. Structural
params (n_modules/n_phase/place_n/world_size/grid_scale0) rebuild the fixed banks; a shape-invariant
restore guard keeps a resumed brain from crashing when they are live-tuned. Refs:
runs/deeper_brain_integrated_design.md (§ spatial); Banino et al. Nature 2018; Moser & Moser 2008.
"""
from __future__ import annotations
import math
import torch

# a distinctive multi-byte "nerve" marker for the spatial/proprioceptive sense (mirrors senses.NERVE).
SPACE_NERVE = bytes([0x02, 0x50, 0x02])   # STX 'P' — place/position sense
SPACE_FRAME_END = bytes([0x03])


class GridWorld:
    """A deterministic embodied body/environment: an N×N open field, a fixed goal cell, 4 cardinal
    actions (N/E/S/W), reward −1 per step and +10 on the goal, episodes capped at max_steps. Walls
    are the boundary (a bump keeps the agent in place). Fully deterministic given the seed."""

    # dx, dy for actions 0=N,1=E,2=S,3=W
    VEL = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def __init__(self, size=12, goal=None, max_steps=100, seed=0):
        self.size = int(size)
        self.goal = tuple(goal) if goal is not None else (self.size - 1, self.size - 1)
        self.max_steps = int(max_steps)
        self._g = torch.Generator().manual_seed(int(seed))
        self.x = self.y = 0
        self.steps = 0
        self.reset()

    def reset(self, pos=None):
        if pos is not None:
            self.x, self.y = int(pos[0]), int(pos[1])
        else:                                        # random start not on the goal
            while True:
                self.x = int(torch.randint(0, self.size, (1,), generator=self._g).item())
                self.y = int(torch.randint(0, self.size, (1,), generator=self._g).item())
                if (self.x, self.y) != self.goal:
                    break
        self.steps = 0
        return self.pos

    @property
    def pos(self):
        return (float(self.x), float(self.y))

    def velocity(self, action):
        return self.VEL[int(action) % 4]

    def step(self, action):
        """Apply an action → (pos, reward, done). Bumping the boundary costs a step but does not move."""
        dx, dy = self.velocity(action)
        self.x = min(self.size - 1, max(0, self.x + dx))
        self.y = min(self.size - 1, max(0, self.y + dy))
        self.steps += 1
        done_goal = (self.x, self.y) == self.goal
        reward = 10.0 if done_goal else -1.0
        done = done_goal or self.steps >= self.max_steps
        return self.pos, reward, done, done_goal


class SpikingSpatial:
    """Entorhinal grid modules + hippocampal place layer + analog path integration (§16-pattern)."""

    _KEYS = ("on", "n_modules", "n_phase", "place_n", "k_place", "grid_scale0", "scale_ratio",
             "grid_thr", "world_size", "vel_gain", "loop_close", "path_integration",
             "loop_thr", "snap_gain")

    def __init__(self, device=None, seed=0, dtype=None):
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.on = False                              # opt-in toggle (verify before defaulting on)
        self._seed = int(seed)
        # structural (rebuild the fixed banks when changed)
        self.n_modules = 5                           # discrete entorhinal scale modules (Stensola 2012)
        self.n_phase = 9                             # phase offsets per module (a 3×3 tiling of the unit cell)
        self.place_n = 64                            # hippocampal place cells
        self.k_place = 8                             # k-WTA winners (de Almeida) — sparse localised bumps
        self.grid_scale0 = 3.0                       # smallest grid period (world units)
        self.scale_ratio = 1.42                      # geometric scale ratio between modules (~√2, Stensola)
        self.grid_thr = 1.0                          # grid firing threshold on Σ_k cos (sparsifies the peaks)
        self.world_size = 12                         # for normalisation / phase tiling
        # dynamic (safe to live-tune without a rebuild)
        self.vel_gain = 1.0                          # how far one action advances the internal estimate
        self.loop_close = True                       # hippocampal loop-closure correction (default ON when module on)
        self.path_integration = True                # integrate velocity on blind steps (default ON)
        self.loop_thr = 0.6                          # min landmark similarity to trust a snap
        self.snap_gain = 0.8                         # fraction of the way to snap the estimate on loop closure
        # state
        self.pos_hat = [0.0, 0.0]                    # internal path-integrated position estimate
        self._last_place_frac = 0.0
        self._pi_drift = 0.0
        self._decode_acc = 0.0
        self.nav_return = 0.0                        # EMA episodic return (earns-keep signal)
        self.nav_steps = 0.0                         # EMA steps-to-goal
        self._place_novelty = 1.0
        # metric decode buffer (place → true pos); gradient-free, metric-only
        self._dec_place, self._dec_pos = [], []
        # landmark map for loop closure (place code → true position)
        self._lm_codes = None
        self._lm_pos = None
        self._build()

    # ---- fixed grid/place banks -------------------------------------- #
    def _build(self):
        g = torch.Generator().manual_seed(self._seed)
        dev = self.device
        kvecs, phases = [], []
        for m in range(self.n_modules):
            scale = self.grid_scale0 * (self.scale_ratio ** m)
            theta = float(torch.rand(1, generator=g).item()) * math.pi / 3.0   # module orientation ∈ [0,60°)
            kappa = 2.0 * math.pi / scale
            ks = []
            for j in range(3):                       # three 60°-separated wave vectors (hexagonal)
                a = theta + j * math.pi / 3.0
                ks.append([kappa * math.cos(a), kappa * math.sin(a)])
            kvecs.append(torch.tensor(ks, device=dev, dtype=self.dtype))       # (3,2)
            p = int(round(self.n_phase ** 0.5))
            offs = []
            for i in range(p):
                for j in range(p):                   # tile the unit cell with phase offsets
                    offs.append([scale * i / p, scale * j / p])
            offs = offs[:self.n_phase] or [[0.0, 0.0]]
            phases.append(torch.tensor(offs, device=dev, dtype=self.dtype))    # (P,2)
        self._kvecs = kvecs
        self._phases = phases
        self._grid_dim = sum(ph.shape[0] for ph in phases)
        # fixed sparse grid→place readout (Solstad 2006): random Gaussian projection
        self.W_gp = (torch.randn(self.place_n, self._grid_dim, generator=g) /
                     max(1, self._grid_dim) ** 0.5).to(device=dev, dtype=self.dtype)
        # clear metric/landmark caches whose width just changed
        self._dec_place, self._dec_pos = [], []
        self._lm_codes = None
        self._lm_pos = None

    @property
    def grid_dim(self):
        return self._grid_dim

    @property
    def place_dim(self):
        return self.place_n

    # ---- the codes ---------------------------------------------------- #
    @torch.no_grad()
    def grid_code(self, x, y):
        """Entorhinal grid population g(x,y): per module, thr(Σ_k cos(k·(pos−φ))) over 3 hexagonal
        wave vectors and P phase offsets. Concatenated across modules → the periodic metric code."""
        pos = torch.tensor([float(x), float(y)], device=self.device, dtype=self.dtype)
        outs = []
        for m in range(self.n_modules):
            d = pos.unsqueeze(0) - self._phases[m]           # (P,2)
            proj = d @ self._kvecs[m].t()                    # (P,3)
            gm = torch.cos(proj).sum(1) - self.grid_thr      # (P,)
            outs.append(torch.clamp(gm, min=0.0))
        g = torch.cat(outs)
        return g / (g.norm() + 1e-6)

    @torch.no_grad()
    def place_code(self, x, y):
        """Hippocampal place layer: fixed sparse grid→place readout + k-WTA → a localised sparse bump."""
        g = self.grid_code(x, y)
        h = torch.relu(self.W_gp @ g)                        # (place_n,)
        k = max(1, min(self.k_place, self.place_n))
        if k < self.place_n:
            thr = h.topk(k).values[-1]
            h = torch.where(h >= thr, h, torch.zeros_like(h))
        self._last_place_frac = float((h > 0).float().mean())
        return h / (h.norm() + 1e-6)

    # ---- path integration + loop closure ------------------------------ #
    def reset_estimate(self, pos):
        self.pos_hat = [float(pos[0]), float(pos[1])]

    @torch.no_grad()
    def _store_landmark(self, place, pos):
        p = place.detach().unsqueeze(0)
        xy = torch.tensor([[float(pos[0]), float(pos[1])]], device=self.device, dtype=self.dtype)
        if self._lm_codes is None or self._lm_codes.shape[1] != p.shape[1]:
            self._lm_codes, self._lm_pos = p, xy
        else:
            self._lm_codes = torch.cat([self._lm_codes, p])[-512:]
            self._lm_pos = torch.cat([self._lm_pos, xy])[-512:]

    @torch.no_grad()
    def snap_loop_close(self, hippo=None):
        """Bound path-integration drift by landmark loop-closure. Recall the stored place code keyed
        on the current estimate's grid key (through the modern-Hopfield hippocampus when provided, and
        the internal landmark map for the metric snap); if the best match is confident, SNAP the
        estimate toward the associated stored position. Returns (snapped, similarity)."""
        cur = self.place_code(*self.pos_hat)
        # interconnection: exercise the brain's own attractor map (pattern-completion + novelty)
        if hippo is not None:
            try:
                rec = hippo.recall(cur.unsqueeze(0))
                sim_h = float(torch.cosine_similarity(rec, cur.unsqueeze(0), dim=1).clamp(-1, 1).mean())
                self._place_novelty = float(max(0.0, min(1.0, 1.0 - sim_h)))
            except Exception:
                pass
        if self._lm_codes is None or self._lm_codes.shape[0] == 0:
            return False, 0.0
        sims = torch.cosine_similarity(self._lm_codes, cur.unsqueeze(0), dim=1)
        j = int(sims.argmax().item()); s = float(sims[j].item())
        if s >= self.loop_thr:
            tgt = self._lm_pos[j]
            self.pos_hat[0] += self.snap_gain * (float(tgt[0]) - self.pos_hat[0])
            self.pos_hat[1] += self.snap_gain * (float(tgt[1]) - self.pos_hat[1])
            return True, s
        return False, s

    @torch.no_grad()
    def observe(self, true_pos, action=None, sensed=True, hippo=None):
        """One embodiment update. When SENSED the estimate is fixed to the true position (a landmark
        fix) and a landmark is laid down; when BLIND, path integration advances the estimate by the
        action's velocity and loop-closure recall corrects the drift. Returns the place vector."""
        if sensed:
            self.pos_hat = [float(true_pos[0]), float(true_pos[1])]
            place = self.place_code(*self.pos_hat)
            self._store_landmark(place, true_pos)
            if hippo is not None:                    # the place vector IS a spatial fingerprint → store it
                try: hippo.store(place.unsqueeze(0))
                except Exception: pass
        else:
            if self.path_integration and action is not None:
                vx, vy = self._vel(action)
                lim = float(self.world_size - 1)
                self.pos_hat[0] = min(lim, max(0.0, self.pos_hat[0] + self.vel_gain * vx))
                self.pos_hat[1] = min(lim, max(0.0, self.pos_hat[1] + self.vel_gain * vy))
            if self.loop_close:
                try: self.snap_loop_close(hippo=hippo)
                except Exception: pass
            place = self.place_code(*self.pos_hat)
        # metric buffer (bounded) + drift
        self._dec_place.append(place.detach()); self._dec_pos.append([float(true_pos[0]), float(true_pos[1])])
        if len(self._dec_place) > 400:
            self._dec_place = self._dec_place[-400:]; self._dec_pos = self._dec_pos[-400:]
        self._pi_drift = math.hypot(self.pos_hat[0] - float(true_pos[0]),
                                    self.pos_hat[1] - float(true_pos[1]))
        return place

    def _vel(self, action):
        return GridWorld.VEL[int(action) % 4]

    # ---- metrics ------------------------------------------------------ #
    @torch.no_grad()
    def decode_acc(self):
        """spatial_decode_acc: 1 − normalised MSE (≈R²) of true (x,y) linearly ridge-decoded from the
        place population. Metric-only (gradient-cut), fit on the cached position buffer."""
        n = len(self._dec_place)
        if n < max(8, self.place_n // 4):
            return self._decode_acc
        X = torch.stack(self._dec_place)                     # (n, place_n)
        X = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], 1)
        Y = torch.tensor(self._dec_pos, device=X.device, dtype=X.dtype)        # (n,2)
        lam = 1e-2
        A = X.t() @ X + lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        W = torch.linalg.solve(A, X.t() @ Y)
        pred = X @ W
        ss_res = ((pred - Y) ** 2).sum(0)
        ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(0) + 1e-6
        r2 = (1.0 - ss_res / ss_tot).mean().clamp(0.0, 1.0)
        self._decode_acc = float(r2)
        return self._decode_acc

    def record_episode(self, ret, steps, a=0.1):
        """Fold one navigation episode's return + steps-to-goal into the earns-keep EMAs."""
        self.nav_return = ret if self.nav_return == 0.0 else (1 - a) * self.nav_return + a * ret
        self.nav_steps = steps if self.nav_steps == 0.0 else (1 - a) * self.nav_steps + a * steps

    # ---- §16 module contract ----------------------------------------- #
    def set_params(self, **kw):
        applied = {}
        structural = ("n_modules", "n_phase", "place_n", "grid_scale0", "scale_ratio",
                      "grid_thr", "world_size")
        rebuild = False
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                        # 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif k in ("n_modules", "n_phase", "place_n", "k_place", "world_size"):
                v = int(v)
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
            if k in structural:
                rebuild = True
        if rebuild:
            self._build()
            applied["_rebuilt"] = True
        return applied

    def state(self):
        return dict(on=self.on, spatial_decode_acc=round(self.decode_acc(), 3),
                    pi_drift=round(self._pi_drift, 3), nav_return=round(self.nav_return, 3),
                    nav_steps_to_goal=round(self.nav_steps, 2),
                    place_active_frac=round(self._last_place_frac, 3),
                    place_novelty=round(self._place_novelty, 3),
                    grid_dim=self._grid_dim, place_dim=self.place_n)

    # ---- persistence (shape-invariant restore guard) ----------------- #
    def state_dict(self):
        return {**{k: getattr(self, k) for k in self._KEYS},
                "W_gp": self.W_gp.detach().cpu(),
                "nav_return": self.nav_return, "nav_steps": self.nav_steps}

    def load_state_dict(self, sd):
        """Restore live params + the fixed readout. Structural params are applied first (rebuilding the
        banks); the W_gp tensor is restored ONLY if its shape still matches — a live-tuned place_n/
        n_modules just keeps the freshly built bank (the same shape-invariant guard the BG uses)."""
        if not sd:
            return
        for k in self._KEYS:
            if k in sd:
                setattr(self, k, sd[k])
        self._build()
        self.nav_return = float(sd.get("nav_return", 0.0)); self.nav_steps = float(sd.get("nav_steps", 0.0))
        w = sd.get("W_gp")
        if w is not None and tuple(w.shape) == tuple(self.W_gp.shape):
            self.W_gp = w.to(self.device)

    # ---- proprioceptive 'space' sensory frame ------------------------ #
    @torch.no_grad()
    def encode_place_bytes(self, place):
        """Quantise the place vector to byte-levels (0-255) so position flows in the SAME universal
        byte-code as text/vision/time — proprioception. Wrapped with the 'space' nerve marker."""
        p = place.detach().float()
        p = p / (p.max() + 1e-6)
        body = [int(max(0, min(255, round(float(v) * 255)))) for v in p.tolist()]
        return list(SPACE_NERVE) + body + list(SPACE_FRAME_END)
