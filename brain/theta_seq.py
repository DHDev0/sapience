"""§17 · ThetaSequenceMemory — hippocampal TEMPORAL-ORDER memory (theta sequences / trajectory replay).

The existing §4 SpikingHippocampus is an ORDERLESS modern-Hopfield content store: store(xi)/recall(cue)
maps a DG key → a value with no notion of "what came next". This companion module gives it the one thing
it lacks — TIME. It records ORDERED TRAJECTORIES of the very same episode fingerprints the hippocampus
content-stores, and can REPLAY them forward and reverse (Foster-Wilson 2006 reverse replay; Diba-Buzsaki
2007), the sharp-wave-ripple substrate of consolidation and path credit-assignment.

Biology: within a ~125 ms theta cycle place cells fire a time-compressed sweep of the past→present→future
trajectory (O'Keefe-Recce 1993; Skaggs 1996), so ordered episodic sequence is encoded in compressed spike
order. Computationally this is a successor representation (Stachenfeld 2017) and CLS sequence replay for
interleaved consolidation (McClelland 1995; van de Ven 2020). "Theta compression" here = a whole ordered
trajectory is emitted within ONE replay() call, not one item per physics tick.

Interconnection (mandatory — this is not an isolated toggle):
  • It does NOT own a DG separator — it calls hippo.separate(xi) so its keys live in the SAME DG space as
    the content store; it borrows hippo's beta/g_inh/thr for the spiking soft-WTA successor lookup; on
    hippo.grow()/adult-neurogenesis it re-derives its keys via _reseparate() exactly as the hippo does.
  • It reads the identical fingerprint the hippo just stored in life._index_episode, so content memory and
    order memory index one episode stream.
  • Its replay() feeds an ORDERED training stream into the cortex sleep-consolidation path, and seeds the
    generative self-replay dreams — coupling sequence memory into the buffer-free CLS replay.
  • The ripple gate reuses endocrine.sleep_pressure() and the CLS _novelty signal.

Data model (mirrors hippo's key/value store so it reuses the same spiking machinery):
  seq_vals (T,N)  shared pool of committed item VALUES (the fingerprints), in commit order.
  seq_keys (T,M)  = hippo.separate(seq_vals) — DG keys, re-derived on DG growth.
  seq_texts       parallel python list of the item TEXTS (for cortex replay).
  traj            ring buffer (capacity-bounded) of committed trajectories, each an ordered list of item
                  INDICES into the shared pool.
  _item_next/_item_prev  successor / predecessor index per item (−1 = none) — the content-addressable
                  SUCCESSOR and PREDECESSOR stores (keys=item_t, vals=item_{t+1} / item_{t-1}).
Default OFF (opt-in; earns its keep via the ordered-probe A/B before ever defaulting on).

Refs: runs/deeper_brain_integrated_design.md §16 (CLS/replay), spiking_modules.py SpikingHippocampus.
"""
from __future__ import annotations
import random
import torch
from .spiking import spike


class ThetaSequenceMemory:
    _KEYS = ("on", "L", "capacity", "beta", "g_inh", "thr", "fwd_frac", "ripple_k", "commit_on_boundary")

    def __init__(self, hippo, device=None, dtype=None):
        self.hippo = hippo                     # SHARE the DG space (no own separator)
        self.device = getattr(hippo, "device", device)
        # DTYPE-faithful: inherit the hippo's projection dtype (so a bf16 hippo does not silently upcast the
        # committed fingerprint pools); the empty pools are created WITH this dtype, not the default float32.
        _ref = getattr(hippo, "proj", None)
        self._sep_dtype = _ref.dtype if isinstance(_ref, torch.Tensor) else torch.get_default_dtype()
        self.dtype = dtype if dtype is not None else self._sep_dtype
        self.on = False                        # opt-in toggle (verify before defaulting on)
        # theta-cycle window + capacity + spiking-WTA borrows (defaults mirror the hippo's)
        self.L = 8.0                           # theta-cycle window: commit a trajectory at this length
        self.capacity = 256.0                  # ring-buffer of committed trajectories
        self.beta = float(getattr(hippo, "beta", 8.0))     # similarity gain (borrowed default)
        self.g_inh = float(getattr(hippo, "g_inh", 1.4))   # lateral WTA inhibition (borrowed default)
        self.thr = float(getattr(hippo, "thr", 0.55))      # CA3 spike threshold (borrowed default)
        self.fwd_frac = 0.5                    # P(forward) vs reverse in a ripple replay
        self.ripple_k = 4.0                    # base number of ripple replays per night (× ripple_gate)
        self.commit_on_boundary = True         # also commit when a sentence/line boundary token is hit
        self._acc_thr = 0.85                   # (legacy) absolute cosine threshold — kept for back-compat only
        self._acc_margin = 0.02                # rank/margin metric: cos(true) must lead the best OTHER item by this
        # state — small tensors on the hippo's device; trajectories are python lists of ints
        N = int(getattr(hippo, "N", 256)); M = int(getattr(hippo, "M", N * 4))
        self.seq_vals = torch.zeros(0, N, device=self.device, dtype=self.dtype)
        self.seq_keys = torch.zeros(0, M, device=self.device, dtype=self.dtype)
        self.seq_texts = []                    # per-item text
        self.traj = []                         # committed trajectories (lists of item indices)
        self._open = []                        # the open WAKE trajectory being built
        self._item_next = torch.zeros(0, dtype=torch.long, device=self.device)
        self._item_prev = torch.zeros(0, dtype=torch.long, device=self.device)
        # metrics
        self.replay_fwd = 0; self.replay_rev = 0
        self.ripple_gate = 0.0                 # scalar 0..1 that gated replay this night (life sets it)
        self._acc = 0.0                        # cached seq_recall_acc (state() reads this)

    @torch.no_grad()
    def _sep(self, x):
        """DG-separate through the SHARED hippo separator, bridging dtypes: feed the hippo in ITS projection
        dtype (so a bf16 module does not break the hippo matmul) and return the key in the module dtype."""
        return self.hippo.separate(x.to(device=self.device, dtype=self._sep_dtype)).to(dtype=self.dtype)

    # ---- wake: append + commit trajectories --------------------------- #
    @torch.no_grad()
    def observe_item(self, xi, text=None):
        """Append the just-experienced episode (the SAME fingerprint the hippo content-stored) to the open
        wake trajectory; commit at length L or on a boundary token — capturing order from the live stream."""
        if not self.on or xi is None:
            return
        xi = xi.to(device=self.device, dtype=self.dtype)     # dtype-faithful: no silent upcast of the pool
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
        idx = self.seq_vals.shape[0]
        self.seq_vals = torch.cat([self.seq_vals, xi])
        self.seq_keys = torch.cat([self.seq_keys, self._sep(xi)])
        self.seq_texts.append("" if text is None else str(text))
        self._open.append(idx)
        boundary = self.commit_on_boundary and text and str(text).strip()[-1:] in ".!?\n"
        if len(self._open) >= max(2, int(self.L)) or boundary:
            self._commit()

    @torch.no_grad()
    def _commit(self):
        """Commit the open trajectory (theta compression: one ordered sweep becomes one stored trajectory)."""
        if len(self._open) >= 2:
            self.traj.append(list(self._open))
        self._open = []
        while len(self.traj) > max(1, int(self.capacity)):
            self._evict_oldest()
        self._rebuild_links()

    @torch.no_grad()
    def _evict_oldest(self):
        """Drop the oldest committed trajectory and its per-occurrence items; reindex the pool + trajectories."""
        drop = set(self.traj[0]); self.traj = self.traj[1:]
        keep = [i for i in range(self.seq_vals.shape[0]) if i not in drop]
        remap = {old: new for new, old in enumerate(keep)}
        if keep:
            sel = torch.tensor(keep, dtype=torch.long, device=self.device)
            self.seq_vals = self.seq_vals.index_select(0, sel)
            self.seq_keys = self.seq_keys.index_select(0, sel)
        else:
            self.seq_vals = self.seq_vals[:0]; self.seq_keys = self.seq_keys[:0]   # slice keeps device+dtype
        self.seq_texts = [self.seq_texts[i] for i in keep]
        self.traj = [[remap[i] for i in tr] for tr in self.traj]
        self._open = [remap[i] for i in self._open if i in remap]

    @torch.no_grad()
    def _rebuild_links(self):
        """Recompute the successor / predecessor index stores from the committed trajectories."""
        T = self.seq_vals.shape[0]
        nxt = torch.full((T,), -1, dtype=torch.long, device=self.device)
        prv = torch.full((T,), -1, dtype=torch.long, device=self.device)
        for tr in self.traj:
            for a, b in zip(tr, tr[1:]):
                nxt[a] = b; prv[b] = a
        self._item_next = nxt; self._item_prev = prv

    @torch.no_grad()
    def _reseparate(self):
        """Re-derive the sequence keys after the hippo's DG separator changed (growth / neurogenesis), so the
        trajectories stay recallable in the SAME (grown) DG space — mirrors SpikingHippocampus._reseparate."""
        M = int(getattr(self.hippo, "M", self.seq_keys.shape[1]))
        if self.seq_vals.shape[0]:
            self.seq_keys = self._sep(self.seq_vals)
        else:
            self.seq_keys = torch.zeros(0, M, device=self.device, dtype=self.dtype)

    # ---- recall / replay --------------------------------------------- #
    @torch.no_grad()
    def _wta_weights(self, q, valid):
        """The hippo's spiking soft-WTA over the stored item keys, restricted to items in `valid`. Returns
        spike-accumulation weights (1,S). Mirrors SpikingHippocampus.recall (competitive LIF + lateral inh)."""
        drive = self.beta * (q @ self.seq_keys.t()) / (q.sum(1, keepdim=True) + 1e-6)
        drive = drive.masked_fill(~valid.unsqueeze(0), float("-inf"))
        mx = drive.max(1, keepdim=True).values.clamp(min=1e-6)
        drive = drive / mx                                          # best valid ≈ 1
        v = torch.zeros_like(drive); s = torch.zeros_like(drive); acc = torch.zeros_like(drive)
        for _ in range(8):
            inh = self.g_inh * (s.sum(1, keepdim=True) - s)         # lateral WTA inhibition
            v = 0.8 * v * (1 - s) + drive - inh
            v = torch.nan_to_num(v, neginf=-1e9)                    # invalid (−inf) items never fire
            s = spike(v - self.thr); acc = acc + s
        return acc / (acc.sum(1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def recall_next(self, cue, reverse=False):
        """DG-separate the cue, spiking soft-WTA match against the item keys (restricted to items that HAVE a
        successor/predecessor), read out the stored successor (or predecessor) value. Returns (1,N) or None."""
        if self.seq_keys.shape[0] == 0:
            return None
        cue = cue.to(device=self.device, dtype=self.dtype)
        if cue.dim() == 1:
            cue = cue.unsqueeze(0)
        link = self._item_prev if reverse else self._item_next
        valid = link >= 0
        if not bool(valid.any()):
            return None
        q = self._sep(cue)                                         # DG key in module dtype (WTA matmul-safe)
        w = self._wta_weights(q, valid)                            # (1,S) spike-weighted winners
        tgt = self.seq_vals.index_select(0, link.clamp(min=0))     # successor/predecessor value per item
        return w @ tgt                                             # invalid items carry weight 0

    @torch.no_grad()
    def _nearest_text(self, val):
        if self.seq_vals.shape[0] == 0:
            return ""
        sims = torch.cosine_similarity(self.seq_vals, val.to(self.device), dim=1)
        return self.seq_texts[int(sims.argmax())]

    @torch.no_grad()
    def replay(self, seed=None, length=None, reverse=False):
        """Roll recall_next forward/back from a seed item across a whole trajectory in ONE call (the biological
        theta-sequence sweep). Returns (values, texts) — the ordered recalled trajectory. Increments the
        forward/reverse ripple counters. If `seed` is None a random committed trajectory's endpoint is used."""
        if not self.traj:
            return [], []
        tr = random.choice(self.traj)
        start = tr[-1] if reverse else tr[0]
        if seed is None:
            seed = self.seq_vals[start:start + 1]
        L = int(length) if length else len(tr)
        cur = seed.to(self.device)
        if cur.dim() == 1:
            cur = cur.unsqueeze(0)
        vals = [cur]; texts = [self._nearest_text(cur)]
        for _ in range(max(0, L - 1)):
            nv = self.recall_next(cur, reverse=reverse)
            if nv is None:
                break
            vals.append(nv); texts.append(self._nearest_text(nv)); cur = nv
        if reverse: self.replay_rev += 1
        else:       self.replay_fwd += 1
        return vals, texts

    @torch.no_grad()
    def replay_text(self, reverse=False):
        """A ripple replay as an ORDERED text stream (for cortex sleep-consolidation learn_text)."""
        _, texts = self.replay(reverse=reverse)
        return " ".join(t for t in texts if t)

    @torch.no_grad()
    def nearest_index(self, val):
        """Argmax stored item by cosine to `val` (−1 if the pool is empty). The RANK primitive the order-
        specificity metric rides on: a recalled successor is order-CORRECT iff its nearest stored item is the
        true next item — a discrimination that is invariant to the collapsed absolute geometry of byte-
        histogram fingerprints (whose any-pair cosine is ~0.75-0.9)."""
        if self.seq_vals.shape[0] == 0:
            return -1
        sims = torch.cosine_similarity(self.seq_vals, val.to(self.device, self.dtype), dim=1)
        return int(sims.argmax())

    @torch.no_grad()
    def _order_correct(self, nv, true_idx):
        """RANK/MARGIN order-specificity test for one recalled successor nv against the true-next index.
        Correct iff (a) the true item is the argmax stored item by cosine AND (b) it leads the best OTHER
        stored item by >= _acc_margin. This measures TEMPORAL order (did we read the right successor?),
        NOT raw reactivation, and is well-posed even when all fingerprints are mutually high-cosine."""
        if true_idx < 0 or true_idx >= self.seq_vals.shape[0]:
            return False
        sims = torch.cosine_similarity(self.seq_vals, nv.to(self.device, self.dtype), dim=1)
        true_cos = float(sims[true_idx])
        others = sims.clone(); others[true_idx] = float("-inf")
        best_other = float(others.max()) if self.seq_vals.shape[0] > 1 else -1.0
        return (int(sims.argmax()) == int(true_idx)) and (true_cos - best_other >= self._acc_margin)

    @torch.no_grad()
    def sequence_recall_accuracy(self):
        """Headline order-memory metric (RANK/MARGIN-based, TEACHER-FORCED). For each stored trajectory and
        each position t, cue with the TRUE stored item_t and score whether the recalled successor is order-
        CORRECT — the true next item is its nearest stored item and leads all others by a margin. This is the
        governing equation of the earns-keep probe ("predict item t+1 GIVEN item t"): it isolates whether the
        stored SUCCESSOR LINK is read specifically, not the compounding reconstruction drift of a rolled
        replay. Order-SPECIFIC by construction: an out-of-order successor is not the argmax, so it never
        scores — which is exactly why it collapses on shuffled targets. Cached to _acc."""
        total = 0; correct = 0
        for tr in self.traj:
            if len(tr) < 2:
                continue
            for pos in range(len(tr) - 1):
                nv = self.recall_next(self.seq_vals[tr[pos]:tr[pos] + 1], reverse=False)   # teacher-forced cue
                if nv is None:
                    break
                total += 1; correct += int(self._order_correct(nv, tr[pos + 1]))
        self._acc = (correct / total) if total else 0.0
        return self._acc

    @torch.no_grad()
    def move_to(self, device):
        self.device = device                                  # dtype preserved (.to(device) keeps dtype)
        self.seq_vals = self.seq_vals.to(device); self.seq_keys = self.seq_keys.to(device)
        self._item_next = self._item_next.to(device); self._item_prev = self._item_prev.to(device)
        return self

    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' must disable, not enable
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on, seq_recall_acc=round(self._acc, 3), n_trajectories=len(self.traj),
                    replay_fwd=self.replay_fwd, replay_rev=self.replay_rev,
                    ripple_gate=round(float(self.ripple_gate), 3))
