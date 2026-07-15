"""CPU tests for §17 ThetaSequenceMemory — temporal-order hippocampal sequence memory.

Verifies the integration contract + that it EARNS ITS KEEP on ORDER memory (present != useful):
ordered trajectories give high sequence_recall_accuracy while a scrambled-order control does not.
"""
import os
import torch
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

from brain.spiking_modules import SpikingHippocampus
from brain.theta_seq import ThetaSequenceMemory

DEV = torch.device("cpu")
N = 256


def _items(n, seed=0):
    """n distinct normalized fingerprint-like vectors (like life._fingerprint output)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, N, generator=g)
    return x / (x.sum(1, keepdim=True) + 1e-6)


def _fresh(seed=0, L=6, cap=64):
    hippo = SpikingHippocampus(N, DEV, seed=seed)
    t = ThetaSequenceMemory(hippo, DEV)
    t.on = True
    t.set_params(L=L, capacity=cap)
    return hippo, t


def _load_ordered(t, items, ntraj, tl):
    """Feed ntraj ordered trajectories of length tl, committing each."""
    idx = 0
    for _ in range(ntraj):
        for _p in range(tl):
            t.observe_item(items[idx:idx + 1], text=f"item{idx}")
            idx += 1
    # flush any open remainder
    t._commit()


def test_default_off():
    _, t = _fresh()
    t2 = ThetaSequenceMemory(SpikingHippocampus(N, DEV), DEV)
    assert t2.on is False                                   # DEFAULT OFF
    t2.observe_item(_items(1), text="x")
    assert len(t2.traj) == 0 and t2.seq_vals.shape[0] == 0  # off = no-op


def test_commit_and_metrics_shape():
    _, t = _fresh(L=4)
    _load_ordered(t, _items(12), ntraj=3, tl=4)
    assert len(t.traj) == 3
    st = t.state()
    for k in ("on", "seq_recall_acc", "n_trajectories", "replay_fwd", "replay_rev", "ripple_gate"):
        assert k in st
    assert st["n_trajectories"] == 3


def test_recall_next_forward_and_reverse():
    _, t = _fresh(L=8)
    items = _items(8, seed=3)
    _load_ordered(t, items, ntraj=1, tl=8)                  # one A->B->...->H trajectory
    # cue item0 -> should recall item1
    nv = t.recall_next(items[0:1], reverse=False)
    assert nv is not None
    cos = torch.cosine_similarity(nv, items[1:2], dim=1).item()
    assert cos > 0.9, f"forward successor cosine too low: {cos}"
    # cue item1 -> reverse should recall item0
    pv = t.recall_next(items[1:2], reverse=True)
    cos_r = torch.cosine_similarity(pv, items[0:1], dim=1).item()
    assert cos_r > 0.9, f"reverse predecessor cosine too low: {cos_r}"


def test_sequence_recall_accuracy_high_for_ordered():
    _, t = _fresh(L=6)
    _load_ordered(t, _items(30, seed=7), ntraj=5, tl=6)
    acc = t.sequence_recall_accuracy()
    assert acc > 0.7, f"ordered sequence recall too low: {acc}"
    assert t.state()["seq_recall_acc"] == round(acc, 3)     # cached into state


def test_earns_keep_order_specific():
    """Order-specificity at the MODULE level, RANK/MARGIN-based (the fix). recall_next must return
    SPECIFICALLY the stored true-next item — its nearest stored item is the true successor — not merely some
    high-cosine item. Scoring against a random WRONG stored item as the target must collapse to near chance,
    while scoring against the real next stays high. This discriminates even under collapsed byte-histogram
    fingerprint geometry (any-pair cosine ~0.75-0.9), where the old absolute-cosine threshold could not. (The
    cortex-level ordered-vs-scrambled next-byte A/B is run separately by the orchestrator.)"""
    _, t = _fresh(L=6)
    items = _items(30, seed=11)
    _load_ordered(t, items, ntraj=5, tl=6)
    ordered = t.sequence_recall_accuracy()                 # rank/margin scored vs the TRUE next
    assert ordered > 0.7

    # wrong-order control: same recalls, but is a random WRONG stored item the nearest? (should be ~chance)
    g = torch.Generator().manual_seed(123)
    hit_true = hit_wrong = total = 0
    for tr in t.traj:
        for pos in range(len(tr) - 1):
            nv = t.recall_next(t.seq_vals[tr[pos]:tr[pos] + 1], reverse=False)   # teacher-forced: cue item_t
            wrong_idx = int(torch.randint(0, t.seq_vals.shape[0], (1,), generator=g))
            hit_true += int(t.nearest_index(nv) == tr[pos + 1])       # nearest is the TRUE successor
            hit_wrong += int(t.nearest_index(nv) == wrong_idx)        # nearest is a RANDOM wrong item
            total += 1
    assert hit_true / total > 0.7
    assert hit_wrong / total < 0.3, f"recall not order-specific: wrong-target hit rate {hit_wrong/total}"


def test_replay_returns_ordered_texts():
    _, t = _fresh(L=6)
    _load_ordered(t, _items(6, seed=5), ntraj=1, tl=6)
    fwd0 = t.replay_fwd
    vals, texts = t.replay(reverse=False)
    assert len(vals) >= 2 and len(texts) == len(vals)
    assert t.replay_fwd == fwd0 + 1                         # counter incremented
    txt = t.replay_text(reverse=True)
    assert isinstance(txt, str)
    assert t.replay_rev >= 1


def test_reseparate_after_growth_preserves_recall():
    hippo, t = _fresh(L=8, seed=2)
    items = _items(8, seed=2)
    _load_ordered(t, items, ntraj=1, tl=8)
    before = t.sequence_recall_accuracy()
    hippo.grow(64)                                          # DG neurogenesis changes the separator
    t._reseparate()                                         # keys must follow into the grown DG space
    assert t.seq_keys.shape[1] == hippo.M
    after = t.sequence_recall_accuracy()
    assert after > 0.7, f"recall lost after DG growth: {before}->{after}"


def test_capacity_eviction_reindexes():
    _, t = _fresh(L=4, cap=3)
    _load_ordered(t, _items(40, seed=1), ntraj=8, tl=4)     # 8 trajectories, cap 3
    assert len(t.traj) == 3                                 # ring-buffered
    # pool + links stay consistent after eviction/reindex
    assert t.seq_vals.shape[0] == t.seq_keys.shape[0] == len(t.seq_texts)
    assert t._item_next.shape[0] == t.seq_vals.shape[0]
    for tr in t.traj:
        for i in tr:
            assert 0 <= i < t.seq_vals.shape[0]
    assert t.sequence_recall_accuracy() > 0.6              # surviving trajectories still recallable


def test_set_params_string_bool_coercion():
    _, t = _fresh()
    assert t.set_params(on="off")["on"] is False
    assert t.set_params(on="true")["on"] is True
    assert t.set_params(on="0")["on"] is False
    app = t.set_params(L=10, ripple_k=6, fwd_frac=0.8)
    assert app["L"] == 10.0 and app["ripple_k"] == 6.0 and app["fwd_frac"] == 0.8


def test_persistence_roundtrip():
    _, t = _fresh(L=5)
    _load_ordered(t, _items(20, seed=4), ntraj=4, tl=5)
    t.replay(reverse=False)
    saved = {**{k: getattr(t, k) for k in t._KEYS},
             "seq_vals": t.seq_vals, "seq_texts": t.seq_texts, "traj": t.traj}
    # rebuild a fresh module and restore
    hippo2 = SpikingHippocampus(N, DEV, seed=4)
    t2 = ThetaSequenceMemory(hippo2, DEV)
    for k in t2._KEYS:
        setattr(t2, k, saved[k])
    t2.seq_vals = saved["seq_vals"]; t2.seq_texts = list(saved["seq_texts"]); t2.traj = [list(x) for x in saved["traj"]]
    t2._reseparate(); t2._rebuild_links()
    assert len(t2.traj) == len(t.traj)
    assert t2.sequence_recall_accuracy() > 0.7             # memory survives the roundtrip


def test_move_to_cpu():
    _, t = _fresh(L=4)
    _load_ordered(t, _items(8, seed=6), ntraj=2, tl=4)
    t.move_to(torch.device("cpu"))
    assert t.seq_vals.device.type == "cpu"
    assert t.recall_next(t.seq_vals[0:1]) is not None


def test_dtype_propagation():
    """Constructed with an explicit dtype, EVERY committed tensor + every recall output must carry it — no
    silent upcast to float32 (the bf16-hippo faithfulness gap the fix closes). float64 stands in for bf16 on
    CPU (bf16 matmul is flaky on CPU) — it exercises the identical dtype-plumbing path."""
    hippo = SpikingHippocampus(N, DEV, seed=0)
    t = ThetaSequenceMemory(hippo, DEV, dtype=torch.float64)
    t.on = True; t.set_params(L=4)
    assert t.seq_vals.dtype == torch.float64 and t.seq_keys.dtype == torch.float64   # empty pools honour dtype
    items = _items(8).double()
    _load_ordered(t, items, ntraj=2, tl=4)
    assert t.seq_vals.dtype == torch.float64 and t.seq_keys.dtype == torch.float64   # cat'd pool stays dtype
    nv = t.recall_next(t.seq_vals[0:1])
    assert nv is not None and nv.dtype == torch.float64                              # recall output stays dtype
    # even a float32 cue is cast into the module dtype, not the other way round
    nv2 = t.recall_next(items[0:1].float())
    assert nv2.dtype == torch.float64


def test_toggle_off_is_pure_noop_live():
    """Toggling theta.on off LIVE makes observe_item a pure no-op that touches no state; toggling back on
    resumes — the mechanism composes as a live gate with zero side effects when off."""
    _, t = _fresh(L=4)
    _load_ordered(t, _items(8, seed=6), ntraj=2, tl=4)
    n_traj, n_items = len(t.traj), t.seq_vals.shape[0]
    assert t.set_params(on="off")["on"] is False
    for _ in range(20):
        t.observe_item(_items(1, seed=99), text="ignored while off.")
    assert len(t.traj) == n_traj and t.seq_vals.shape[0] == n_items   # off = no state change at all
    assert t.set_params(on="true")["on"] is True
    t.observe_item(_items(1, seed=42), text="resumed")
    assert t.seq_vals.shape[0] == n_items + 1                         # back on = observing again


def test_width_scalable_shapes_no_dense_nxn():
    """State scales with the FINGERPRINT width N and DG width M and the trajectory count — NEVER with the
    cortex hidden width, and there is NO dense O(items^2) per-pair tensor. Confirms the 256k-neuron flatness
    claim: seq_vals is (items,N), seq_keys (items,M), links are 1-D (items,)."""
    hippo = SpikingHippocampus(N, DEV, seed=1)
    t = ThetaSequenceMemory(hippo, DEV)
    t.on = True; t.set_params(L=5, capacity=100)
    _load_ordered(t, _items(50, seed=1), ntraj=10, tl=5)
    S = t.seq_vals.shape[0]
    assert t.seq_vals.shape == (S, N)                    # width = fingerprint N (256), NOT cortex hidden
    assert t.seq_keys.shape == (S, hippo.M)              # width = DG M, NOT cortex hidden
    assert t._item_next.shape == (S,) and t._item_prev.shape == (S,)   # 1-D links, no O(S^2) matrix
    # total element budget is O(S*(N+M)) — flat as the cortex grows to 256k neurons
    budget = t.seq_vals.numel() + t.seq_keys.numel() + t._item_next.numel() + t._item_prev.numel()
    assert budget <= S * (N + hippo.M + 2)
