"""The spiking cortex (SpikingBrain): learn/think/generate/resonate/grow/prune/save-load/diagnostics."""
import os, torch
from brain.spiking_brain import SpikingBrain

DEV = torch.device("cpu")
TXT = "the quick brown fox jumps over the lazy dog. water runs to the sea. " * 40


def _brain(**kw):
    torch.manual_seed(0)
    return SpikingBrain(DEV, emb=32, hidden=64, layers=2, seed=0, **kw)


def test_eprop_learns_dense_and_sparse():
    # e-prop (forward-in-time, local eligibility + random-feedback signal, no BPTT) must LEARN,
    # both on the dense path and on the sparse (O(nnz), no H²) path.
    for kw in (dict(), dict(sparse=True, rec_fanin=8, in_fanin=8, syn_density=0.6)):
        b = SpikingBrain(DEV, emb=32, hidden=64, layers=2, cell="lif", seed=0, **kw)
        b.learn_rule = "eprop"; b.eprop_lr_scale = 5000.0   # fan-in-normalized regime (width-invariant)
        bpb0 = b.bits_per_byte(TXT)
        for _ in range(120):
            b.learn_eprop(TXT, epochs=1, bs=16, max_steps=1, seq=32)
        assert b.bits_per_byte(TXT) < bpb0 - 0.3                  # genuinely learns forward-in-time


def test_eprop_neuromod_gate_scales_update():
    # the three-factor gate M must scale the update — gate=0 → NO weight change (§5 coupling is real).
    b = SpikingBrain(DEV, emb=32, hidden=64, layers=2, cell="lif", seed=0)
    b.learn_rule = "eprop"
    w0 = b.head.weight.clone()
    b.learn_eprop(TXT, epochs=1, bs=8, max_steps=2, seq=32, gate=0.0)
    assert torch.equal(b.head.weight, w0)                        # M=0 → plasticity gated off
    b.learn_eprop(TXT, epochs=1, bs=8, max_steps=2, seq=32, gate=1.0)
    assert not torch.equal(b.head.weight, w0)                    # M=1 → it updates


def test_membrane_readout_default_and_learns():
    b = _brain()
    assert b.readout == "mem" and b.cell_kind == "lif"           # the A/B-chosen default
    bpb0 = b.bits_per_byte(TXT)
    for _ in range(8):
        b.learn_text(TXT, epochs=1, bs=8, max_steps=6)
    assert b.bits_per_byte(TXT) < bpb0                           # it learns


def test_learn_returns_loss_trajectory():
    b = _brain()
    r = b.learn_text(TXT, epochs=1, bs=8, max_steps=5)
    assert isinstance(r, tuple) and len(r) == 2 and r[0] >= r[1] - 1e-6


def test_think_generate_persist_state():
    b = _brain()
    b.learn_text(TXT, epochs=1, bs=8, max_steps=5)
    assert isinstance(b.think(n=12), str)
    assert b.generate("The ", n=20).startswith("The ")


def test_resonate_k_parallel_streams():
    b = _brain()
    b.learn_text(TXT, epochs=1, bs=8, max_steps=4)
    b.observe_stream("the city of ")
    streams = b.resonate(k=5, n=10)
    assert len(streams) == 5 and all(isinstance(s, str) for s in streams)


def test_grow_identity_preserving():
    b = _brain(); b.learn_text(TXT, epochs=1, bs=8, max_steps=6)
    pre = b.bits_per_byte(TXT); b.grow(32); post = b.bits_per_byte(TXT)
    assert abs(pre - post) < 1e-3 and b.hidden == 96             # grow preserves function


def test_synaptic_pruning_keeps_neurons_and_sticks():
    b = _brain(); b.learn_text(TXT, epochs=1, bs=8, max_steps=6)
    h = b.hidden
    k = b.prune(frac=0.1)                                        # prune weakest SYNAPSES
    assert b.hidden == h and k > 0                              # NEURON count unchanged; connections cut
    cut = sum(int((~m).sum()) for m in b._pmask)
    b.learn_text(TXT, epochs=1, bs=8, max_steps=4)              # train again
    still_zero = sum(int((lin.weight == 0).sum()) for lin in b._prune_targets())
    assert still_zero >= cut                                     # pruned synapses did NOT regrow


def test_save_load_roundtrip_after_growth():
    b = _brain(); b.learn_text(TXT, epochs=1, bs=8, max_steps=6)
    b.grow(32); b.grow(32)                                       # non-uniform layer widths
    tmp = "/tmp/_cortex_test.pt"; b.save(tmp)
    b2 = SpikingBrain(DEV, emb=32, hidden=64, layers=2, seed=9); b2.load(tmp)
    assert b2.hidden == b.hidden
    assert abs(b2.bits_per_byte(TXT) - b.bits_per_byte(TXT)) < 1e-4
    os.remove(tmp)


def test_diagnostics():
    b = _brain(); b.learn_text(TXT, epochs=1, bs=8, max_steps=6)
    assert b.train_perplexity(TXT) > 1.0
    g = b.generate_diag(n=40)
    assert 0 <= g["entropy_bits"] <= 8 and g["perplexity"] >= 1.0
    assert 0.0 <= b.spike_rate(TXT) <= 1.0
    assert len(b.weight_stats()) >= 3


def test_develop_grows_synapses_keeps_neurons_fixed():
    # NEW model: neurons are fixed at birth; development is SYNAPTIC (childhood synaptogenesis
    # densifies, adolescence prunes). The neuron count must NOT change across the cycle.
    b = SpikingBrain(DEV, emb=16, hidden=80, layers=1, seed=0, syn_density=0.5); b.seen_bytes = 10**9
    b.grow_until, b.prune_until = 2, 4
    n0 = b.neuron_count()
    grown = pruned = 0
    for _ in range(5):
        d = b.develop(add=24)
        grown += d["grown"]; pruned += d["pruned"]
        assert b.neuron_count() == n0                           # NEURONS FIXED every cycle
    assert grown > 0                                            # child GREW synapses (not neurons)
    assert pruned > 0                                          # adolescent PRUNED synapses
    assert b.hidden == 80                                       # neuron population unchanged
