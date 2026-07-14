"""The sparse (CSR) cortex — the connectome that lets the neuron count reach the hundreds of
thousands within RAM. Correctness anchor (sparse == dense scatter), learning, synaptic
growth/prune with fixed neurons, save/load, and the guard that the default path stays dense."""
import os
import torch
from brain.spiking import SparseLIFCell, LIFCell
from brain.spiking_brain import SpikingBrain

DEV = torch.device("cpu")
TXT = "the quick brown fox jumps over the lazy dog. " * 60


def _dense_wrec_from_sparse(c):
    """Scatter a sparse cell's active CSR values into a dense (H,H) recurrent weight matrix.
    accumulate=True sums parallel edges (a row may draw a column more than once), matching the
    gather/scatter forward's index_add semantics."""
    H = c.hid
    Wd = torch.zeros(H, H)
    rows = torch.repeat_interleave(torch.arange(H), (c.rec_crow[1:] - c.rec_crow[:-1]).long())
    Wd.index_put_((rows, c.rec_col.long()), (c.rec_val * c.rec_mask).detach(), accumulate=True)
    return Wd


def test_sparse_cell_equals_dense_scatter():
    # THE correctness anchor: a SparseLIFCell's run_seq must be bit-equal to a dense LIFCell whose
    # Wrec is the same CSR values scattered into a full matrix. Proves the spmm path is exact.
    torch.manual_seed(0)
    c = SparseLIFCell(8, 32, beta=0.9, thr=1.0, rec_fanin=6, syn_density=0.7, seed=1)
    dense = LIFCell(8, 32, beta=0.9, thr=1.0)
    with torch.no_grad():
        dense.Win.weight.copy_(c.Win.weight); dense.Win.bias.copy_(c.Win.bias)
        dense.Wrec.weight.copy_(_dense_wrec_from_sparse(c))
    x = torch.randn(4, 12, 8)
    ss, ms, _ = c.run_seq(x, c.init_state(4, DEV))
    sd, md, _ = dense.run_seq(x, dense.init_state(4, DEV))
    assert torch.allclose(ss, sd, atol=1e-5) and torch.allclose(ms, md, atol=1e-5)


def test_sparse_backward_is_dense_nnz_vector():
    # the custom autograd must return a DENSE (nnz,) grad on the values Parameter (no H² materialization),
    # and masked (silent) synapses must get exactly zero gradient.
    c = SparseLIFCell(8, 40, rec_fanin=6, syn_density=0.5, seed=2)
    ss, ms, _ = c.run_seq(torch.randn(4, 8, 8), c.init_state(4, DEV))
    ms.pow(2).mean().backward()
    assert c.rec_val.grad.shape == c.rec_val.shape                 # dense (nnz,), not H²
    assert bool((c.rec_val.grad[~c.rec_mask] == 0).all())          # silent synapses: zero grad


def test_sparse_value_gradient_equals_dense_scatter():
    # REGRESSION for the critical bug: the value-gradient must equal the dense recurrent gradient
    # sampled at the CSR pattern — i.e. the custom O(nnz·B) backward is EXACT, not just cheap.
    torch.manual_seed(0)
    c = SparseLIFCell(8, 24, rec_fanin=5, syn_density=0.6, seed=3)
    dense = LIFCell(8, 24)
    H = c.hid
    rows = torch.repeat_interleave(torch.arange(H), (c.rec_crow[1:] - c.rec_crow[:-1]).long())
    with torch.no_grad():
        dense.Win.weight.copy_(c.Win.weight); dense.Win.bias.copy_(c.Win.bias)
        Wd = torch.zeros(H, H)
        Wd.index_put_((rows, c.rec_col.long()), (c.rec_val * c.rec_mask).detach(), accumulate=True)
        dense.Wrec.weight.copy_(Wd)
    x = torch.randn(4, 9, 8)
    c.run_seq(x, c.init_state(4, DEV))[1].pow(2).sum().backward()
    dense.run_seq(x, dense.init_state(4, DEV))[1].pow(2).sum().backward()
    dvg = dense.Wrec.weight.grad[rows, c.rec_col.long()]           # dense grad at the same edges
    assert torch.allclose(c.rec_val.grad, dvg * c.rec_mask, atol=1e-4)


def test_sparse_cortex_learns_and_neurons_fixed():
    b = SpikingBrain(DEV, emb=32, hidden=96, layers=2, cell="alif", seed=0,
                     syn_density=0.5, sparse=True, rec_fanin=12, in_fanin=12)
    assert all(hasattr(c, "rec_val") for c in b.cells)             # both layers sparse
    n0 = b.neuron_count()
    bpb0 = b.bits_per_byte(TXT[-2000:])
    for _ in range(8):
        b.learn_text(TXT, epochs=1, bs=8, max_steps=6)
    assert b.bits_per_byte(TXT[-2000:]) < bpb0                     # it LEARNS
    s0 = b.active_synapse_count(); g = b.grow_synapses(0.3); s1 = b.active_synapse_count()
    p = b.prune(0.1); s2 = b.active_synapse_count()
    assert g > 0 and s1 > s0 and p > 0 and s2 < s1                 # synapses grow then prune
    assert b.neuron_count() == n0                                  # NEURONS FIXED throughout


def test_sparse_cortex_save_load_roundtrip():
    b = SpikingBrain(DEV, emb=32, hidden=80, layers=2, cell="alif", seed=1,
                     syn_density=0.6, sparse=True, rec_fanin=10, in_fanin=10)
    b.learn_text(TXT, epochs=1, bs=8, max_steps=6); b.prune(0.1)
    tmp = "/tmp/_sparse_rt.pt"; b.save(tmp)
    b2 = SpikingBrain(DEV, emb=8, hidden=8, layers=2, seed=9); b2.load(tmp)
    assert b2.neuron_count() == b.neuron_count()
    assert all(hasattr(c, "rec_val") for c in b2.cells)           # sparse cells reconstructed
    assert b2.active_synapse_count() == b.active_synapse_count()
    assert abs(b2.bits_per_byte(TXT[-2000:]) - b.bits_per_byte(TXT[-2000:])) < 1e-4
    os.remove(tmp)


def test_default_construction_is_dense():
    # GUARD: the default (and every small-net test) must take the dense path — no sparse cells,
    # no non-ones mask — so the existing suite stays byte-identical and fast.
    b = SpikingBrain(DEV, emb=16, hidden=64, layers=2, seed=0)     # syn_density default 0.5
    assert not any(hasattr(c, "rec_val") for c in b.cells)        # all dense
    assert b.sparse_cfg["threshold"] > 96                         # threshold above every test width
