"""The four other spiking systems: §1 cerebellum, §2 basal ganglia, §4 hippocampus, §5 neuromod."""
import torch
import torch.nn.functional as F
from brain.spiking_modules import (SpikingCerebellum, SpikingBasalGanglia,
                                    SpikingHippocampus, SpikingNeuromod)

DEV = torch.device("cpu")


def test_cerebellum_golgi_sparsity_and_learns():
    c = SpikingCerebellum(32, 4, DEV, n_granule=600, seed=1)
    u = torch.randn(16, 32)
    g = c.granule(u)
    assert g.shape == (16, 600) and (g >= 0).all()               # sparse granule rate code
    Wt = torch.randn(4, 32) * 0.3                                 # a linear target to learn
    mse0 = None
    for _ in range(200):
        x = torch.randn(32, 32)
        mse = c.train_step(x, x @ Wt.t(), eta=0.3)
        if mse0 is None: mse0 = float(mse)
    assert float(mse) < mse0                                      # delta rule reduces error


def test_cerebellum_grows():
    c = SpikingCerebellum(16, 2, DEV, n_granule=200, seed=0)
    c.grow(100)
    assert c.M.shape[0] == 300 and c.W.shape[1] == 300


def test_basal_ganglia_spiking_actor_critic_learns_and_grows():
    torch.manual_seed(0)
    bg = SpikingBasalGanglia(3, 3, DEV, n_msn=64, alpha_v=0.1, alpha_pi=0.5, seed=1)
    def feat(ctx): return F.one_hot(ctx, 3).float()
    r0 = (bg.act(feat(torch.arange(3)))[0] == torch.arange(3)).float().mean().item()  # untrained
    late = 0.0
    for it in range(600):
        ctx = torch.randint(0, 3, (64,)); phi = feat(ctx)
        a, _ = bg.act(phi)
        r = (a == ctx).float()                                   # rewarded action == context
        bg.train_step(phi, a, r)
        if it >= 580: late += r.mean().item() / 20
    assert late > 0.9 and late > r0                             # learns the optimal policy (chance≈0.33)
    assert 0.0 < bg.spike_rate < 1.0                            # medium-spiny neurons genuinely FIRE
    n = bg.M.shape[0]; bg.grow(32)                              # §10 growable
    assert bg.M.shape[0] == n + 32 and bg.act(feat(torch.tensor([0])))[0].shape == (1,)


def test_hippocampus_modern_hopfield_recall_and_grow():
    h = SpikingHippocampus(64, DEV, sparsity=0.1, beta=10.0, seed=1)
    pats = torch.sign(torch.randn(200, 64))
    h.store(pats)
    noisy = pats * (1 - 2 * (torch.rand_like(pats) < 0.1).float())   # 10% flipped
    rec = h.recall(noisy)
    fid = torch.cosine_similarity(rec, pats, 1).mean().item()
    assert fid > 0.9                                             # high-capacity pattern completion
    assert 0.0 < h.spike_rate < 1.0                             # CA3 neurons genuinely SPIKE on recall
    assert hasattr(h, "grow")
    h.grow(64); assert torch.cosine_similarity(h.recall(noisy), pats, 1).mean().item() > 0.9


def test_neuromod_phase_tone():
    nm = SpikingNeuromod((1,), DEV)
    assert nm.set_phase("nrem")["ach"] < nm.set_phase("wake")["ach"]   # NREM lowers ACh


def test_modules_are_sparse_fixed_neuron_growing_synapse():
    # ALL three modules must follow the same model as the cortex: fixed neurons, synapses that
    # grow then prune. density=1.0 default keeps direct constructions byte-identical (all-ones).
    for mk in (lambda d: SpikingCerebellum(32, 4, DEV, n_granule=200, seed=1, syn_density=d),
               lambda d: SpikingBasalGanglia(6, 3, DEV, n_msn=64, seed=1, syn_density=d),
               lambda d: SpikingHippocampus(48, DEV, seed=1, syn_density=d)):
        assert mk(1.0).active_synapse_count() == mk(1.0).synapse_capacity()   # default = fully wired
        m = mk(0.5); cap = m.synapse_capacity(); a0 = m.active_synapse_count()
        assert a0 < cap                                              # sparse connectome
        g = m.grow_synapses(0.3); a1 = m.active_synapse_count()
        p = m.prune_synapses(0.1); a2 = m.active_synapse_count()
        assert g > 0 and a1 > a0 and p > 0 and a2 < a1              # synapses grow then prune


def test_cerebellum_training_keeps_silent_synapses_zero():
    c = SpikingCerebellum(16, 2, DEV, n_granule=100, seed=0, syn_density=0.5)
    Wt = torch.randn(2, 16) * 0.3
    for _ in range(30):
        c.train_step(torch.randn(16, 16), torch.randn(16, 16) @ Wt.t(), eta=0.2)
    assert bool((c.W[~c._smask["W"]] == 0).all())                  # pruned/silent stay 0 through training
    n = c.M.shape[0]; c.grow(50)                                    # grow_neurons resizes the mask
    assert c._smask["M"].shape == c.M.shape and c.M.shape[0] == n + 50
