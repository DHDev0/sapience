"""
spiking_modules.py — the other four systems (§1, §2, §4, §5). The §3 cortex lives in
spiking_brain.py; here are the other four, each with its module-specific structure and
teaching signal. Four are genuinely spiking + growable; §5 is scalar modulatory glue:

  §1 SpikingCerebellum   SPIKING + growable: sparse LIF granule expansion (Golgi-gain-controlled)
                         → Purkinje readout; delta rule ΔW=−η·c·g (climbing-fibre error).
  §2 SpikingBasalGanglia SPIKING + growable: LIF MEDIUM-SPINY neurons (spike-rate code) → critic
                         + softmax actor; dopamine RPE δ=r+γV′−V trains BOTH; grow() adds MSNs.
  §4 SpikingHippocampus  SPIKING + growable: sparse dentate-gyrus separation + a modern-Hopfield
                         store (DG-key → pattern-value) with SPIKING CA3 winner-take-all recall
                         (LIF memory neurons + lateral inhibition emit real spikes).
  §5 SpikingNeuromod     scalar ACh/NE/DA/5HT tone (three-factor gate M(t) in Δw=η·M(t)·e); the
                         ACh tone scales the cortex learning rate per phase (life._learn_text).

Coupled to the cortex by ACTIVATION and REWARD only, never gradients (§6 gradient cut).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .spiking import spike
from .synapse import SynapseMaskMixin


class SpikingCerebellum(SynapseMaskMixin):
    """§1 supervised delta rule with a spiking granule code. Fixed granule NEURONS; the mossy→granule
    (M) and parallel-fibre→Purkinje (W) SYNAPSES grow/prune over the life (SynapseMaskMixin)."""

    def _synapse_matrices(self):
        return ["M", "W"]

    def __init__(self, in_dim, out_dim, device, n_granule=2000, sparsity=0.1, beta=0.9,
                 thr0=0.6, g_golgi=2.0, seed=0, syn_density=1.0):
        self.device = device; self.beta = beta; self.sparsity = sparsity
        g = torch.Generator().manual_seed(seed)
        self.M = (torch.randn(n_granule, in_dim, generator=g) / in_dim ** 0.5).to(device)  # fixed mossy→granule
        self.thr0, self.g_golgi = thr0, g_golgi                     # base threshold + Golgi gain
        self.W = torch.zeros(out_dim, n_granule, device=device)     # plastic parallel-fibre→Purkinje
        self._init_synapse_mask(syn_density)                        # fixed neurons, growable synapses

    @torch.no_grad()
    def granule(self, u, steps=4):
        """LIF granule cells fire sparsely over a few time-steps → spike-rate code. Sparsity is
        held at target by a GOLGI feedback loop rather than a frozen calibrated threshold: an
        inhibitory Golgi state integrates the granule-activity error (active − target) and
        subtracts feedback inhibition, so the active fraction self-regulates to `sparsity`
        regardless of input scale or how many granules have grown (§1 gain control)."""
        drive = u @ self.M.t()
        v = torch.zeros(u.shape[0], self.M.shape[0], device=self.device)
        golgi = torch.zeros(u.shape[0], 1, device=self.device)      # inhibitory interneuron state
        rate = 0
        for _ in range(steps):
            v = self.beta * v + drive - self.g_golgi * golgi        # feedback inhibition
            s = spike(v - self.thr0); v = v * (1 - s); rate = rate + s
            active = s.mean(dim=1, keepdim=True)                    # fraction of granules firing
            golgi = torch.clamp(golgi + (active - self.sparsity), min=0.0)   # integrate sparsity error
        r = rate / steps
        return r / (r.norm(dim=1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def predict(self, u):
        return self.granule(u) @ self.W.t()

    @torch.no_grad()
    def train_step(self, u, y, eta):
        g = self.granule(u); c = g @ self.W.t() - y                 # climbing-fibre error
        self.W -= eta * (c.t() @ g) / u.shape[0]
        self._apply_synapse_mask()                                  # keep pruned/silent synapses at 0
        return (c * c).mean()

    @torch.no_grad()
    def grow(self, add, seed=0):
        """§10 granule NEURON growth (grow_neurons): add granule cells (new random mossy→granule
        rows, new zero Purkinje columns so the output is unchanged the instant they appear). The
        Golgi loop re-balances sparsity automatically. Normally neurons stay fixed and only
        synapses develop (grow_synapses); this is the explicit lever for a bigger granule layer."""
        g = torch.Generator().manual_seed(seed + self.M.shape[0])
        self.M = torch.cat([self.M, (torch.randn(add, self.M.shape[1], generator=g) / self.M.shape[1] ** 0.5).to(self.device)])
        self.W = torch.cat([self.W, torch.zeros(self.W.shape[0], add, device=self.device)], dim=1)
        self._resize_synapse_mask()                                 # pad masks to the new granule count
        return add


class SpikingBasalGanglia(SynapseMaskMixin):
    """§2 dopamine reinforcement with spiking LIF MEDIUM-SPINY neurons. The striatal input
    projects to a population of LIF medium-spiny neurons (MSNs) that fire a spike-rate code; a
    critic reads a value V and a softmax actor reads a policy from those spikes; the dopamine
    RPE δ = r + γ·V(s′) − V(s) trains BOTH readouts (policy gradient ∇logπ = onehot(a) − π,
    scaled by the advantage δ). Genuinely spiking (LIF MSNs emit spikes). Fixed MSN NEURONS; the
    input (M), critic (w_v) and actor (W_pi) SYNAPSES grow/prune over the life (SynapseMaskMixin);
    grow_neurons() remains the explicit lever to add MSNs (identity-preserving)."""

    def _synapse_matrices(self):
        return ["M", "w_v", "W_pi"]

    def __init__(self, feat_dim, n_actions, device, n_msn=64, alpha_v=0.05, alpha_pi=0.05,
                 beta=0.9, thr=0.8, seed=0, syn_density=1.0):
        self.device = device; self.av = alpha_v; self.ap = alpha_pi
        self.beta, self.thr, self.feat_dim, self._seed = beta, thr, feat_dim, seed
        g = torch.Generator().manual_seed(seed)
        self.M = (torch.randn(n_msn, feat_dim, generator=g) / feat_dim ** 0.5).to(device)  # input→MSN
        self.w_v = torch.zeros(n_msn, device=device)                # critic readout (from spikes)
        self.W_pi = (torch.randn(n_actions, n_msn, generator=g) * 0.01).to(device)  # actor readout
        self._last_rate = 0.0
        self._init_synapse_mask(syn_density)                        # fixed MSNs, growable synapses

    @torch.no_grad()
    def msn(self, phi, steps=4):
        """LIF medium-spiny neurons fire sparsely over a few steps → a spike-rate code."""
        v = torch.zeros(phi.shape[0], self.M.shape[0], device=self.device); rate = 0
        drive = phi @ self.M.t()
        for _ in range(steps):
            v = self.beta * v + drive; s = spike(v - self.thr); v = v * (1 - s); rate = rate + s
        r = rate / steps
        self._last_rate = float(r.mean())
        return r

    @torch.no_grad()
    def act(self, phi, greedy=False):
        pi = torch.softmax((self.msn(phi) @ self.W_pi.t()).clamp(-30, 30), 1)
        a = pi.argmax(1) if greedy else torch.multinomial(pi, 1).squeeze(1)
        return a, pi

    @torch.no_grad()
    def train_step(self, phi, action, reward, phi_next=None, gamma=0.9):
        r = self.msn(phi)                                           # medium-spiny spike code
        V = r @ self.w_v
        Vn = (self.msn(phi_next) @ self.w_v) if phi_next is not None else torch.zeros_like(V)
        delta = reward + gamma * Vn - V                             # dopamine RPE (TD error)
        self.w_v += self.av * (delta.unsqueeze(1) * r).mean(0)      # critic ascent (on spikes)
        pi = torch.softmax((r @ self.W_pi.t()).clamp(-30, 30), 1)
        onehot = F.one_hot(action.long(), pi.shape[1]).float()
        self.W_pi += self.ap * ((delta.unsqueeze(1) * (onehot - pi)).t() @ r) / r.shape[0]  # actor
        self._apply_synapse_mask()                                  # keep pruned/silent synapses at 0
        return float(delta.abs().mean())

    @property
    def spike_rate(self):
        return self._last_rate

    @torch.no_grad()
    def grow(self, add, seed=0):
        """§10: add medium-spiny neurons — new random input projections; new readout weights ~0
        so V and the policy are unchanged the instant they appear (identity-preserving)."""
        g = torch.Generator().manual_seed(self._seed + self.M.shape[0] + seed)
        self.M = torch.cat([self.M, (torch.randn(add, self.feat_dim, generator=g) / self.feat_dim ** 0.5).to(self.device)])
        self.w_v = torch.cat([self.w_v, torch.zeros(add, device=self.device)])
        self.W_pi = torch.cat([self.W_pi, torch.zeros(self.W_pi.shape[0], add, device=self.device)], dim=1)
        self._resize_synapse_mask()                                 # pad masks to the new MSN count
        return add


class SpikingHippocampus(SynapseMaskMixin):
    """§4 one-shot associative memory: sparse dentate-gyrus separation + a modern-Hopfield
    (Ramsauer et al. 2020) key/value store with a spiking soft-WTA (competitive LIF) recall.

    Why this over a classic covariance-Hopfield net: the classic W += pᵀp store saturates at
    ~0.14·M patterns and then collapses into spurious attractors (the old class did exactly
    this — recall fell to chance well before the DG capacity). Storing the sparse DG code as an
    explicit KEY and the pattern as an explicit VALUE gives one-shot writes, no catastrophic
    interference, and capacity that grows with storage — while recall stays an attractor: the
    memory neurons (one per stored trace) compete via LIF divisive inhibition, which is the
    spiking realisation of the modern-Hopfield softmax (β→∞ = winner-take-all pattern
    completion). CA3 = the competing key population; CA1 = the value read-out. Growable (§10):
    grow() adds silent DG neurons (identity-preserving)."""

    # `M` (an int DG-unit count) is NOT a synapse tensor; the plastic connectome is the DG
    # separator `proj` (fixed neuron count, growable synapses via SynapseMaskMixin).
    def _synapse_matrices(self):
        return ["proj"]

    def __init__(self, n_units, device, dg_expand=4, sparsity=0.1, beta=8.0,
                 capacity=4000, thr=0.55, g_inh=1.4, seed=0, syn_density=1.0):
        self.N = n_units; self.device = device; self.a = sparsity; self.beta = beta
        self.thr, self.g_inh = thr, g_inh; self._last_rate = 0.0
        self.M = int(dg_expand * n_units)
        self._seed = seed
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(self.M, n_units, generator=g) / n_units ** 0.5).to(device)
        self.keys = torch.zeros(0, self.M, device=device)       # sparse DG codes (CA3)
        self.vals = torch.zeros(0, n_units, device=device)      # stored patterns  (CA1 target)
        self.cap = capacity
        self.n_stored = 0
        self._init_synapse_mask(syn_density)                    # fixed DG neurons, growable synapses

    @torch.no_grad()
    def separate(self, xi):
        """Dentate-gyrus pattern separation: random expansion → k-winners-take-all sparse
        {0,1} code (DG is sparse and excitatory; a decorrelated high-dim key)."""
        h = xi @ self.proj.t(); k = max(1, int(self.a * self.M))
        thr = h.topk(k, 1).values[:, -1:]
        return (h >= thr).float()

    @torch.no_grad()
    def store(self, xi):
        keys = self.separate(xi)
        self.keys = torch.cat([self.keys, keys])[-self.cap:]     # ring buffer at capacity
        self.vals = torch.cat([self.vals, xi])[-self.cap:]
        self.n_stored += xi.shape[0]

    @torch.no_grad()
    def recall(self, cue, steps=8):
        """SPIKING CA3 pattern completion: each stored trace is a LIF memory neuron driven by
        its similarity to the cue's DG key; lateral (winner-take-all) inhibition from the OTHER
        neurons' spikes suppresses weak matches over a few LIF steps until the best trace(s)
        fire; CA1 reads out the value weighted by accumulated spikes. Genuinely spiking (real
        Heaviside spikes), the spiking realisation of the modern-Hopfield attractor."""
        S = self.keys.shape[0]
        if S == 0:
            return cue
        q = self.separate(cue)
        drive = self.beta * (q @ self.keys.t()) / (q.sum(1, keepdim=True) + 1e-6)   # similarity current
        drive = drive / (drive.max(1, keepdim=True).values + 1e-6)                  # normalise → best≈1
        v = torch.zeros_like(drive); s = torch.zeros_like(drive); acc = torch.zeros_like(drive)
        for _ in range(steps):
            inh = self.g_inh * (s.sum(1, keepdim=True) - s)     # lateral WTA inhibition from other spikes
            v = 0.8 * v * (1 - s) + drive - inh                 # LIF memory-neuron membrane
            s = spike(v - self.thr)                             # CA3 neurons SPIKE (Heaviside + surrogate)
            acc = acc + s
        self._last_rate = float(acc.mean() / steps)             # mean CA3 firing over the recall
        w = acc / (acc.sum(1, keepdim=True) + 1e-6)             # spike-weighted winners
        return w @ self.vals                                    # CA1 read-out

    @property
    def spike_rate(self):
        return self._last_rate

    @torch.no_grad()
    def grow(self, add, seed=0):
        """§10 synaptogenesis: add `add` dentate-gyrus neurons with fresh random projections.
        Keys are a deterministic function of the stored patterns, so they are recomputed in the
        expanded DG space — this keeps every prior memory RECALLABLE (function preserved, the
        meaningful invariant for an associative store) while the richer DG code improves the
        separation of memories stored from now on."""
        g = torch.Generator().manual_seed(self._seed + self.M + seed)
        new_rows = (torch.randn(add, self.N, generator=g) / self.N ** 0.5).to(self.device)
        self.proj = torch.cat([self.proj, new_rows])
        self.M += add
        self._resize_synapse_mask()                   # pad the proj mask to the new DG size
        if self.vals.shape[0]:
            self.keys = self.separate(self.vals)      # re-separate stored patterns consistently
        else:
            self.keys = torch.zeros(0, self.M, device=self.device)   # advance the empty keys' DG width
        return add

    @torch.no_grad()
    def move_to(self, device):
        """Move the DG separator + the stored key/value memories to `device`."""
        super().move_to(device)
        self.keys = self.keys.to(self.device); self.vals = self.vals.to(self.device)
        return self

    @torch.no_grad()
    def _reseparate(self):
        """Recompute the stored DG keys after the separator `proj` changes, so every prior memory
        stays recallable (keys and cues must live in the SAME DG space)."""
        if self.vals.shape[0]:
            self.keys = self.separate(self.vals)
        else:
            self.keys = torch.zeros(0, self.M, device=self.device)

    @torch.no_grad()
    def grow_synapses(self, frac=0.15):
        """Densifying the DG separator changes the codes → re-separate the stored memories."""
        n = super().grow_synapses(frac)
        if n: self._reseparate()
        return n

    @torch.no_grad()
    def prune_synapses(self, frac=0.05):
        """Pruning the DG separator also changes the codes → re-separate so memories stay recallable."""
        n = super().prune_synapses(frac)
        if n: self._reseparate()
        return n


class SpikingNeuromod:
    """§5 neuromodulatory tone M(t) — the third factor of the cortex's three-factor plasticity. The
    eligibility trace itself lives in the cortex's e-prop update; here we hold only the diffuse tone
    (ACh/NE/DA/5HT), and the ACh channel is what gates the cortex's local e-prop weight update per
    phase (Δw = η·M(t)·L·e, see SpikingBrain._eprop_step). Wake = full plasticity, NREM = low."""

    def __init__(self, shape, device, tau_e=5.0):
        self.tone = dict(da=0.5, ach=1.0, ne=1.0, ht=0.5)           # dopamine/ACh/NE/5HT

    def set_phase(self, phase):
        self.tone = {"wake": dict(da=0.5, ach=1.0, ne=1.0, ht=0.5),
                     "nrem": dict(da=0.1, ach=0.15, ne=0.2, ht=0.4),
                     "rem": dict(da=0.1, ach=1.0, ne=0.0, ht=0.0)}[phase]
        return self.tone
