"""
synapse.py — shared synaptic-connectome helpers.

The whole architecture follows one rule: the NEURON population is fixed at birth (settable, can be
large), and it is the SYNAPSES that develop — childhood synaptogenesis densifies the connectome,
adolescence prunes the weak connections. These helpers implement that mechanism over any list of
plastic weight tensors + a parallel list of boolean "active-synapse" masks, so the cortex (§3) and
the §1/§2/§4 modules all grow and prune synapses the same way, with the neuron count untouched.

A mask entry True = an active synapse; False = a silent (never-wired or pruned) connection whose
weight is held at zero. `apply_masks` must be called after every weight update so pruned/inactive
synapses stay silent (they do not regrow on their own).
"""
import torch


@torch.no_grad()
def init_masks(weights, density, seed=1234):
    """Seed a sparse connectome: a `density` fraction of each matrix's connections are active, the
    rest silent (zeroed in place). density>=1 → fully connected. Returns the list of masks."""
    density = max(0.02, min(1.0, float(density)))
    if density >= 1.0:
        return [torch.ones_like(w, dtype=torch.bool) for w in weights]
    g = torch.Generator(device="cpu").manual_seed(seed)
    masks = []
    for w in weights:
        m = (torch.rand(w.shape, generator=g) < density).to(w.device)
        w.mul_(m)
        masks.append(m)
    return masks


@torch.no_grad()
def apply_masks(weights, masks):
    """Re-zero silent synapses after a weight update, so pruned/inactive connections stay silent."""
    for w, m in zip(weights, masks):
        if w.shape == m.shape:
            w.mul_(m)


@torch.no_grad()
def grow_synapses(weights, masks, frac, fanins=None):
    """Synaptogenesis: activate `frac` of the currently-silent connections with fresh fan-in-scaled
    weights. `fanins[i]` gives the TRUE fan-in for weights[i] (needed for 1-D value vectors whose
    last dim is nnz, not the fan-in). The neuron count is unchanged. Returns the number added."""
    n = 0
    for i, (w, m) in enumerate(zip(weights, masks)):
        inactive = (~m).flatten().nonzero(as_tuple=False).flatten()
        k = int(frac * inactive.numel())
        if k <= 0:
            continue
        pick = inactive[torch.randperm(inactive.numel(), device=inactive.device)[:k]]
        fm = m.flatten(); fm[pick] = True; m.copy_(fm.view_as(m))
        fan = (fanins[i] if fanins is not None else float(w.shape[-1]))
        scale = max(1.0, float(fan)) ** 0.5
        fw = w.flatten(); fw[pick] = torch.randn(k, device=w.device) / scale * 0.5; w.copy_(fw.view_as(w))
        n += k
    return n


@torch.no_grad()
def prune_synapses(weights, masks, frac):
    """Silence the weakest `frac` of active synapses PER MATRIX (mask-persistent). A per-matrix rank
    cut — not one pooled threshold — so matrices at different weight scales each lose their own
    weakest frac (pooling would over-prune the small-scale matrix). kthvalue avoids torch.quantile's
    2^24 limit. NEURON count untouched. Returns the number cut."""
    n = 0
    for w, m in zip(weights, masks):
        if w.shape != m.shape:
            continue
        vals = w[m].abs().flatten()
        if vals.numel() < 8:
            continue
        k = max(1, min(vals.numel(), int(frac * vals.numel())))
        thr = vals.kthvalue(k).values
        cut = m & (w.abs() <= thr)
        m &= ~cut; w.mul_(m); n += int(cut.sum())
    return n


def active_count(masks):
    """Number of active synapses across the masks (the connectome that carries signal)."""
    return int(sum(int(m.sum()) for m in masks)) if masks else 0


def capacity(weights):
    """Total possible connections (active + silent) across the weight matrices."""
    return int(sum(w.numel() for w in weights))


class SynapseMaskMixin:
    """Name-keyed synaptic development for the plain-tensor modules (§1 cerebellum, §2 basal
    ganglia, §4 hippocampus). Unlike the cortex (whose nn.Linear weights keep identity), these
    modules REASSIGN their weight tensors (torch.cat on grow), so masks are keyed by ATTRIBUTE
    NAME and re-fetched each call. Same contract as the cortex: fixed neurons, synapses grow/prune.

    A subclass declares `_synapse_matrices()` -> list of attribute names holding plastic weight
    tensors, and calls `self._init_synapse_mask(density)` at the end of __init__."""

    def _synapse_matrices(self):
        return []

    def _syn_weights(self):
        return [getattr(self, n) for n in self._synapse_matrices()]

    @torch.no_grad()
    def _init_synapse_mask(self, density=1.0, seed=1234):
        """Seed the connectome. density>=1 → all-ones (byte-identical: every op below is a no-op)."""
        self.syn_density = float(density)
        if not hasattr(self, "grow_syn_frac"):
            self.grow_syn_frac, self.prune_frac = 0.15, 0.05    # per-region synaptic growth/prune rates
        names = self._synapse_matrices()
        density = max(0.02, min(1.0, float(density)))
        if density >= 1.0:
            self._smask = {n: torch.ones_like(getattr(self, n), dtype=torch.bool) for n in names}
            return
        g = torch.Generator(device="cpu").manual_seed(seed)
        self._smask = {}
        for n in names:
            w = getattr(self, n)
            m = (torch.rand(w.shape, generator=g) < density).to(w.device)
            w.mul_(m); self._smask[n] = m

    @torch.no_grad()
    def _apply_synapse_mask(self):
        """Re-zero silent synapses after a weight update (pruned/inactive stay silent)."""
        for n, m in getattr(self, "_smask", {}).items():
            w = getattr(self, n)
            if w.shape == m.shape:
                w.mul_(m)

    def synapse_capacity(self):
        return capacity(self._syn_weights())

    def parameter_count(self):
        """Number of trainable parameters (weight elements) in this region's connectome."""
        return capacity(self._syn_weights())

    @torch.no_grad()
    def move_to(self, device):
        """Move this region's connectome (weight matrices + masks) to `device` and record it, so a
        part can live on a different device than the rest (life converts at the boundaries)."""
        device = torch.device(device)
        for n in self._synapse_matrices():
            setattr(self, n, getattr(self, n).to(device))
        for n, m in getattr(self, "_smask", {}).items():
            self._smask[n] = m.to(device)
        self.device = device
        return self

    def active_synapse_count(self):
        sm = getattr(self, "_smask", None)
        if not sm:
            return self.synapse_capacity()
        return int(sum(int(m.sum()) for m in sm.values()))

    @torch.no_grad()
    def grow_synapses(self, frac=0.15):
        """Activate `frac` of silent connections with fresh fan-in-scaled weights. Neurons fixed."""
        sm = getattr(self, "_smask", None)
        if not sm:
            return 0
        n = 0
        for name, m in sm.items():
            w = getattr(self, name)
            if w.shape != m.shape:
                continue
            inactive = (~m).flatten().nonzero(as_tuple=False).flatten()
            k = int(frac * inactive.numel())
            if k <= 0:
                continue
            pick = inactive[torch.randperm(inactive.numel(), device=inactive.device)[:k]]
            fm = m.flatten(); fm[pick] = True; m.copy_(fm.view_as(m))
            fanin = max(1.0, float(w.shape[-1])) ** 0.5
            fw = w.flatten(); fw[pick] = torch.randn(k, device=w.device) / fanin * 0.5; w.copy_(fw.view_as(w))
            n += k
        return n

    @torch.no_grad()
    def prune_synapses(self, frac=0.05):
        """Silence the weakest `frac` of active synapses PER MATRIX (mask-persistent). Neurons fixed.
        Per-matrix rank cut (kthvalue, no 2^24 limit) so different-scale matrices each lose frac."""
        sm = getattr(self, "_smask", None)
        if not sm:
            return 0
        n = 0
        for name, m in sm.items():
            w = getattr(self, name)
            if w.shape != m.shape:
                continue
            vals = w[m].abs().flatten()
            if vals.numel() < 8:
                continue
            k = max(1, min(vals.numel(), int(frac * vals.numel())))
            thr = vals.kthvalue(k).values
            cut = m & (w.abs() <= thr)
            m &= ~cut; w.mul_(m); n += int(cut.sum())
        return n

    @torch.no_grad()
    def _resize_synapse_mask(self):
        """After a NEURON grow reassigns a larger tensor, pad its mask (old kept, new connections
        active) without touching weights — identity-safe. Handles 1-D (w_v) and 2-D matrices."""
        sm = getattr(self, "_smask", None)
        if not sm:
            return
        for name, m in list(sm.items()):
            w = getattr(self, name)
            if w.shape == m.shape:
                continue
            nm = torch.ones_like(w, dtype=torch.bool)
            slc = tuple(slice(0, s) for s in m.shape)
            nm[slc] = m
            sm[name] = nm
