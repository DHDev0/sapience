"""
spiking.py — the spiking substrate (no external SNN library).

The field guide's substrate is spiking (§3.7 two-compartment pyramidal neurons with
PV/SOM/VIP interneurons; §11.3 spiking models). This builds it natively:

  spike()              Heaviside firing with a fast-sigmoid SURROGATE gradient, so a
                       spiking network can be trained by error signals — which §3.5 shows
                       is predictive coding in the β→0 limit.
  LIFCell              leaky integrate-and-fire: membrane leaks, integrates input, fires,
                       resets. The basic neuron of every module.
  TwoCompartmentLIF    §3.7 cortical pyramidal cell: a SOMA (basal/feedforward drive) and
                       an APICAL dendrite (top-down = the credit-assignment error). Firing
                       is somatic; the apical modulates it. Recurrent in time.
  All layers GROW (§10 synaptogenesis): new neurons are added with ~zero output weight so
  the function is preserved the instant capacity appears.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _SurrGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):                       # x = membrane − threshold
        ctx.save_for_backward(x)
        return (x >= 0).float()                # the spike (0/1)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (10.0 * x.abs() + 1.0) ** 2  # fast-sigmoid surrogate derivative


spike = _SurrGrad.apply


class _SparseConnMM(torch.autograd.Function):
    """y = (W @ s.t()).t() for a CSR weight W(out,in) with values `val`, computed so the BACKWARD
    stays O(nnz·B) instead of densifying to O(out·in).

    Both directions are pure gather/scatter (index_add), O(nnz·B) — NO torch.sparse.mm. This is
    deliberate: (1) sparse.mm's own autograd densifies a full (out×in) value-gradient (~250 GB at
    500k neurons → OOM), which the sampled `grad_val` here avoids; and (2) rocSPARSE (ROCm) is flaky
    — its CSR spmm intermittently aborts the process with "Invalid device argument" and has no bf16
    kernel — whereas index_add/gather are rock-stable at scale on both CUDA and ROCm. So a
    half-million-neuron connectome is both trainable in RAM and stable on this GPU. Forced to fp32
    (autocast off) because the connectome is where precision matters; dense ops still autocast to bf16."""

    _CHUNK = 1 << 27                                            # cap the (rows, nnz) buffer at ~128M elems

    @staticmethod
    def forward(ctx, val, crow, col, row_idx, s, out_dim):
        s_dtype = s.dtype
        vf, sf, cl = val.float(), s.float(), col.long()
        B, nnz = sf.shape[0], cl.numel()
        ch = max(1, min(B, _SparseConnMM._CHUNK // max(1, nnz)))  # bound the (chunk, nnz) intermediate
        y = torch.empty(B, out_dim, device=sf.device, dtype=sf.dtype)
        for i in range(0, B, ch):                              # chunk the batch (B·T for the input proj)
            sc = sf[i:i + ch]                                  # y[b,row]+= val·s[b,col]: gather then scatter
            contrib = vf.unsqueeze(0) * sc[:, cl]              # (chunk, nnz), transient
            y[i:i + ch] = torch.zeros(sc.shape[0], out_dim, device=sf.device, dtype=sf.dtype).index_add_(1, row_idx, contrib)
        ctx.save_for_backward(vf, col, row_idx, sf)
        ctx.s_dtype = s_dtype
        return y.to(s_dtype)

    @staticmethod
    def backward(ctx, gy):
        val, col, row_idx, s = ctx.saved_tensors               # already fp32
        cl = col.long(); gyf = gy.contiguous().float()
        B, nnz = s.shape[0], cl.numel()
        ch = max(1, min(B, _SparseConnMM._CHUNK // max(1, nnz)))
        grad_val = torch.zeros_like(val); grad_s = torch.empty_like(s)
        for i in range(0, B, ch):
            gy_e = gyf[i:i + ch][:, row_idx]                   # (chunk, nnz) — grad_out at each edge's row
            grad_val += (gy_e * s[i:i + ch][:, cl]).sum(0)     # SDDMM sample: no dense out×in
            grad_s[i:i + ch] = torch.zeros(gy_e.shape[0], s.shape[1], device=s.device).index_add_(1, cl, val.unsqueeze(0) * gy_e)
        return grad_val, None, None, None, grad_s.to(ctx.s_dtype), None


def _row_index(crow):
    """The output-row id of every nnz edge, from CSR row pointers (for the gather/scatter backward)."""
    lengths = (crow[1:] - crow[:-1]).long()
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=crow.device), lengths)


@torch.no_grad()
def _seed_csr(rows, cols, fanin, seed):
    """Seed a fan-in connectome: `fanin` random pre-neuron columns per row from [0,cols). Columns
    MAY repeat within a row (rare at fanin<<cols); the gather/scatter forward just SUMS parallel
    edges, so uniqueness is not required — which makes this O(rows·fanin), instant even at H=1e6
    (the old unique-per-row top-k over an H×H random matrix was the construction bottleneck at
    scale). Returns (crow, col, fanin)."""
    fanin = int(min(fanin, cols))
    gen = torch.Generator().manual_seed(seed)
    col = torch.randint(0, cols, (rows * fanin,), generator=gen, dtype=torch.int32)
    crow = torch.arange(0, rows * fanin + 1, fanin, dtype=torch.int32)
    return crow, col, fanin


class SparseLIFCell(nn.Module):
    """Leaky integrate-and-fire layer with a SPARSE recurrent connectome — the same dynamics as
    LIFCell (v ← β·v·(1−s) + Win x + Wrec s ; s = spike(v−θ)) but Wrec (and Win for deep layers
    whose input width is large) is stored as a fixed CSR wiring (int32 crow/col buffers) + a dense
    values Parameter + a 1-D active-synapse mask. This is what lets the NEURON count reach the
    hundreds of thousands within RAM: memory is O(H·fanin), not O(H²). The values are a normal
    dense Parameter, so the same Adam optimizes them and autograd returns a dense (nnz,) gradient
    (verified) — no H² densification. Synapses grow/prune by flipping mask bits within the fixed
    superset; the neuron count is fixed (grow() is a rare structural rebuild)."""

    def __init__(self, in_dim, hid, beta=0.9, thr=1.0, rec_fanin=64, in_fanin=64,
                 sparse_in=False, syn_density=1.0, seed=0):
        super().__init__()
        self.beta, self.thr, self.hid, self.in_dim = beta, thr, hid, in_dim
        self.sparse_in = bool(sparse_in)
        self.rec_fanin, self.in_fanin = rec_fanin, in_fanin
        # --- recurrent connectome (always sparse) ---
        crow, col, f = _seed_csr(hid, hid, rec_fanin, seed + 1)
        self.rec_fanin = f
        self.register_buffer("rec_crow", crow)
        self.register_buffer("rec_col", col)
        self.register_buffer("rec_row", _row_index(crow))          # edge→row (gather/scatter backward)
        nnz = col.numel()
        self.rec_val = nn.Parameter(torch.randn(nnz) / max(1.0, f) ** 0.5)
        self.register_buffer("rec_mask", self._seed_mask(nnz, syn_density, seed + 7))
        self.rec_val.data.mul_(self.rec_mask)
        # --- input projection: dense for a small in_dim (layer 0), else sparse ---
        if self.sparse_in:
            icrow, icol, fi = _seed_csr(hid, in_dim, in_fanin, seed + 2)
            self.in_fanin = fi
            self.register_buffer("in_crow", icrow)
            self.register_buffer("in_col", icol)
            self.register_buffer("in_row", _row_index(icrow))
            innz = icol.numel()
            self.in_val = nn.Parameter(torch.randn(innz) / max(1.0, fi) ** 0.5)
            self.register_buffer("in_mask", self._seed_mask(innz, syn_density, seed + 9))
            self.in_val.data.mul_(self.in_mask)
            self.in_bias = nn.Parameter(torch.zeros(hid))
        else:
            self.Win = nn.Linear(in_dim, hid)

    @staticmethod
    def _seed_mask(nnz, density, seed):
        if density >= 1.0:
            return torch.ones(nnz, dtype=torch.bool)
        g = torch.Generator().manual_seed(seed)
        return (torch.rand(nnz, generator=g) < max(0.02, min(1.0, density)))

    def _rec(self, s):
        """Recurrent drive Wrec·s via the O(nnz·B)-backward custom op (no dense H² gradient)."""
        return _SparseConnMM.apply(self.rec_val * self.eff_rec_mask(), self.rec_crow, self.rec_col,
                                   self.rec_row, s, self.hid)

    def eff_rec_mask(self):
        """§17 rec_mask AND the laminar adjacency mask when laminar is on (else just rec_mask → identical)."""
        lm = getattr(self, "lam_rec_mask", None)
        return (self.rec_mask * lm) if lm is not None else self.rec_mask

    def eff_in_mask(self):
        lm = getattr(self, "lam_in_mask", None)
        return (self.in_mask * lm) if lm is not None else self.in_mask

    def _in_proj(self, x):
        """(B,T,in) → (B,T,hid). Dense Win for layer 0; a single sparse projection over all time else."""
        if not self.sparse_in:
            return self.Win(x)
        B, T, _ = x.shape
        flat = x.reshape(-1, self.in_dim)                      # (B*T, in)
        y = _SparseConnMM.apply(self.in_val * self.eff_in_mask(), self.in_crow, self.in_col,
                                self.in_row, flat, self.hid)
        return (y + self.in_bias).view(B, T, self.hid)

    def init_state(self, B, device, dtype=torch.float32):
        z = torch.zeros(B, self.hid, device=device, dtype=dtype)
        return (z, z.clone())

    def forward(self, x, state):
        v, s = state
        v = self.beta * v * (1.0 - s) + self._in_proj(x.unsqueeze(1))[:, 0] + self._rec(s)
        s = spike(v - self.thr)
        return s, (v, s)

    def run_seq(self, x, state, stp=None, stp_layer=0):
        v, s = state
        pre = self._in_proj(x)                                  # (B,T,hid) vectorized input
        spikes, mems = [], []
        for t in range(x.shape[1]):
            zt = s if (stp is None or not stp.on) else s * stp.transmit(stp_layer, s)   # §17 STP presynaptic gain (g≡1 off)
            v = self.beta * v * (1.0 - s) + pre[:, t] + self._rec(zt)
            s = spike(v - self.thr)
            spikes.append(s); mems.append(v)
        return torch.stack(spikes, 1), torch.stack(mems, 1), (v, s)

    @torch.no_grad()
    def grow(self, add):
        """Rare structural NEURON growth: append `add` post-neurons that receive fan-in from the
        existing population but have NO outgoing edges (and the head zeros their read-out), so the
        function is preserved. Neurons are normally fixed — synapses are what develop."""
        old = self.hid; new = old + add
        crow2, col2, _ = _seed_csr(add, old, self.rec_fanin, int(self.rec_crow.numel()))
        self.rec_col = torch.cat([self.rec_col, col2.to(self.rec_col.device)])
        self.rec_crow = torch.cat([self.rec_crow, (self.rec_crow[-1] + crow2[1:]).to(self.rec_crow.device)])
        addv = (torch.randn(col2.numel(), device=self.rec_val.device) / max(1.0, self.rec_fanin) ** 0.5)
        self.rec_val = nn.Parameter(torch.cat([self.rec_val.data, addv]))
        self.rec_mask = torch.cat([self.rec_mask, torch.ones(col2.numel(), dtype=torch.bool, device=self.rec_mask.device)])
        self.rec_row = _row_index(self.rec_crow)
        if self.sparse_in:
            icrow2, icol2, _ = _seed_csr(add, self.in_dim, self.in_fanin, int(self.in_crow.numel()) + 3)
            self.in_col = torch.cat([self.in_col, icol2.to(self.in_col.device)])
            self.in_crow = torch.cat([self.in_crow, (self.in_crow[-1] + icrow2[1:]).to(self.in_crow.device)])
            iaddv = (torch.randn(icol2.numel(), device=self.in_val.device) / max(1.0, self.in_fanin) ** 0.5)
            self.in_val = nn.Parameter(torch.cat([self.in_val.data, iaddv]))
            self.in_mask = torch.cat([self.in_mask, torch.ones(icol2.numel(), dtype=torch.bool, device=self.in_mask.device)])
            self.in_bias = nn.Parameter(torch.cat([self.in_bias.data, torch.zeros(add, device=self.in_bias.device)]))
            self.in_row = _row_index(self.in_crow)
        else:
            nWin = nn.Linear(self.in_dim, new).to(self.Win.weight.device, self.Win.weight.dtype)
            nWin.weight.zero_(); nWin.weight[:old] = self.Win.weight; nWin.bias.zero_(); nWin.bias[:old] = self.Win.bias
            self.Win = nWin
        self.hid = new
        return add


class SparseALIFCell(SparseLIFCell):
    """SparseLIFCell + the ALIF adaptive threshold (per-neuron trace a; θ_eff = θ₀ + β_a·a). The
    adaptation is a per-NEURON state, orthogonal to the sparse connectivity."""

    def __init__(self, in_dim, hid, beta=0.9, thr=1.0, rho=0.97, beta_adapt=1.2, **kw):
        super().__init__(in_dim, hid, beta=beta, thr=thr, **kw)
        self.thr0 = thr; self.rho, self.beta_adapt = rho, beta_adapt

    def init_state(self, B, device, dtype=torch.float32):
        z = torch.zeros(B, self.hid, device=device, dtype=dtype)
        return (z, z.clone(), z.clone())

    def forward(self, x, state):
        v, s, a = state
        v = self.beta * v * (1.0 - s) + self._in_proj(x.unsqueeze(1))[:, 0] + self._rec(s)
        a = self.rho * a + s
        s = spike(v - (self.thr0 + self.beta_adapt * a))
        return s, (v, s, a)

    def run_seq(self, x, state, stp=None, stp_layer=0):
        v, s, a = state
        pre = self._in_proj(x)
        spikes, mems = [], []
        for t in range(x.shape[1]):
            zt = s if (stp is None or not stp.on) else s * stp.transmit(stp_layer, s)   # §17 STP presynaptic gain (g≡1 off)
            v = self.beta * v * (1.0 - s) + pre[:, t] + self._rec(zt)
            a = self.rho * a + s
            s = spike(v - (self.thr0 + self.beta_adapt * a))
            spikes.append(s); mems.append(v)
        return torch.stack(spikes, 1), torch.stack(mems, 1), (v, s, a)


class LIFCell(nn.Module):
    """Leaky integrate-and-fire recurrent layer: v ← β·v·(1−s) + W x + R s ; s = spike(v−θ)."""

    def __init__(self, in_dim, hid, beta=0.9, thr=1.0):
        super().__init__()
        self.Win = nn.Linear(in_dim, hid)
        self.Wrec = nn.Linear(hid, hid, bias=False)
        self.beta, self.thr, self.hid, self.in_dim = beta, thr, hid, in_dim

    def init_state(self, B, device, dtype=torch.float32):
        z = torch.zeros(B, self.hid, device=device, dtype=dtype)
        return (z, z.clone())

    def forward(self, x, state):
        v, s = state
        v = self.beta * v * (1.0 - s) + self.Win(x) + self.Wrec(s)
        s = spike(v - self.thr)
        return s, (v, s)

    def run_seq(self, x, state, stp=None, stp_layer=0):
        """Run the whole (B,T,in) sequence. The input projection Win(x) — the bulk of the
        FLOPs — is computed for ALL timesteps in ONE matmul; only the recurrence Wrec(s)
        stays in the Python loop (it must, it depends on the previous spike). Mathematically
        identical to stepping forward() T times. Returns (spikes, membranes, final_state)."""
        v, s = state
        pre = self.Win(x)                              # (B,T,hid) — vectorized over time
        spikes, mems = [], []
        for t in range(x.shape[1]):
            zt = s if (stp is None or not stp.on) else s * stp.transmit(stp_layer, s)   # §17 STP presynaptic gain (g≡1 off)
            _rw = getattr(self, "lam_rec_w", None)                                       # §17 laminar dense adjacency (test nets)
            _rec = (zt @ (self.Wrec.weight * _rw).t()) if _rw is not None else self.Wrec(zt)
            v = self.beta * v * (1.0 - s) + pre[:, t] + _rec
            s = spike(v - self.thr)
            spikes.append(s); mems.append(v)
        return torch.stack(spikes, 1), torch.stack(mems, 1), (v, s)

    @torch.no_grad()
    def grow(self, add):
        dev = self.Win.weight.device; dt = self.Win.weight.dtype
        old = self.hid; new = old + add
        # widen Win (out), Wrec (both dims); new output weights ~0 (identity-preserving)
        nWin = nn.Linear(self.in_dim, new).to(dev, dt)
        nWin.weight.zero_(); nWin.weight[:old] = self.Win.weight; nWin.bias.zero_(); nWin.bias[:old] = self.Win.bias
        nWrec = nn.Linear(new, new, bias=False).to(dev, dt)
        nWrec.weight.zero_(); nWrec.weight[:old, :old] = self.Wrec.weight
        self.Win, self.Wrec, self.hid = nWin, nWrec, new
        return add


class ALIFCell(nn.Module):
    """Adaptive LIF (LSNN, Bellec et al. 2018): the firing threshold ADAPTS via a slow
    per-neuron trace a, θ_eff = θ₀ + β_a·a, a ← ρ·a + s. That slow state is a working
    memory over hundreds of steps that a plain LIF lacks — the key lever that closes much
    of the gap to a rate GRU while the network stays genuinely spiking and growable.
    The analog membrane v is kept in the state so a readout can be taken from the graded
    potential (lossless) rather than the binary spike (lossy)."""

    def __init__(self, in_dim, hid, beta=0.9, thr=1.0, rho=0.97, beta_adapt=1.2):
        super().__init__()
        self.Win = nn.Linear(in_dim, hid)
        self.Wrec = nn.Linear(hid, hid, bias=False)
        self.beta, self.thr0, self.hid, self.in_dim = beta, thr, hid, in_dim
        self.rho, self.beta_adapt = rho, beta_adapt

    def init_state(self, B, device, dtype=torch.float32):
        z = torch.zeros(B, self.hid, device=device, dtype=dtype)
        return (z, z.clone(), z.clone())       # v (membrane), s (spike), a (adaptation)

    def forward(self, x, state):
        v, s, a = state
        v = self.beta * v * (1.0 - s) + self.Win(x) + self.Wrec(s)
        a = self.rho * a + s                    # slow adaptation trace (working memory)
        thr = self.thr0 + self.beta_adapt * a
        s = spike(v - thr)
        return s, (v, s, a)

    def run_seq(self, x, state, stp=None, stp_layer=0):
        """Sequence run with the input projection vectorized over time (see LIFCell.run_seq)."""
        v, s, a = state
        pre = self.Win(x)
        spikes, mems = [], []
        for t in range(x.shape[1]):
            zt = s if (stp is None or not stp.on) else s * stp.transmit(stp_layer, s)   # §17 STP presynaptic gain (g≡1 off)
            _rw = getattr(self, "lam_rec_w", None)                                       # §17 laminar dense adjacency (test nets)
            _rec = (zt @ (self.Wrec.weight * _rw).t()) if _rw is not None else self.Wrec(zt)
            v = self.beta * v * (1.0 - s) + pre[:, t] + _rec
            a = self.rho * a + s
            s = spike(v - (self.thr0 + self.beta_adapt * a))
            spikes.append(s); mems.append(v)
        return torch.stack(spikes, 1), torch.stack(mems, 1), (v, s, a)

    @torch.no_grad()
    def grow(self, add):
        dev = self.Win.weight.device; dt = self.Win.weight.dtype
        old = self.hid; new = old + add
        nWin = nn.Linear(self.in_dim, new).to(dev, dt)
        nWin.weight.zero_(); nWin.weight[:old] = self.Win.weight; nWin.bias.zero_(); nWin.bias[:old] = self.Win.bias
        nWrec = nn.Linear(new, new, bias=False).to(dev, dt)
        nWrec.weight.zero_(); nWrec.weight[:old, :old] = self.Wrec.weight
        self.Win, self.Wrec, self.hid = nWin, nWrec, new
        return add


class TwoCompartmentLIF(nn.Module):
    """§3.7 pyramidal cell: SOMA (basal feedforward + recurrent) fires; APICAL dendrite
    carries top-down drive (the error/context). v_apical gates somatic firing."""

    def __init__(self, in_dim, hid, top_dim=0, beta=0.9, thr=1.0, apical_gain=0.5):
        super().__init__()
        self.Wb = nn.Linear(in_dim, hid)                # basal (feedforward)
        self.Wr = nn.Linear(hid, hid, bias=False)       # recurrent
        self.Wa = nn.Linear(top_dim, hid, bias=False) if top_dim else None  # apical (top-down)
        self.beta, self.thr, self.hid, self.in_dim = beta, thr, hid, in_dim
        self.top_dim, self.g_ap = top_dim, apical_gain

    def init_state(self, B, device, dtype=torch.float32):
        z = torch.zeros(B, self.hid, device=device, dtype=dtype)
        return (z, z.clone())

    def forward(self, x, state, top=None):
        v, s = state
        drive = self.Wb(x) + self.Wr(s)
        if self.Wa is not None and top is not None:
            drive = drive + self.g_ap * self.Wa(top)    # apical top-down (§3.7)
        v = self.beta * v * (1.0 - s) + drive
        s = spike(v - self.thr)
        return s, (v, s)

    @torch.no_grad()
    def grow(self, add):
        dev = self.Wb.weight.device; dt = self.Wb.weight.dtype
        old = self.hid; new = old + add
        nWb = nn.Linear(self.in_dim, new).to(dev, dt)
        nWb.weight.zero_(); nWb.weight[:old] = self.Wb.weight; nWb.bias.zero_(); nWb.bias[:old] = self.Wb.bias
        nWr = nn.Linear(new, new, bias=False).to(dev, dt)
        nWr.weight.zero_(); nWr.weight[:old, :old] = self.Wr.weight
        self.Wb, self.Wr = nWb, nWr
        if self.Wa is not None:
            nWa = nn.Linear(self.top_dim, new, bias=False).to(dev, dt)
            nWa.weight.zero_(); nWa.weight[:old] = self.Wa.weight
            self.Wa = nWa
        self.hid = new
        return add
