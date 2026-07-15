"""§17 · LaminarMicrocircuit — the canonical Douglas–Martin / Bastos cortical column.

The neocortex is a 6-layer laminar SHEET with stereotyped inter-laminar wiring, NOT a flat
recurrent pool. Today SparseLIFCell seeds rec_col/rec_row uniformly random over [0,hid): a flat
pool. This module carves that flat H×fanin CSR into the CANONICAL microcircuit by partitioning
each cortical layer's `hid` neurons into three sublaminae and constraining the recurrent + input
connectomes to the allowed inter-laminar projection pattern:

  L4  (l=0)  granular  — thalamo-recipient INPUT relay (external drive lands HERE)
  L2/3 (l=1) supragranular — associative RECURRENCE + feedforward output stream (gamma)
  L5/6 (l=2) infragranular — subcortical OUTPUT + feedback to lower areas (alpha/beta)

Canonical allowed adjacency A[pre,post] (Douglas & Martin 2004; Bastos et al. 2012):
  L4->L4, L4->L2/3            (feedforward into the column)
  L2/3->L2/3, L2/3->L5/6      (associative recurrence + feedforward down)
  L5/6->L5/6                  (deep recurrence)
  L5/6->L4, L5/6->L2/3        (feedback, gated by allow_fb)
  FORBIDDEN by default: L2/3->L4, L4->L5/6.

The constraint is realized as a multiplicative BOOLEAN mask over the existing sparse CSR — it adds
ZERO parameters and is learned THROUGH by the same e-prop rule (forbidden lamina-pair synapses get
exactly zero eligibility-driven update and stay frozen at init). The effective connectome becomes
    W_rec_eff[e] = rec_val[e] · rec_mask[e] · A[l(col_e), l(row_e)]
(edge e connects pre=col_e → post=row_e; A indexed [pre,post]).

INTERCONNECTION (mandatory): laminar supplies `lam_apical_gain` (1.0 for L2/3+L5/6, apical_l4_gain
for L4) that multiplies the two_compartment apical feedback g_ap·ap[l] in _eprop_step, so the
VIP→SOM-gated top-down error lands on the APICAL tufts of L2/3 & L5 cells and SPARES the granular
L4 input relay — exactly the Bastos predictive-coding picture. Laminar and the interneuron circuit
become ONE microcircuit: soma-driven L4 input, apical-driven L2/3+L5 error.

DEVICE/DTYPE/FSDP: every per-edge / per-neuron tensor is built by INDEXING the cell's OWN sharded
rec_row/rec_col buffers, so lam_rec_mask has the SAME shape+nnz-partition as rec_val and SHARDS WITH
IT under FSDP2; per-neuron tensors (lamina, lam_apical_gain, lam_rec_fanin) share the layer's neuron
partition. The only replicated state is the tiny 3×3 A matrix + the fraction/gain floats (a small
controller). Masks are DERIVED (not pickled) — rebuilt deterministically after load/grow.

SCALE: at hidden=128000, fanin 32 → nnz=4.096M/layer. lam_rec_mask bool = 4.1 MB/layer, lamina
int8 = 128 KB, lam_rec_fanin f32 = 0.5 MB, lam_apical_gain (model dtype) = 0.25 MB → ≈10–18 MB for
the whole 256k net. O(nnz)+O(hid), NO dense O(N²) on the sparse path (dense lam_rec_w exists ONLY
for the small <8192 test nets). Width-invariance: the recurrent update is divided by the per-neuron
EFFECTIVE fan-in (lam_rec_fanin), so a laminar-thinned neuron is not under-driven — the p=1
width-invariant descent is preserved and no fan-in starvation is reintroduced at width.

Refs: Douglas & Martin, Annu Rev Neurosci 2004; Bastos et al., Neuron 2012 (canonical microcircuits
for predictive coding); Haeusler & Maass, Cereb Cortex 2007 (laminar > size-matched random recurrent
— the earns-keep hypothesis).
"""
from __future__ import annotations
import torch

# lamina labels
L4, L23, L56 = 0, 1, 2


class LaminarMicrocircuit:
    _KEYS = ("on", "frac_L4", "frac_L23", "frac_L56", "allow_fb",
             "input_to_l23", "apical_l4_gain", "strict")

    def __init__(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype                      # model dtype (for lam_apical_gain / dense in-gate)
        self.on = False                         # opt-in toggle (DEFAULT OFF; verify before defaulting on)
        # contiguous fractional laminar split (normalised at assignment time)
        self.frac_L4 = 0.25
        self.frac_L23 = 0.45
        self.frac_L56 = 0.30
        self.allow_fb = True                    # L5/6 → L4,L2/3 feedback allowed
        self.input_to_l23 = 0.0                 # >0 also routes external drive to L2/3 (else L4-only)
        self.apical_l4_gain = 0.05              # apical top-down gain onto granular L4 (spared)
        self.strict = True                      # enforce the two forbidden pairs (L2/3→L4, L4→L5/6)
        # last-observed metrics (surfaced to state())
        self._rate_L4 = 0.0
        self._rate_L23 = 0.0
        self._rate_L56 = 0.0
        self._forbidden = 0.0                   # fraction of active edges masked out by A

    # ---- controller ------------------------------------------------- #
    def set_params(self, **kw):
        applied = {}
        for k, v in kw.items():
            if k not in self._KEYS:
                continue
            cur = getattr(self, k, None)
            if isinstance(cur, bool):                          # string 'false'/'0'/'off' disables, not enables
                v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "off", "no", "")
            elif cur is not None:
                v = float(v)
            setattr(self, k, v); applied[k] = getattr(self, k)
        return applied

    def state(self):
        return dict(on=self.on,
                    frac_L4=round(self.frac_L4, 3), frac_L23=round(self.frac_L23, 3),
                    frac_L56=round(self.frac_L56, 3), allow_fb=bool(self.allow_fb),
                    input_to_l23=round(self.input_to_l23, 3),
                    apical_l4_gain=round(self.apical_l4_gain, 3), strict=bool(self.strict),
                    rate_L4=round(self._rate_L4, 4), rate_L23=round(self._rate_L23, 4),
                    rate_L56=round(self._rate_L56, 4), n_forbidden_frac=round(self._forbidden, 4))

    # ---- laminar assignment + adjacency ----------------------------- #
    def _fracs(self):
        f4, f23, f56 = max(0.0, self.frac_L4), max(0.0, self.frac_L23), max(0.0, self.frac_L56)
        tot = f4 + f23 + f56
        if tot <= 0:
            return 0.25, 0.45, 0.30
        return f4 / tot, f23 / tot, f56 / tot

    def _assign_lamina(self, hid, device, existing=None):
        """int8 lamina label per neuron by CONTIGUOUS fractional index blocks. If `existing` labels
        are supplied (a mid-life grow), they are PRESERVED for [0:len(existing)] and only the appended
        indices get fresh labels by the fractional rule → grow() never relabels an existing neuron
        (identity-safe for e-prop + the head)."""
        f4, f23, _ = self._fracs()
        idx = torch.arange(hid, device=device)
        b4 = f4 * hid
        b23 = (f4 + f23) * hid
        lam = torch.full((hid,), L56, dtype=torch.int8, device=device)
        lam[idx.float() < b4] = L4
        lam[(idx.float() >= b4) & (idx.float() < b23)] = L23
        if existing is not None and existing.numel() > 0:
            n = min(existing.numel(), hid)
            lam[:n] = existing[:n].to(device=device, dtype=torch.int8)
        return lam

    def _allow_matrix(self, device):
        """3×3 bool A[pre,post] on `device`. Canonical Douglas–Martin/Bastos adjacency."""
        A = torch.zeros(3, 3, dtype=torch.bool, device=device)
        A[L4, L4] = True; A[L4, L23] = True                    # feedforward into the column
        A[L23, L23] = True; A[L23, L56] = True                 # associative recurrence + feedforward down
        A[L56, L56] = True                                     # deep recurrence
        if self.allow_fb:                                      # feedback (gated)
            A[L56, L4] = True; A[L56, L23] = True
        if not self.strict:                                    # loosen the two forbidden pairs
            A[L23, L4] = True; A[L4, L56] = True
        return A

    # ---- (re)build / clear the masks over the live connectome -------- #
    def _model_dtype(self, brain):
        return self.dtype or getattr(getattr(brain, "head", None), "weight", torch.empty(0)).dtype or torch.float32

    def rebuild(self, brain):
        """(Re)derive c.lamina + the laminar masks for every cortical cell from the CURRENT CSR.
        Idempotent; safe to call after grow()/prune()/load. Preserves existing lamina labels."""
        md = self._model_dtype(brain)
        tot_edge = 0.0; masked_out = 0.0
        for c in brain.cells:
            sparse = hasattr(c, "rec_val")
            dev = c.rec_val.device if sparse else c.Wrec.weight.device
            c.lamina = self._assign_lamina(c.hid, dev, getattr(c, "lamina", None))
            lam = c.lamina.long()
            A = self._allow_matrix(dev)
            # apical gain: spare the granular L4 input relay, drive L2/3 + L5/6 tufts
            gain = torch.ones(c.hid, dtype=md, device=dev)
            gain[c.lamina == L4] = float(self.apical_l4_gain)
            c.lam_apical_gain = gain
            if sparse:
                pre = lam[c.rec_col.long()]                    # edge e: pre = col, post = row
                post = lam[c.rec_row.long()]
                c.lam_rec_mask = A[pre, post]                  # (nnz,) bool — SAME partition as rec_val
                eff = (c.rec_mask & c.lam_rec_mask).to(torch.float32)
                fin = torch.zeros(c.hid, dtype=torch.float32, device=dev).index_add_(0, c.rec_row.long(), eff)
                c.lam_rec_fanin = fin.clamp_(min=1.0)          # per-neuron effective fan-in (width-invariant norm)
                tot_edge += float(c.rec_mask.sum()); masked_out += float((c.rec_mask & ~c.lam_rec_mask).sum())
                if getattr(c, "sparse_in", False):
                    post_in = lam[c.in_row.long()]
                    c.lam_in_mask = (post_in == L4) | ((self.input_to_l23 > 0) & (post_in == L23))
                    if hasattr(c, "lam_in_row"): del c.lam_in_row
                else:                                          # dense Win: a (hid,) row-gate (input → L4 rows)
                    g = (c.lamina == L4) | ((self.input_to_l23 > 0) & (c.lamina == L23))
                    c.lam_in_row = g.to(md)
                    if hasattr(c, "lam_in_mask"): del c.lam_in_mask
            else:                                              # DENSE cell (small <8192 test nets only)
                rows = lam.view(c.hid, 1).expand(c.hid, c.hid)  # post per out-row
                cols = lam.view(1, c.hid).expand(c.hid, c.hid)  # pre  per in-col
                c.lam_rec_w = A[cols, rows]                     # (hid,hid) bool: mask[out,in]=A[pre,post]
                eff = c.lam_rec_w.to(torch.float32)
                c.lam_rec_fanin = eff.sum(1).clamp_(min=1.0)
                tot_edge += float(c.hid * c.hid); masked_out += float((~c.lam_rec_w).sum())
                g = (c.lamina == L4) | ((self.input_to_l23 > 0) & (c.lamina == L23))
                c.lam_in_row = g.to(md)
        self._forbidden = (masked_out / tot_edge) if tot_edge > 0 else 0.0
        return self._forbidden

    def clear(self, brain):
        """Toggle OFF: delete the derived attrs (rec_val untouched → the flat pool is restored
        bit-identically). Off-state honoured by the getattr fast-path in spiking.py / _eprop_step."""
        for c in brain.cells:
            for a in ("lamina", "lam_rec_mask", "lam_in_mask", "lam_in_row",
                      "lam_rec_fanin", "lam_apical_gain", "lam_rec_w"):
                if hasattr(c, a):
                    delattr(c, a)

    # ---- metric ----------------------------------------------------- #
    @torch.no_grad()
    def measure(self, brain, text):
        """Run the cortex on `text` and bucket spikes by lamina ACROSS ALL LAYERS → per-lamina firing
        rates (the functional-differentiation indicator). O(hid), off the hot loop. Aggregating over every
        layer (not just the top one, which is the sparsest-firing — near-silent at depth, so sampling it
        alone pinned rate_L23/rate_L56 at 0.0) captures the active shallow laminae."""
        ids = brain.to_bytes(text)[:512]
        if len(ids) < 2:
            return self.state()
        inp = brain.E(torch.tensor([ids], device=brain.device))
        states = [c.init_state(1, brain.device) for c in brain.cells]
        acc = {L4: [0.0, 0], L23: [0.0, 0], L56: [0.0, 0]}     # per-lamina [spike_sum, cell·time count]
        any_lam = False
        for i, c in enumerate(brain.cells):
            spikes, _, states[i] = c.run_seq(inp, states[i]); inp = spikes
            lam = getattr(c, "lamina", None)
            if lam is None:
                continue
            any_lam = True
            s = spikes[0]; T = s.shape[0]                      # (T, hid)
            for lab in (L4, L23, L56):
                m = (lam == lab)
                if bool(m.any()):
                    acc[lab][0] += float(s[:, m].sum()); acc[lab][1] += int(m.sum()) * T
        if not any_lam:
            return self.state()
        for lab, attr in ((L4, "_rate_L4"), (L23, "_rate_L23"), (L56, "_rate_L56")):
            tot, n = acc[lab]
            setattr(self, attr, (tot / n) if n > 0 else 0.0)
        return self.state()
