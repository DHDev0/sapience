"""
spiking_brain.py — SpikingBrain: a growable spiking cortex (§3), the faithful core.

A stack of leaky integrate-and-fire layers over the byte stream: byte → embedding →
spiking recurrent layers (membrane carries temporal context) → readout → next byte.
By DEFAULT it learns by e-prop (§15.16): forward-in-time eligibility traces + a random-
feedback learning signal + a three-factor neuromodulator gate — no backprop-through-time,
no weight transport, a fully local + biologically plausible rule (the faithful default).
A surrogate-BPTT + Adam path — which §3.5 identifies as predictive coding in the β→0 limit —
is kept as the opt-in fast, non-plausible reference (learn_rule="bptt"). Random e-prop is a
weaker temporal-credit learner than true BPTT: the accepted price of plausibility, so its
bits/byte sits somewhat above the BPTT run's. It GROWS by §10 synaptogenesis (add LIF
neurons, identity-preserving). Drop-in for the living loop (same generate / learn_text /
think / develop / model_gb / save / load surface).

Honest: spiking is lossy and the temporal loop is slower than a rate GRU, so this is the
faithful-but-modest option you chose — fidelity to the biology over raw fluency.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spiking import LIFCell, ALIFCell, SparseLIFCell, SparseALIFCell
from . import synapse


class SpikingBrain(nn.Module):
    def __init__(self, device, dtype=torch.float32, emb=96, hidden=384, layers=2,
                 lr=2e-3, seq=64, max_model_gb=14.0, cell="lif", readout="mem",
                 read_alpha=0.5, seed=0, syn_density=0.5,
                 sparse=None, sparse_hidden_threshold=8192, rec_fanin=64, in_fanin=64):
        super().__init__()
        torch.manual_seed(seed)
        self.V = 256
        self.device = device
        self.emb_dim, self.hidden, self.layers_n = emb, hidden, layers
        self.seq = seq
        # cell: "alif" = adaptive-threshold LIF (long-timescale working memory, the
        # capability lever) | "lif" = plain LIF. readout: "mem" = tap the analog membrane
        # (lossless) | "spike" = tap the binary spike (faithful but lossy).
        self.cell_kind, self.readout, self.read_alpha = cell, readout, read_alpha
        # SPARSE connectome: at a large neuron count the H×H recurrence can't be dense (750 GB at
        # H=250k). A per-layer CSR connectome (O(H·fanin) memory) is used when the layer width
        # crosses `sparse_hidden_threshold` (or sparse=True); below it the fast dense path is kept
        # byte-identical (so the small-net tests and normal runs are unchanged). rec_fanin/in_fanin
        # set the wire-able superset; syn_density sets how much of it is initially active.
        self.sparse_cfg = dict(sparse=sparse, threshold=sparse_hidden_threshold,
                               rec_fanin=rec_fanin, in_fanin=in_fanin)
        DenseCell = ALIFCell if cell == "alif" else LIFCell
        SparseCell = SparseALIFCell if cell == "alif" else SparseLIFCell
        self.E = nn.Embedding(self.V, emb)
        cells = []
        d = emb
        for li in range(layers):
            use_sparse = sparse if sparse is not None else (hidden >= sparse_hidden_threshold)
            if use_sparse:
                cells.append(SparseCell(d, hidden, rec_fanin=rec_fanin, in_fanin=in_fanin,
                                        sparse_in=(d >= sparse_hidden_threshold or (sparse and d > emb)),
                                        syn_density=syn_density, seed=seed + li))
            else:
                cells.append(DenseCell(d, hidden))
            d = hidden
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(hidden, self.V)
        self.to(device)
        # bf16 mixed precision (weights stay fp32, matmuls autocast to bf16) — on GPU AND CPU
        self.use_amp = (dtype == torch.bfloat16)
        self.amp_dtype = torch.bfloat16
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr
        self.age = 0
        self.seen_bytes = 0
        self.max_model_gb = max_model_gb
        self.grow_until, self.prune_until = 8, 16
        self.grow_syn_frac, self.prune_frac = 0.15, 0.05
        # learning rule: "eprop" = biologically faithful e-prop (forward-in-time, local eligibility
        # traces + random-feedback learning signal, no weight transport, three-factor neuromod gate) —
        # the DEFAULT, since faithfulness is the point; "bptt" = surrogate backprop-through-time + Adam,
        # the opt-in fast, non-plausible reference (learns lower bits/byte, but breaks plausibility).
        self.learn_rule = "eprop"
        # e-prop learning rate. The update divides each synapse by its postsynaptic neuron's fan-in
        # (N_j, see _eprop_step), which makes the effective rate width-invariant → this raw scale TRANSFERS
        # across network size (identical descent 8k↔64k↔256k). 2000 is the sustainably-stable default:
        # higher (e.g. 10000) descends faster short-term but can run away over long training once
        # synaptogenesis inflates the representation magnitude (measured). Tune up cautiously ≤4000.
        self.eprop_lr_scale = 2000.0           # the BASE rate; the EFFECTIVE rate self-adapts via `attention`
        self._fanin_pow = 1.0                  # divide the RECURRENT/input update by N_j^p (p=1 → width-invariant descent)
        # The readout HEAD reads the WHOLE hidden width (fan-in = hidden), so the same p=1 fan-in norm divides its
        # update by ~hidden and STARVES it at scale: at hidden=128000 the head moves ~2.8e-8/step and stays frozen
        # at its random init → the whole net can only reach the byte-frequency baseline (measured: head_w_std≈init,
        # fb_align_cos≈0, understanding≈0). The head is a simple readout, not a width-invariant recurrent map, so it
        # gets a gentler power — live-tunable; 0 = no fan-in norm (fastest, UNSTABLE). A/B (hidden=2000, 300 steps):
        # pow 1.0 → bpb 4.02 head frozen; pow 0.7 → bpb 2.95 acc 0.49 (WIN); pow ≤0.5 → explodes. 0.7 is the sweet spot.
        self.head_fanin_pow = 0.7
        # SELF-ADAPTING plasticity (§15.17): the learning rate is NOT a fixed dial to hand-tune — it is
        # gated by `attention`, which tracks the brain's OWN learning health (loss vs. its running
        # baseline). A loss spike above baseline (a shock / struggling) DROPS attention → the update
        # shrinks → the representation is protected and re-learns gently (self-healing); at/below baseline
        # attention rises to engage. This is the Yerkes–Dodson arousal→plasticity curve; it removes the
        # need to manually chase eprop_lr_scale and would have auto-damped the Dale/lr excursions.
        self.attention = 1.0
        self.loss_ema = None
        self.attn_sensitivity = 0.8            # how sharply attention drops with above-baseline loss
        self.attn_rate_sens = 0.25             # how sharply attention drops when firing runs > 2× the homeostatic
                                               # target (the over-excitation brake; loss-blind drift protection)
        # e-prop's top-down error path (§15.16). "learned" (DEFAULT) = Kolen-Pollack: the feedback
        # matrix B gets the SAME local gradient as the readout head (+ tiny decay), so it LEARNS to
        # align with the forward weights — no weight transport, strictly more faithful than fixed random.
        # "random" = classic DFA (fixed random B, biologically plausible but leaves an alignment gap).
        self.feedback_mode = "learned"
        self.fb_decay = 1e-4                    # Kolen-Pollack weight decay that pulls B and W together
        # Dale's law (§15.16): each neuron is excitatory or inhibitory — its RECURRENT outgoing synapses
        # all share one sign (imposed on the neuron→neuron connectome by _project_dale; the feedforward
        # input projection is left unconstrained). A real biophysical constraint that SHRINKS the usable
        # weight space (costs capability).
        # Independently toggleable (off by default; measured). Dendritic/burst error delivery: see below.
        self.dale = False
        # dendritic error (§15.16): deliver the top-down learning signal L as an APICAL-dendrite drive
        # that is BURST-coded (thresholded, low-bandwidth) rather than a clean somatic scalar — the
        # Naud/Richards burst-prop picture. Independently toggleable; costs capability (noisy, low-BW).
        self.dendritic = False
        self.burst_thr = 0.5                   # apical-burst threshold (fraction of mean |L|) when dendritic
        # More faithfulness constraints (§15.16, each INDEPENDENTLY toggleable + measured; all cost some
        # capability — that is the point of the fidelity↔capability curve). See set_faith()/faith_config().
        self.bounded_synapses = False          # Fusi bounded synapses: weights clamped to ±w_max (real
        self.w_max = 1.0                       #   synapses are bounded + low-precision → stability/forgetting)
        self.homeostasis = False               # intrinsic firing-rate homeostasis (metaplasticity): each
        self.target_rate = 0.08                #   neuron's threshold drifts to hold a target spike rate
        self.homeo_lr = 0.02                   #   (Turrigiano) — keeps a continual net off silence/saturation
        self.btsp = False                      # behavioral-timescale plasticity (Bittner–Magee): the
        self.btsp_beta = 0.98                  #   eligibility trace outlives the membrane (seconds-long
        #   credit window) — decouples the eligibility decay from the membrane decay c.beta.
        # UNIFIED two-compartment cortical microcircuit (§15.17) — the biological completion that stitches
        # the substrate, the error delivery, the interneurons, and the neuromodulator into ONE circuit
        # (not separate toggles). Each neuron gets an APICAL dendrite (TwoCompartmentLIF §3.7) that
        # (a) INTEGRATES the top-down error over time, (b) admitted only through a VIP→SOM DISINHIBITION
        # gate driven by the neuromod "learn-now" tone M, (c) burst-codes it onto somatic spikes to drive
        # plasticity, and (d) FEEDS BACK onto somatic firing. PV gives fast feedforward divisive gain
        # control. When on, this SUBSUMES the standalone `dendritic` toggle — error runs THROUGH the apical
        # compartment, not alongside it. (Turning it off falls back to the somatic learning signal.)
        self.two_compartment = False
        self.g_ap = 0.15                       # apical→soma feedback coupling (the TwoCompartmentLIF gain)
        self.beta_ap = 0.9                     # apical-dendrite membrane decay (integrates the error)
        self.som_baseline = 0.5                # SOM activity-driven apical inhibition strength
        self.pv_gain = 0.3                     # PV feedforward divisive-normalization strength
        # Differentiated neuromodulation (§15.17): the four tones gate DISTINCT pathways rather than one
        # scalar — ACh gates cortical encoding/plasticity (the VIP "learn-now" drive), DA reward-modulates
        # the plasticity magnitude, NE sets somatic gain (surprise/attention), 5-HT sets apical patience.
        self.diff_neuromod = False
        # Stochastic spiking + metabolic cost (§15.17): real neurons fire probabilistically (noisy vesicle
        # release) and are energy-constrained. `stochastic` adds membrane noise before threshold; `metabolic`
        # adds a spike-rate penalty to the learning signal (a synapse driving excess spikes is pushed down).
        self.stochastic = False
        self.spike_noise = 0.1
        self.metabolic = False
        self.metabolic_lambda = 0.01
        self._mind = None                      # persistent per-layer state = stream of thought
        self._last = None
        # §10: the NEURON count is fixed at birth; the SYNAPSE count is what develops. Seed a
        # sparse connectome (syn_density of connections active) that childhood then densifies.
        self.syn_density = syn_density
        self._init_synapse_mask(syn_density)

    @staticmethod
    def to_bytes(text):
        return list(text.encode("utf-8", errors="replace"))

    def model_gb(self):
        return sum(p.numel() for p in self.parameters()) * 4 / 1e9

    @property
    def eta(self):
        return self.lr

    # ---- run the spiking dynamics over a byte sequence --------------- #
    def _run(self, x, states=None):
        """x: (B,T) ids -> logits (B,T,V) and final states. Membrane carries context.

        Layer-outer, time-inner: each cell runs over the WHOLE sequence (input projection
        vectorized in one matmul, head vectorized over time), which is mathematically
        identical to time-outer stepping for this feedforward stack but far faster — the
        per-timestep Python loop no longer does the input-projection or readout matmuls."""
        B, T = x.shape
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            inp = self.E(x)                                # (B,T,emb)
            if states is None:
                states = [c.init_state(B, self.device) for c in self.cells]
            top_mem = None
            for i, c in enumerate(self.cells):
                spikes, mems, states[i] = c.run_seq(inp, states[i])
                inp = spikes; top_mem = mems
            read = self._readout(top_mem, inp)             # (B,T,hid)
            logits = self.head(read)                       # (B,T,V) in one matmul
            return logits, states

    def _readout(self, mem, spk):
        """How the head taps the top layer. 'spike' = binary (faithful, lossy); 'mem' = raw
        analog membrane (lossless, the winner); 'memtanh' = squashed membrane; 'mix' =
        (1-α)·spike + α·tanh(membrane). All keep head input dim = hidden, so growth stays
        identity-preserving (new neurons have membrane≈0 and head[:,new]≈0)."""
        ro = self.readout
        if ro == "spike":   return spk
        if ro == "memtanh": return torch.tanh(mem)
        if ro == "mix":     return (1.0 - self.read_alpha) * spk + self.read_alpha * torch.tanh(mem)
        return mem                                          # 'mem' default

    # ---- LEARN: surrogate-gradient BPTT (= PC at β→0, §3.5) ---------- #
    def learn_text(self, text, epochs=1, bs=16, max_steps=12, store=True,
                   replay_interleave=0, consolidate_rounds=0, seq=None, on_step=None, gate=1.0, tone=None):
        if getattr(self, "learn_rule", "bptt") == "eprop":     # faithful forward-in-time route
            return self.learn_eprop(text, epochs=epochs, bs=bs, max_steps=max_steps, seq=seq,
                                    on_step=on_step, gate=gate, tone=tone)
        data = text if isinstance(text, list) else self.to_bytes(text)
        seq = seq or self.seq                          # sleep can consolidate on a longer context
        if len(data) <= seq + 1:
            return None
        t = torch.tensor(data, device=self.device)
        n = t.numel()
        first = last = None
        for _ in range(epochs):
            steps = max(1, min(max_steps, (n - seq) // (bs * seq) + 1))
            for _s in range(steps):
                i = torch.randint(0, n - seq - 1, (bs,), device=self.device)
                x = torch.stack([t[k:k + seq] for k in i])
                y = torch.stack([t[k + 1:k + seq + 1] for k in i])
                logits, _ = self._run(x)
                loss = F.cross_entropy(logits.reshape(-1, self.V), y.reshape(-1))
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.opt.step()
                self._apply_prune_mask()            # keep pruned synapses at zero (§10)
                last = loss.item()
                if first is None: first = last
                if on_step is not None:             # intra-epoch heartbeat (birth progress, etc.)
                    on_step(_s + 1, steps, last)
        self.seen_bytes += n
        # (first_loss, last_loss): the drop across this text = free learning-progress signal
        return (first if first is not None else 0.0, last if last is not None else 0.0)

    # ---- LEARN (faithful): e-prop, forward-in-time, local, no weight transport ---- #
    # Bellec et al. 2020 (Nature Comms) — the online approximation to BPTT for spiking recurrent
    # nets, purpose-built for LIF/ALIF. Instead of loss.backward() (a reverse pass unrolled through
    # ALL of time, using the transpose of the forward weights — the exact things biology cannot do),
    # each synapse keeps a forward eligibility TRACE and is updated by a per-neuron LEARNING SIGNAL:
    #   ΔW_ji = -η · M · Σ_t  L_j^t · e_ji^t ,   e_ji^t = ψ_j^t · ε_i^t ,   ε_i^t = β·ε_i^{t-1} + z_i^{t-1}
    # ε_i (per PRE-neuron) rides forward in time; ψ_j is the surrogate pseudo-derivative; L_j^t is
    # the output error projected back through a FIXED RANDOM feedback matrix (random e-prop / DFA →
    # no weight transport, and each layer gets its own signal → no cross-layer backprop); M is the
    # neuromodulator tone (three-factor gate — the §5 coupling, now load-bearing). The readout head
    # reads the current membrane, so its gradient is already local in time (err ⊗ v). Nothing is
    # unrolled backward; the whole update is computed online during one forward pass.
    @staticmethod
    def _psi(x):
        """Surrogate pseudo-derivative of the spike at (membrane − threshold) — fast-sigmoid."""
        return 1.0 / (10.0 * x.abs() + 1.0) ** 2

    def _ensure_feedback(self):
        """Fixed random feedback B_l (V × hid_l) per layer — the plausible top-down error path that
        replaces weight transport. Built once; grows with the layer (identity-neutral new columns)."""
        if not hasattr(self, "_fb"):
            self._fb = []
        while len(self._fb) < len(self.cells):
            l = len(self._fb); h = self.cells[l].hid
            self._fb.append((torch.randn(self.V, h, device=self.device) / (self.V ** 0.5)))
        for l, c in enumerate(self.cells):                 # keep width in sync after neuron growth
            if self._fb[l].shape[1] != c.hid:
                b = torch.randn(self.V, c.hid, device=self.device) / (self.V ** 0.5)
                b[:, :self._fb[l].shape[1]] = self._fb[l]; self._fb[l] = b

    @torch.no_grad()
    def learn_eprop(self, text, epochs=1, bs=16, max_steps=12, seq=None, on_step=None, gate=1.0, tone=None):
        """Train the cortex by e-prop (see above). Returns (first_loss, last_loss) like learn_text.
        `tone` = the §5 neuromodulator dict {da,ach,ne,ht}; used per-pathway when diff_neuromod is on."""
        data = text if isinstance(text, list) else self.to_bytes(text)
        seq = seq or self.seq
        if len(data) <= seq + 1:
            return None
        self._ensure_feedback()
        t = torch.tensor(data, device=self.device); n = t.numel()
        first = last = None
        for _ in range(epochs):
            steps = max(1, min(max_steps, (n - seq) // (bs * seq) + 1))
            for _s in range(steps):
                i = torch.randint(0, n - seq - 1, (bs,), device=self.device)
                x = torch.stack([t[k:k + seq] for k in i])
                y = torch.stack([t[k + 1:k + seq + 1] for k in i])
                loss = self._eprop_step(x, y, gate, tone=tone)
                last = loss
                if first is None: first = last
                if on_step is not None:
                    on_step(_s + 1, steps, last)
        self.seen_bytes += n
        return (first if first is not None else 0.0, last if last is not None else 0.0)

    _EP_CHUNK = 1 << 26                                        # cap on any transient (chunk, nnz) buffer

    @torch.no_grad()
    def _eprop_step(self, x, y, gate=1.0, tone=None):
        """One e-prop gradient step over a (B,T) window — pure PyTorch, NO O(H²) anywhere. Eligibility
        traces are per-neuron (O(H)); the recurrent grad is accumulated PER SYNAPSE by gather/scatter
        (O(nnz) for a sparse cortex, so it scales to hundreds of thousands of neurons). Timestep-outer
        (== the layer-outer forward), online, gated by the neuromodulator. No autograd, no BPTT."""
        B, T = x.shape
        cells = self.cells; dev = self.device
        inp = self.E(x)                                        # (B,T,emb)
        sp = lambda c: hasattr(c, "rec_val")                   # sparse cell?
        al = [hasattr(c, "rho") for c in cells]                # ALIF (adaptive threshold) cell?
        v = [torch.zeros(B, c.hid, device=dev) for c in cells]
        z = [torch.zeros(B, c.hid, device=dev) for c in cells]
        a = [torch.zeros(B, c.hid, device=dev) for c in cells]                 # ALIF adaptation state
        eps_rec = [torch.zeros(B, c.hid, device=dev) for c in cells]           # per-PRE recurrent trace ε^v
        eps_in = [torch.zeros(B, c.in_dim, device=dev) for c in cells]         # per-PRE input trace ε^v
        # ALIF per-SYNAPSE adaptation eligibility ε^a (sparse: (B,nnz); dense: (B,out,in) — only for
        # the small dense test nets). LIF cells keep ε^a=0 (their eligibility is exactly ε^v).
        ea_rec = [torch.zeros(B, c.rec_val.numel(), device=dev) if (al[i] and sp(c))
                  else (torch.zeros(B, c.hid, c.hid, device=dev) if al[i] else None)
                  for i, c in enumerate(cells)]
        ea_in = [None] * len(cells)
        for i, c in enumerate(cells):
            if not al[i]: continue
            if sp(c) and c.sparse_in: ea_in[i] = torch.zeros(B, c.in_val.numel(), device=dev)
            else: ea_in[i] = torch.zeros(B, c.hid, c.in_dim, device=dev)
        g_rec = [torch.zeros_like(c.rec_val) if sp(c) else torch.zeros_like(c.Wrec.weight) for c in cells]
        g_in, g_in_b = [], []
        for c in cells:
            if sp(c) and c.sparse_in: g_in.append(torch.zeros_like(c.in_val)); g_in_b.append(torch.zeros_like(c.in_bias))
            else: g_in.append(torch.zeros_like(c.Win.weight)); g_in_b.append(torch.zeros_like(c.Win.bias))
        gHead = torch.zeros_like(self.head.weight); gHead_b = torch.zeros_like(self.head.bias)
        gE = torch.zeros_like(self.E.weight)                   # the sensory byte-embedding also learns (e-prop)
        learned_fb = getattr(self, "feedback_mode", "learned") == "learned"
        gFB = [torch.zeros_like(self._fb[l]) for l in range(len(cells) - 1)] if learned_fb else None
        dendritic = getattr(self, "dendritic", False)          # apical burst-coded error delivery?
        burst_thr = float(getattr(self, "burst_thr", 0.5))
        burst_frac = 0.0
        btsp = getattr(self, "btsp", False)                    # long (behavioral-timescale) eligibility?
        ebeta = float(getattr(self, "btsp_beta", 0.98))
        homeo = getattr(self, "homeostasis", False)            # intrinsic firing-rate homeostasis?
        bounded = getattr(self, "bounded_synapses", False)     # Fusi bounded weights?
        twocomp = getattr(self, "two_compartment", False)      # UNIFIED apical/interneuron/neuromod circuit?
        g_ap = float(getattr(self, "g_ap", 0.15)); beta_ap = float(getattr(self, "beta_ap", 0.9))
        som_b = float(getattr(self, "som_baseline", 0.5)); pv_g = float(getattr(self, "pv_gain", 0.3))
        ap = [torch.zeros(B, c.hid, device=dev) for c in cells] if twocomp else None   # apical dendrite state
        # differentiated neuromodulation: the 4 tones gate 4 distinct pathways (else one scalar `gate`=ACh).
        diffnm = getattr(self, "diff_neuromod", False) and isinstance(tone, dict)
        da = float(tone.get("da", 0.5)) if diffnm else 0.5     # DA → reward-modulated plasticity magnitude
        ne_gain = (0.5 + float(tone.get("ne", 1.0))) if diffnm else 1.0   # NE → somatic gain (surprise/attention)
        ht = float(tone.get("ht", 0.5)) if diffnm else 0.5     # 5-HT → patience (apical + eligibility timescale)
        ht_pat = (0.85 + 0.3 * ht) if diffnm else 1.0          # stretches the eligibility window (works w/o twocomp)
        if diffnm: beta_ap = min(0.99, beta_ap * (0.7 + 0.6 * ht))        # 5-HT → apical patience (twocomp)
        stoch = getattr(self, "stochastic", False)             # probabilistic (noisy) spiking?
        snoise = float(getattr(self, "spike_noise", 0.1))
        metab = getattr(self, "metabolic", False)              # spike-rate energy penalty on the update?
        mlam = float(getattr(self, "metabolic_lambda", 0.01))
        if homeo:
            self._ensure_homeo(); spk_sum = [torch.zeros(c.hid, device=dev) for c in cells]
        lr = self.lr * getattr(self, "eprop_lr_scale", 2000.0)
        CH = self._EP_CHUNK
        def spmm(val, col, row, xin, out_dim):                 # y[b,row] += val · xin[b,col]  (O(nnz·B))
            cl = col.long(); ch = max(1, min(B, CH // max(1, cl.numel())))
            y = torch.zeros(B, out_dim, device=dev)
            for i in range(0, B, ch):
                xc = xin[i:i + ch].float()
                y[i:i + ch] = torch.zeros(xc.shape[0], out_dim, device=dev).index_add_(1, row, val.float().unsqueeze(0) * xc[:, cl])
            return y
        def sddmm(gp, ep, row, col):                           # Σ_b gp[b,row]·ep[b,col] → (nnz)  (O(nnz·B))
            cl = col.long(); nnz = cl.numel(); ch = max(1, min(B, CH // max(1, nnz)))
            out = torch.zeros(nnz, device=dev)
            for i in range(0, B, ch):
                out += (gp[i:i + ch][:, row] * ep[i:i + ch][:, cl]).sum(0)
            return out
        def edge_reduce(gp, ea, row):                          # Σ_b gp[b,row]·ea[b]  → (nnz)  (ALIF term)
            nnz = ea.shape[1]; ch = max(1, min(B, CH // max(1, nnz)))
            out = torch.zeros(nnz, device=dev)
            for i in range(0, B, ch):
                out += (gp[i:i + ch][:, row] * ea[i:i + ch]).sum(0)
            return out
        tot_loss = 0.0
        for tt in range(T):
            layer_in = inp[:, tt]; psi = []
            for l, c in enumerate(cells):
                z_prev = z[l]
                if sp(c):
                    rec = spmm(c.rec_val * c.rec_mask, c.rec_col, c.rec_row, z_prev, c.hid)
                    pre = (spmm(c.in_val * c.in_mask, c.in_col, c.in_row, layer_in, c.hid) + c.in_bias) \
                        if c.sparse_in else c.Win(layer_in)
                else:
                    rec = c.Wrec(z_prev); pre = c.Win(layer_in)
                drive = pre + rec
                if twocomp:                                    # PV fast divisive gain control + apical→soma feedback
                    drive = drive / (1.0 + pv_g * z_prev.mean(1, keepdim=True)) + g_ap * ap[l]
                if diffnm: drive = drive * ne_gain             # NE sets somatic gain (surprise/attention)
                v[l] = c.beta * v[l] * (1.0 - z_prev) + drive
                if al[l]:
                    a[l] = c.rho * a[l] + z_prev               # adaptation from the previous spike
                    thr = c.thr0 + c.beta_adapt * a[l]         # adaptive threshold
                else:
                    thr = c.thr
                if homeo: thr = thr + self._thr_adapt[l]       # intrinsic homeostasis offset (metaplasticity)
                vfire = v[l] + snoise * torch.randn_like(v[l]) if stoch else v[l]   # stochastic (noisy) firing
                psi_l = self._psi(vfire - thr); z[l] = (vfire >= thr).float()
                if homeo: spk_sum[l] = spk_sum[l] + z[l].sum(0)
                eb = ebeta if btsp else c.beta                 # BTSP: eligibility outlives the membrane
                dyn_eb = getattr(self, "_dyn_elig_beta", None)  # §16 P2: attention→FREQUENCY sets the window
                if dyn_eb is not None: eb = float(dyn_eb)       #   (gamma-short when focused, alpha-long when not)
                if diffnm: eb = min(0.995, eb * ht_pat)        # 5-HT stretches the eligibility window (patience)
                eps_rec[l] = eb * eps_rec[l] + z_prev          # ε^v forward eligibility (per pre-neuron)
                eps_in[l] = eb * eps_in[l] + layer_in
                if al[l]:                                      # ε^a = ψ_j·ε^v_i + (ρ − β_a·ψ_j)·ε^a (per synapse)
                    ba, rho = c.beta_adapt, c.rho
                    if sp(c):
                        pr = psi_l[:, c.rec_row]                                    # (B,nnz) ψ at post-row
                        ea_rec[l] = pr * eps_rec[l][:, c.rec_col.long()] + (rho - ba * pr) * ea_rec[l]
                        if c.sparse_in:
                            pi = psi_l[:, c.in_row]
                            ea_in[l] = pi * eps_in[l][:, c.in_col.long()] + (rho - ba * pi) * ea_in[l]
                        else:
                            ea_in[l] = psi_l.unsqueeze(2) * eps_in[l].unsqueeze(1) + (rho - ba * psi_l).unsqueeze(2) * ea_in[l]
                    else:
                        ea_rec[l] = psi_l.unsqueeze(2) * eps_rec[l].unsqueeze(1) + (rho - ba * psi_l).unsqueeze(2) * ea_rec[l]
                        ea_in[l] = psi_l.unsqueeze(2) * eps_in[l].unsqueeze(1) + (rho - ba * psi_l).unsqueeze(2) * ea_in[l]
                psi.append(psi_l); layer_in = z[l]
            top_v = v[-1]; logits = self.head(top_v)           # membrane readout → logits
            p = torch.softmax(logits.float(), 1)
            oh = torch.zeros_like(p); oh.scatter_(1, y[:, tt].long().unsqueeze(1), 1.0)
            err = p - oh                                        # CE gradient wrt logits
            tot_loss += float(-(oh * (p + 1e-9).log()).sum(1).mean())
            gHead += err.t() @ top_v.float(); gHead_b += err.sum(0)   # head grad is LOCAL in time
            if learned_fb:                                     # Kolen-Pollack: B learns the head's grad (top
                for l in range(len(cells) - 1):                # layer) / an error·activity correlation (lower)
                    gFB[l] += err.t() @ v[l].float()
            for l, c in enumerate(cells):
                Lsig = err @ self._fb[l].float()               # top-down learning signal (random or learned B)
                if twocomp:                                    # UNIFIED apical circuit — error runs THROUGH the
                    vip = float(gate)                          # apical dendrite, not alongside it:
                    som = som_b * z[l].mean(1, keepdim=True)    #  SOM inhibition ∝ population activity, and
                    agate = (vip - som).clamp(min=0.0)         #  VIP (neuromod "learn-now" tone) DISINHIBITS it
                    ap[l] = beta_ap * ap[l] + agate * Lsig     #  → the apical compartment integrates the gated error
                    thr_b = burst_thr * (ap[l].abs().mean(1, keepdim=True) + 1e-9)
                    brst = (ap[l].abs() > thr_b).float() * z[l]  # apical BURST rides a somatic spike (plateau)
                    Lsig = ap[l] * brst
                    if l == len(cells) - 1: burst_frac = float(brst.mean())
                elif dendritic:                                # standalone apical burst code (Naud/Richards),
                    thr = burst_thr * (Lsig.abs().mean(1, keepdim=True) + 1e-9)   # not yet routed through a
                    brst = (Lsig.abs() > thr).float() * z[l]   # two-compartment neuron — low-bandwidth, noisy
                    Lsig = Lsig * brst
                    if l == len(cells) - 1: burst_frac = float(brst.mean())
                if metab: Lsig = Lsig + mlam * z[l]            # metabolic cost: a spike-rate loss (mlam·Σz)
                #   has dLoss/dz=+mlam → ADDS to the per-neuron error, pushing incoming weights DOWN (less firing)
                g = (Lsig * psi[l]).float()                    # g_j = L_j · ψ_j
                if l == 0 and getattr(c, "Win", None) is not None:   # the byte-embedding learns too: project the
                    gE.index_add_(0, x[:, tt].long(), g @ c.Win.weight)   # layer-0 signal back to the used rows.
                    #   (this input-projection gradient is the ONE weight-transport path in the rule — a
                    #   deliberate, disclosed exception; all cortical credit still flows through _fb, not W^T.)
                ba = c.beta_adapt if al[l] else 0.0            # e_ji = ψ_j(ε^v_i − β_a·ε^a_ji); grad = Σ g_j·(ε^v_i − β_a·ε^a_ji)
                if sp(c):
                    g_rec[l] += sddmm(g, eps_rec[l], c.rec_row, c.rec_col)     # membrane part, O(nnz)
                    if al[l]: g_rec[l] += -ba * edge_reduce(g, ea_rec[l], c.rec_row)   # adaptation part
                    if c.sparse_in:
                        g_in[l] += sddmm(g, eps_in[l], c.in_row, c.in_col)
                        if al[l]: g_in[l] += -ba * edge_reduce(g, ea_in[l], c.in_row)
                        g_in_b[l] += g.sum(0)
                    else:
                        g_in[l] += g.t() @ eps_in[l].float()                          # dense in (emb small)
                        if al[l]: g_in[l] += -ba * (g.unsqueeze(2) * ea_in[l]).sum(0)
                        g_in_b[l] += g.sum(0)
                else:
                    g_rec[l] += g.t() @ eps_rec[l].float()
                    if al[l]: g_rec[l] += -ba * (g.unsqueeze(2) * ea_rec[l]).sum(0)    # (h,h) dense adaptation
                    g_in[l] += g.t() @ eps_in[l].float()
                    if al[l]: g_in[l] += -ba * (g.unsqueeze(2) * ea_in[l]).sum(0)
                    g_in_b[l] += g.sum(0)
        # FULLY LOCAL three-factor update: Δw_ji = -η·M · clamp(mean_t[L_j·e_ji]/N_j^p, ±Δmax). Each
        # synapse sees only its own pre-trace, post learning-signal and the neuromodulator M — no global
        # norm (the old global grad-norm clip was the one non-local operation). Two local homeostatic
        # constraints keep it stable at any width: (1) a bounded per-synapse change rate Δmax; (2)
        # per-postsynaptic-neuron fan-in normalization by N_j^p — a wide neuron's afferent gradient is
        # fan-in-coherent (g_ji ∝ pre-activity), so dividing by its OWN afferent count makes the drive
        # change O(1) and the stable rate width-invariant (input scaling; each neuron knows only N_j).
        p = float(getattr(self, "_fanin_pow", 1.0))
        denom = float(B * T); dmax = 0.02
        # self-adapting attention: this step's loss vs. the brain's running baseline sets plasticity.
        L = tot_loss / T
        if getattr(self, "loss_ema", None) is None: self.loss_ema = L
        surprise = (L - self.loss_ema) / max(self.loss_ema, 1.0)     # >0 = worse than usual (struggling)
        sens = float(getattr(self, "attn_sensitivity", 0.8))
        # OVER-EXCITATION brake: firing far above the homeostatic target is a REPRESENTATION runaway the
        # loss-surprise term is BLIND to — training loss stays low while the net inflates (mem_mag/bpb drift
        # up). Unlike a self-following EMA baseline, the fixed rate target is an ABSOLUTE anchor a slow drift
        # cannot escape, so attention (and thus the effective rate) self-corrects the over-firing. Fires only
        # past 2× target; live-tunable via attn_rate_sens (0 disables). Needs homeostasis' per-step spk_sum.
        over = 0.0
        if homeo and spk_sum is not None:
            rate = float(spk_sum[-1].sum()) / (float(B * T) * float(cells[-1].hid))   # top-layer mean firing rate
            over = min(3.0, max(0.0, rate / max(float(getattr(self, "target_rate", 0.08)), 1e-3) - 2.0))
        rate_sens = float(getattr(self, "attn_rate_sens", 0.25))
        attn_t = min(1.3, max(0.2, 1.0 - sens * surprise - rate_sens * over))   # loss-shock OR over-firing → less plasticity
        cur_at = float(getattr(self, "attention", 1.0))
        aw = 0.3 if attn_t < cur_at else 0.1        # ASYMMETRIC: brake FAST (safety-biased), release SLOW — so an
        self.attention = (1.0 - aw) * cur_at + aw * attn_t   # over-firing/loss signal pulls the rate down promptly, not over ~10 steps
        self.loss_ema = 0.98 * self.loss_ema + 0.02 * L             # slow learning-health baseline
        scale = float(gate) * lr * self.attention * ((0.5 + da) if diffnm else 1.0)   # ACh gates; ATTENTION self-adapts; DA reward-modulates
        def _upd(w, g, fin, pw=p):
            w.add_((scale * (g / (denom * float(fin) ** pw))).clamp_(-dmax, dmax).to(w.dtype), alpha=-1.0)
        hpow = float(getattr(self, "head_fanin_pow", 0.5))     # gentler fan-in norm for the wide readout head
        for l, c in enumerate(cells):
            if sp(c):
                _upd(c.rec_val, g_rec[l] * c.rec_mask, c.rec_fanin)   # silent synapses get no update
                if c.sparse_in:
                    _upd(c.in_val, g_in[l] * c.in_mask, c.in_fanin); _upd(c.in_bias, g_in_b[l], 1)
                else:
                    _upd(c.Win.weight, g_in[l], c.Win.weight.shape[1]); _upd(c.Win.bias, g_in_b[l], 1)
            else:
                _upd(c.Wrec.weight, g_rec[l], c.Wrec.weight.shape[1])
                _upd(c.Win.weight, g_in[l], c.Win.weight.shape[1]); _upd(c.Win.bias, g_in_b[l], 1)
        _upd(self.head.weight, gHead, self.head.weight.shape[1], pw=hpow)   # head: gentler norm so it isn't starved at width
        _upd(self.head.bias, gHead_b, 1)
        _upd(self.E.weight, gE, self.E.weight.shape[1])        # sensory byte-embedding update (no longer frozen)
        if learned_fb:                                         # Kolen-Pollack: mirror the head/error grad into
            fb_dec = float(getattr(self, "fb_decay", 1e-4))    # B, then weight-decay → B aligns with W, locally
            for l in range(len(cells)):                         # B must move at the SAME rate as the head (hpow),
                _upd(self._fb[l], gHead if l == len(cells) - 1 else gFB[l], self._fb[l].shape[1], pw=hpow)  # else it can't align
                self._fb[l].mul_(1.0 - fb_dec)
        if bounded:                                            # Fusi bounded synapses: clamp to ±w_max
            wm = float(getattr(self, "w_max", 1.0))
            for c in cells:
                if sp(c):
                    c.rec_val.data.clamp_(-wm, wm)
                    if c.sparse_in: c.in_val.data.clamp_(-wm, wm)
                else:
                    c.Wrec.weight.data.clamp_(-wm, wm); c.Win.weight.data.clamp_(-wm, wm)
        if homeo:                                              # intrinsic homeostasis: threshold → target rate
            for l in range(len(cells)):
                self._thr_adapt[l] += self.homeo_lr * (spk_sum[l] / float(B * T) - float(self.target_rate))
        if getattr(self, "dale", False):
            self._project_dale()                               # re-impose E/I sign law after the update
        self._burst_frac = burst_frac
        if twocomp: self._apical_mag = float(ap[-1].abs().mean())   # apical-dendrite drive magnitude
        # DIAGNOSTIC METRICS — the leading indicators the root-cause read needed (bpb alone lagged):
        #  mem_mag = top-layer membrane |v| = the REPRESENTATION magnitude; a runaway here (the actual
        #  collapse mechanism) climbs BEFORE bpb blows up. update_mag = per-step head Δw; grad_mag = raw
        #  readout gradient. Together they show whether an excursion is drive-runaway, over-plasticity, or data.
        _ct = cells[-1]
        _grt = g_rec[-1]; _rf = (_ct.rec_fanin if sp(_ct) else _ct.hid)      # RECURRENT update = runaway-relevant
        self._diag = dict(
            mem_mag=float(v[-1].abs().mean()),                               # representation magnitude (runaway indicator)
            grad_mag=float(gHead.abs().mean()),
            update_mag=float((scale * (_grt / (denom * float(_rf) ** p))).clamp(-dmax, dmax).abs().mean()),
            # head_update_mag = per-step readout Δw. If ~0 while grad_mag>0, the head is STARVED (fan-in norm too
            # strong at width) → it stays at random init and the net can't learn past the byte-frequency baseline.
            head_update_mag=float((scale * (gHead / (denom * float(self.head.weight.shape[1]) ** hpow))).clamp(-dmax, dmax).abs().mean()),
            surprise=float(surprise),
            rec_w_mag=float(_ct.rec_val.abs().mean() if sp(_ct) else _ct.Wrec.weight.abs().mean()),
        )
        self._apply_prune_mask()
        return tot_loss / T

    # ---- THINK: continue the persistent spiking mind-state ----------- #
    @torch.no_grad()
    def think(self, n=16, temperature=0.7):
        self.eval()
        if self._mind is None:
            self._mind = [c.init_state(1, self.device) for c in self.cells]
            cur = torch.tensor([[ord("\n")]], device=self.device)
        else:
            cur = self._last if self._last is not None else torch.tensor([[ord(" ")]], device=self.device)
        out = []
        for _ in range(n):
            logits, self._mind = self._run(cur, self._mind)
            p = torch.softmax(logits[0, -1].float() / max(temperature, 1e-3), 0)
            cur = torch.multinomial(p, 1).view(1, 1); out.append(int(cur.item()))
        self._last = cur
        self.train()
        return bytes(out).decode("utf-8", "replace")

    @torch.no_grad()
    def observe_stream(self, text):
        self.eval()
        ids = self.to_bytes(text)[-256:]
        if ids:
            x = torch.tensor([ids], device=self.device)
            _, self._mind = self._run(x, self._mind)
            self._mind = [tuple(z.detach() for z in st) for st in self._mind]  # LIF (v,s) or ALIF (v,s,a)
            self._last = x[:, -1:].clone()
        self.train()

    # ---- RESONATE IN PARALLEL: k thought streams in one batched forward ---- #
    @torch.no_grad()
    def resonate(self, k=4, n=24, temperature=0.9):
        """Run k independent thought streams from the CURRENT mind-state in ONE batched
        forward pass (~= the wall-time of a single stream, since the LIF matmuls batch over
        streams). Returns k continuations. The primary stream (self._mind) is left untouched
        — this is parallel exploration/curiosity, not a commit."""
        self.eval()
        if self._mind is None:
            states = [c.init_state(k, self.device) for c in self.cells]
            cur = torch.full((k, 1), ord("\n"), device=self.device, dtype=torch.long)
        else:
            states = [tuple(z.expand(k, *z.shape[1:]).contiguous() for z in st) for st in self._mind]
            last = self._last if self._last is not None else torch.full((1, 1), ord(" "), device=self.device, dtype=torch.long)
            cur = last.expand(k, 1).contiguous()
        outs = [[] for _ in range(k)]
        for _ in range(n):
            logits, states = self._run(cur, states)
            p = torch.softmax(logits[:, -1].float() / max(temperature, 1e-3), -1)   # (k,V)
            cur = torch.multinomial(p, 1)                                            # (k,1)
            for j in range(k):
                outs[j].append(int(cur[j].item()))
        self.train()
        return [bytes(o).decode("utf-8", "replace") for o in outs]

    @torch.no_grad()
    def generate(self, prompt="", n=200, temperature=0.6, seed=0):
        self.eval()
        ids = self.to_bytes(prompt) or [ord("\n")]
        x = torch.tensor([ids], device=self.device)
        _, states = self._run(x)
        cur = x[:, -1:]
        out = []
        for _ in range(n):
            logits, states = self._run(cur, states)
            p = torch.softmax(logits[0, -1].float() / max(temperature, 1e-3), 0)
            cur = torch.multinomial(p, 1).view(1, 1); out.append(int(cur.item()))
        self.train()
        return prompt + bytes(out).decode("utf-8", "replace")

    def generative_replay(self, n=8, dream_len=160, temperature=1.1, cues=None,
                          probe=None, anchor=None, anchor_frac=0.2):
        # NOT @torch.no_grad: the inner generate()/bits_per_byte() self-wrap in no_grad, and learn_text must
        # keep grad enabled for the opt-in bptt route (loss.backward) — a blanket no_grad here crashed it.
        """§16 GENERATIVE SELF-REPLAY (pseudo-rehearsal) — the buffer-free consolidation. The cortex DREAMS
        sequences from its OWN dynamics and hard-learns them, so its generalized memory is rehearsed with NO
        raw replay buffer (CLS; Robins 1995, Shin 2017, van de Ven 2020). Diverse high-temperature dreams
        sample the whole learned distribution → it rehearses everything it knows (forgetting-resistance).
        Safeguards vs the self-reinforcing 'overfitted brain': a small VERIDICAL anchor fraction keeps replay
        on the data manifold, and an ACCEPTANCE monitor on a held-out probe reports drift so the caller can
        raise the anchor if replay degrades. `cues` = sparse byte-cues (the hippocampal index, not the episode)."""
        cues = cues or [""]
        before = self.bits_per_byte(probe) if probe is not None else None
        dreamed = 0
        for i in range(n):
            if anchor is not None and (i / max(1, n)) < anchor_frac:      # veridical anchor (small fraction)
                text = anchor if isinstance(anchor, str) else anchor[i % len(anchor)]
            else:
                cue = cues[i % len(cues)]                                 # sparse cue → DREAM the continuation
                text = self.generate(cue, n=dream_len, temperature=temperature)
            if len(text) >= 32:
                self.learn_text(text, epochs=1, max_steps=2); dreamed += 1
        drift = (self.bits_per_byte(probe) - before) if probe is not None else 0.0   # >0 = replay hurt the probe
        return dict(dreamed=dreamed, probe_drift=round(float(drift), 4))

    # ---- eval -------------------------------------------------------- #
    @torch.no_grad()
    def next_byte_acc(self, text):
        self.eval()
        data = self.to_bytes(text)
        if len(data) <= self.seq:
            self.train(); return 0.0
        t = torch.tensor(data[:2048], device=self.device).unsqueeze(0)
        logits, _ = self._run(t[:, :-1])
        acc = (logits.argmax(-1) == t[:, 1:]).float().mean().item()
        self.train(); return acc

    @torch.no_grad()
    def bits_per_byte(self, text):
        self.eval()
        data = self.to_bytes(text)
        if len(data) < 8:
            self.train(); return float("nan")
        t = torch.tensor(data[:2048], device=self.device).unsqueeze(0)
        logits, _ = self._run(t[:, :-1])
        bpb = F.cross_entropy(logits.reshape(-1, self.V), t[:, 1:].reshape(-1)).item() / 0.6931
        self.train(); return bpb

    # ---- diagnostics: entropy / perplexity / firing / weight health -- #
    @torch.no_grad()
    def train_perplexity(self, text):
        """Perplexity on `text` = 2^(bits/byte). How surprised the model is by real data."""
        b = self.bits_per_byte(text)
        return float(2.0 ** b) if b == b else float("nan")     # nan-safe

    @torch.no_grad()
    def generate_diag(self, prompt="", n=140, temperature=0.7):
        """Generate a sample and measure the output distribution: mean per-step entropy (bits;
        0 = deterministic, 8 = uniform over 256) and self-perplexity (how surprised it is by
        its OWN samples — low = confident/repetitive, high = diverse/uncertain)."""
        self.eval()
        ids = self.to_bytes(prompt) or [ord("\n")]
        x = torch.tensor([ids], device=self.device)
        _, states = self._run(x); cur = x[:, -1:]
        ent = 0.0; nll = 0.0; out = []
        for _ in range(n):
            logits, states = self._run(cur, states)
            pc = torch.softmax(logits[0, -1].float(), 0)                # untempered distribution
            ent += float(-(pc * (pc + 1e-9).log()).sum())
            p = torch.softmax(logits[0, -1].float() / max(temperature, 1e-3), 0)
            idx = torch.multinomial(p, 1)
            nll += float(-(pc[idx] + 1e-9).log())
            cur = idx.view(1, 1); out.append(int(idx.item()))
        self.train()
        txt = bytes(out).decode("utf-8", "replace")
        return dict(text=txt, entropy_bits=ent / n / 0.6931, perplexity=math.exp(nll / n))

    @torch.no_grad()
    def spike_rate(self, text):
        """Mean firing fraction across the spiking layers on `text` (0 = silent/dead, 1 = all
        firing). A core 'state of the net' — too low means dead neurons, too high means no
        sparsity."""
        ids = self.to_bytes(text)[:512]
        if len(ids) < 2:
            return 0.0
        inp = self.E(torch.tensor([ids], device=self.device))
        states = [c.init_state(1, self.device) for c in self.cells]
        rates = []
        for i, c in enumerate(self.cells):
            spikes, _, states[i] = c.run_seq(inp, states[i]); inp = spikes
            rates.append(float(spikes.mean()))
        return sum(rates) / len(rates)

    @torch.no_grad()
    def weight_stats(self):
        """Per-layer weight magnitude (mean |W|) and spread (std) — a blow-up/collapse read — plus the
        faithfulness metrics: feedback↔forward alignment (how far learned feedback has aligned with the
        readout; ~0 for random DFA, →1 for Kolen-Pollack) and the excitatory fraction under Dale's law."""
        out = {}
        for i, c in enumerate(self.cells):
            w = (c.rec_val if self._is_sparse(c) else c.Win.weight).detach()   # sparse: value vector
            out[f"L{i}_w_absmean"] = float(w.abs().mean())
            out[f"L{i}_w_std"] = float(w.std())
        out["head_w_std"] = float(self.head.weight.detach().std())
        out["attention"] = float(getattr(self, "attention", 1.0))              # self-adapting plasticity gate
        out["eff_lr_scale"] = float(getattr(self, "eprop_lr_scale", 2000.0) * getattr(self, "attention", 1.0))
        if getattr(self, "loss_ema", None) is not None:
            out["loss_ema"] = float(self.loss_ema)                             # the brain's learning-health baseline
        for k, v in (getattr(self, "_diag", None) or {}).items():             # leading-indicator diagnostics
            out[k] = round(float(v), 5)                                        # mem_mag/grad_mag/update_mag/surprise/rec_w_mag
        if getattr(self, "_fb", None):                                         # feedback↔forward alignment
            fb = self._fb[-1].detach().flatten().float(); hw = self.head.weight.detach().flatten().float()
            out["fb_align_cos"] = float(torch.dot(fb, hw) / (fb.norm() * hw.norm() + 1e-9))
        if getattr(self, "dale", False) and getattr(self, "_ei_sign", None):
            out["ei_frac_excit"] = float((torch.cat(self._ei_sign) > 0).float().mean())
        if getattr(self, "dendritic", False) or getattr(self, "two_compartment", False):
            out["burst_frac"] = float(getattr(self, "_burst_frac", 0.0))       # apical error bandwidth
        if getattr(self, "two_compartment", False):
            out["apical_mag"] = float(getattr(self, "_apical_mag", 0.0))       # apical-dendrite drive
        if getattr(self, "homeostasis", False) and getattr(self, "_thr_adapt", None):
            out["homeo_thr_mean"] = float(torch.cat(self._thr_adapt).mean())   # homeostatic threshold drift
        if getattr(self, "bounded_synapses", False):                          # fraction of synapses saturated
            wm = float(getattr(self, "w_max", 1.0)); c0 = self.cells[0]
            w0 = (c0.rec_val if self._is_sparse(c0) else c0.Wrec.weight).detach().abs()
            out["synapse_sat_frac"] = float((w0 >= 0.999 * wm).float().mean())
        return out

    def _ensure_ei(self):
        """Assign each neuron an excitatory (+1) or inhibitory (−1) type for Dale's law — a neuron's
        OUTGOING synapses all share one sign. Fixed at birth (like a real cell's E/I identity), ~80/20
        E:I (the cortical ratio); grows with the layer so new neurons also get a type."""
        g = torch.Generator(device="cpu").manual_seed(4242)
        if not hasattr(self, "_ei_sign"):
            self._ei_sign = []
        while len(self._ei_sign) < len(self.cells):
            h = self.cells[len(self._ei_sign)].hid
            self._ei_sign.append(torch.where(torch.rand(h, generator=g) < 0.8, 1.0, -1.0).to(self.device))
        for l, c in enumerate(self.cells):                                     # extend if a layer grew
            if self._ei_sign[l].numel() < c.hid:
                add = c.hid - self._ei_sign[l].numel()
                s = torch.where(torch.rand(add, generator=g) < 0.8, 1.0, -1.0).to(self.device)
                self._ei_sign[l] = torch.cat([self._ei_sign[l], s])

    # The faithfulness stack (§15.16): each biological constraint is an INDEPENDENT toggle (extends the
    # learn_rule switch), so its capability cost can be measured on the fidelity↔capability curve.
    _FAITH_KEYS = ("learn_rule", "eprop_lr_scale", "head_fanin_pow", "attn_sensitivity", "attn_rate_sens",
                   "feedback_mode", "fb_decay", "dale", "dendritic", "burst_thr", "bounded_synapses", "w_max",
                   "homeostasis", "target_rate", "homeo_lr", "btsp", "btsp_beta", "two_compartment", "g_ap",
                   "beta_ap", "som_baseline", "pv_gain", "diff_neuromod", "stochastic", "spike_noise", "metabolic",
                   "metabolic_lambda")

    @torch.no_grad()
    def set_faith(self, **kw):
        """Set any faithfulness toggle / hyperparameter LIVE (type, e.g. feedback_mode, and hyperparams).
        Each constraint is independent. Applies enable-time projections so a freshly-toggled constraint
        takes effect immediately. Returns the applied subset."""
        applied = {}
        for k, v in kw.items():
            if k not in self._FAITH_KEYS:
                continue
            if k == "learn_rule":                          # invalid values must NOT silently disable e-prop
                if v not in ("eprop", "bptt"): continue
            elif k == "feedback_mode":
                if v not in ("learned", "random"): continue
            else:
                cur = getattr(self, k, None)
                if isinstance(cur, bool):    v = bool(v)
                elif isinstance(cur, float): v = float(v)
                if k == "eprop_lr_scale":                  # clamp numeric hyperparams to sane ranges
                    v = max(0.1, v)
                elif k == "w_max":
                    v = max(1e-3, v)
                elif k in ("target_rate", "homeo_lr", "pv_gain", "g_ap", "fb_decay", "burst_thr", "head_fanin_pow",
                           "som_baseline", "spike_noise", "metabolic_lambda", "attn_sensitivity", "attn_rate_sens"):
                    v = max(0.0, v)
                elif k in ("btsp_beta", "beta_ap"):
                    v = min(0.9999, max(0.0, v))
            setattr(self, k, v); applied[k] = getattr(self, k)
        if self.dale:        self._project_dale()          # make weights Dale-compliant immediately
        if self.homeostasis: self._ensure_homeo()
        return applied

    def faith_config(self):
        """The current state of every faithfulness toggle/hyperparameter — the fidelity axis settings."""
        return {k: getattr(self, k, None) for k in self._FAITH_KEYS}

    def _ensure_homeo(self):
        """Per-neuron intrinsic-threshold offset for firing-rate homeostasis (metaplasticity). Grows
        with each layer so new neurons also get a homeostatic setpoint."""
        if not hasattr(self, "_thr_adapt"):
            self._thr_adapt = []
        while len(self._thr_adapt) < len(self.cells):
            self._thr_adapt.append(torch.zeros(self.cells[len(self._thr_adapt)].hid, device=self.device))
        for l, c in enumerate(self.cells):
            if self._thr_adapt[l].numel() < c.hid:
                pad = torch.zeros(c.hid - self._thr_adapt[l].numel(), device=self.device)
                self._thr_adapt[l] = torch.cat([self._thr_adapt[l], pad])

    @torch.no_grad()
    def _project_dale(self):
        """Re-impose Dale's law on the RECURRENT (neuron→neuron) connectome: every outgoing synapse
        takes the sign of its PREsynaptic neuron's type. Magnitude preserved; only the sign is clamped."""
        self._ensure_ei()
        for l, c in enumerate(self.cells):
            s = self._ei_sign[l]
            if self._is_sparse(c):
                c.rec_val.data.copy_(s[c.rec_col.long()] * c.rec_val.data.abs())   # rec_col = presynaptic
            else:
                c.Wrec.weight.data.copy_(s.unsqueeze(0) * c.Wrec.weight.data.abs())  # column = presynaptic

    # ---- §10 development: NEURONS FIXED at birth, SYNAPSES grow then prune ---- #
    # Biologically faithful: the neuron count is largely set at birth (neurogenesis is ~complete),
    # so it is a fixed, settable population (it can be large). Development is SYNAPTIC — childhood
    # synaptogenesis DENSIFIES the connectome, adolescence PRUNES the weak synapses. The SYNAPSE
    # count is therefore what evolves over the lifetime; the neuron count only changes if a caller
    # deliberately grows it (grow_neurons, e.g. from the API).
    def develop(self, allow_grow=True, add=64):
        self.age += 1
        self.lr = 2e-3 / (1 + self.age / 8.0)
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        phase = ("child" if self.age <= self.grow_until else
                 "adolescent" if self.age <= self.prune_until else "adult")
        grown = pruned = 0
        if allow_grow and phase == "child":                 # childhood: grow synapses (fixed neurons)
            grown = self.grow_synapses(getattr(self, "grow_syn_frac", 0.15))
        elif allow_grow and phase == "adolescent":          # adolescence: prune weak synapses
            pruned = self.prune(getattr(self, "prune_frac", 0.05))
        return dict(age=self.age, phase=phase, eta=round(self.lr, 5),
                    n_granule=self.hidden, neurons=self.neuron_count(),
                    synapses=self.active_synapse_count(), grown=grown, pruned=pruned)

    def _is_sparse(self, c):
        return isinstance(c, (SparseLIFCell, SparseALIFCell))

    def _plastic_targets(self):
        """DENSE weight matrices synapses live in: dense cells' input+recurrent, a sparse cell's
        dense input (layer 0), and always the readout head. The `_pmask` list aligns to these.
        (Sparse cells' recurrent/input connectomes are handled separately by `_sparse_pairs`.)"""
        t = []
        for c in self.cells:
            if self._is_sparse(c):
                if not c.sparse_in:
                    t.append(c.Win)                          # sparse cell, dense input projection
            else:
                t += [c.Win, c.Wrec]
        t.append(self.head)
        return t

    _prune_targets = _plastic_targets                       # back-compat alias

    def _sparse_pairs(self):
        """(value_Parameter, mask_buffer) pairs for the CSR connectomes of any sparse cells."""
        pairs = []
        for c in self.cells:
            if self._is_sparse(c):
                pairs.append((c.rec_val, c.rec_mask))
                if c.sparse_in:
                    pairs.append((c.in_val, c.in_mask))
        return pairs

    def neuron_count(self):
        """Total LIF neurons across the cortical stack (fixed unless deliberately grown)."""
        return int(sum(c.hid for c in self.cells))

    def synapse_capacity(self):
        """Total wire-able connections (active + silent). For sparse layers this is the CSR
        SUPERSET (fan-in cap), not H² — the honest capacity of a sparse connectome."""
        dense = sum(t.weight.numel() for t in self._plastic_targets())
        sparse = sum(v.numel() for v, _ in self._sparse_pairs())
        return int(dense + sparse)

    def active_synapse_count(self):
        """Active (non-silent) synapses across dense masks + sparse cell masks."""
        pm = getattr(self, "_pmask", None)
        dense = sum(int(m.sum()) for m in pm) if pm else sum(t.weight.numel() for t in self._plastic_targets())
        sparse = sum(int(m.sum()) for _, m in self._sparse_pairs())
        return int(dense + sparse)

    def _ensure_pmask(self, ws):
        if getattr(self, "_pmask", None) is None or [m.shape for m in self._pmask] != [w.shape for w in ws]:
            self._pmask = [torch.ones_like(w, dtype=torch.bool) for w in ws]

    @torch.no_grad()
    def _init_synapse_mask(self, density):
        """Seed the DENSE connectome sparsely (a `density` fraction active, rest zeroed). Sparse
        cells seed their own masks in their constructors, so this only handles the dense targets."""
        density = max(0.02, min(1.0, float(density)))
        ws = [t.weight for t in self._plastic_targets()]
        if density >= 1.0:
            self._pmask = [torch.ones_like(w, dtype=torch.bool) for w in ws]
            return
        g = torch.Generator(device="cpu").manual_seed(1234)
        self._pmask = []
        for w in ws:
            m = (torch.rand(w.shape, generator=g) < density).to(w.device)
            w.mul_(m); self._pmask.append(m)

    @torch.no_grad()
    def prune(self, frac=0.05):
        """§10 SYNAPTIC pruning (adolescence): silence the weakest active SYNAPSES (dense + sparse)
        while the NEURON count stays fixed. Mask-persistent (pruned synapses do not regrow)."""
        dense = self._plastic_targets(); dws = [l.weight for l in dense]
        self._ensure_pmask(dws)
        pairs = list(zip(dws, self._pmask)) + self._sparse_pairs()
        live = torch.cat([w[m].abs().flatten() for w, m in pairs if w.shape == m.shape])
        if live.numel() < 16:
            return 0
        # kthvalue (not torch.quantile — it raises above 2^24 elements, which a large sparse
        # connectome exceeds) gives the frac-quantile magnitude threshold.
        k = max(1, min(live.numel(), int(frac * live.numel())))
        thr = live.kthvalue(k).values
        n = 0
        for w, m in pairs:
            if w.shape != m.shape:
                continue
            cut = m & (w.abs() <= thr)
            m &= ~cut; w.mul_(m if w.dim() == m.dim() else m.view_as(w)); n += int(cut.sum())
        return n

    @torch.no_grad()
    def grow_synapses(self, frac=0.15):
        """§10 synaptogenesis (childhood): activate `frac` of the currently-SILENT connections
        (dense + sparse) with fresh small weights — the neuron count is unchanged. Returns count."""
        dense = self._plastic_targets(); dws = [l.weight for l in dense]
        if getattr(self, "_pmask", None) is None or [m.shape for m in self._pmask] != [w.shape for w in dws]:
            self._pmask = [torch.ones_like(w, dtype=torch.bool) for w in dws]
        n = synapse.grow_synapses(dws, self._pmask, frac)          # dense targets
        sp = self._sparse_pairs()
        if sp:
            sw = [v for v, _ in sp]; sm = [m for _, m in sp]
            fanins = self._sparse_fanins()                         # true fan-in for the 1-D vectors
            n += synapse.grow_synapses(sw, sm, frac, fanins=fanins)  # sparse cell connectomes
        return n

    def _sparse_fanins(self):
        """The true fan-in of each sparse (value, mask) pair, aligned to _sparse_pairs order — so
        newly-grown 1-D-vector synapses are scaled by the fan-in, not by nnz."""
        f = []
        for c in self.cells:
            if self._is_sparse(c):
                f.append(c.rec_fanin)
                if c.sparse_in:
                    f.append(c.in_fanin)
        return f

    @torch.no_grad()
    def _apply_prune_mask(self):
        """Re-zero silent synapses after an optimiser step so pruned/inactive ones stay silent
        (both dense targets and sparse cell value vectors)."""
        pm = getattr(self, "_pmask", None)
        if pm is not None:
            for lin, m in zip(self._plastic_targets(), pm):
                if lin.weight.shape == m.shape:
                    lin.weight.mul_(m)
                else:
                    self._pmask = None; break
        for v, m in self._sparse_pairs():
            v.mul_(m)

    _apply_synapse_mask = _apply_prune_mask         # clearer name for the same operation

    @torch.no_grad()
    def _resize_synapse_mask(self):
        """After a dense NEURON grow, pad the dense synapse mask to the new weight shapes (old kept,
        new connections active) without touching trained weights. Sparse cells resize in grow()."""
        if getattr(self, "_pmask", None) is None:
            return
        new = []
        for w, m in zip([t.weight for t in self._plastic_targets()], self._pmask):
            if w.shape == m.shape:
                new.append(m); continue
            nm = torch.ones_like(w, dtype=torch.bool)       # new connections start active
            nm[tuple(slice(0, s) for s in m.shape)] = m     # keep the old sparsity pattern
            new.append(nm)
        self._pmask = new

    @torch.no_grad()
    def grow(self, add=64):
        """Deliberate NEURON growth (grow_neurons): widen the top spiking layer + head with new LIF
        units (new head weights ~0 → function preserved). Neurons normally stay FIXED over the life;
        this is the explicit lever (API / big developmental step) to enlarge the population."""
        if self.model_gb() >= self.max_model_gb:
            return 0
        self.cells[-1].grow(add)               # add LIF neurons to the top layer
        old = self.hidden; new = old + add
        dev, dt = self.head.weight.device, self.head.weight.dtype
        nhead = nn.Linear(new, self.V).to(dev, dt)
        with torch.no_grad():
            nhead.weight.zero_(); nhead.weight[:, :old] = self.head.weight; nhead.bias.copy_(self.head.bias)
        self.head = nhead
        self.hidden = new
        self._mind = None
        self._resize_synapse_mask()            # keep the sparse connectome consistent, identity-safe
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return add

    grow_neurons = grow                        # explicit, self-documenting alias

    # ---- checkpoint -------------------------------------------------- #
    def save(self, path):
        # a per-layer arch descriptor so load() can rebuild sparse vs dense cells at their exact
        # sizes BEFORE load_state_dict (a sparse cell's structure can't be inferred from a Linear).
        arch = [dict(sparse=self._is_sparse(c), hid=int(c.hid), in_dim=int(c.in_dim),
                     sparse_in=bool(getattr(c, "sparse_in", False))) for c in self.cells]
        torch.save(dict(sd=self.state_dict(), opt=self.opt.state_dict(),
                        emb=self.emb_dim, hidden=self.hidden, layers=self.layers_n, arch=arch,
                        cell=self.cell_kind, readout=self.readout, read_alpha=self.read_alpha,
                        age=self.age, seen=self.seen_bytes, lr=self.lr, seq=self.seq,
                        syn_density=getattr(self, "syn_density", 1.0),
                        grow_syn_frac=getattr(self, "grow_syn_frac", 0.15),
                        prune_frac=getattr(self, "prune_frac", 0.05),
                        sparse_cfg=getattr(self, "sparse_cfg", None),
                        pmask=getattr(self, "_pmask", None),
                        faith=self.faith_config(),                         # every fidelity-axis setting
                        attention=getattr(self, "attention", 1.0),
                        loss_ema=getattr(self, "loss_ema", None),
                        fb=(getattr(self, "_fb", None) if self.feedback_mode == "learned" else None),
                        ei=getattr(self, "_ei_sign", None),                # Dale E/I typing (learned state)
                        thr_adapt=getattr(self, "_thr_adapt", None)), path)  # homeostatic thresholds

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.readout = d.get("readout", self.readout)
        self.read_alpha = d.get("read_alpha", self.read_alpha)
        self.cell_kind = d.get("cell", self.cell_kind)
        DenseCell = ALIFCell if self.cell_kind == "alif" else LIFCell
        SparseCell = SparseALIFCell if self.cell_kind == "alif" else SparseLIFCell
        sd = d["sd"]
        n_layers = d.get("layers", self.layers_n)
        arch = d.get("arch")
        self.emb_dim = d.get("emb", self.emb_dim)                 # rebuild the embedding at saved width
        self.E = nn.Embedding(self.V, self.emb_dim).to(self.device)
        cells = []
        for i in range(n_layers):
            pre = f"cells.{i}."
            a = arch[i] if arch else None
            is_sparse = (a["sparse"] if a else (pre + "rec_val") in sd)
            if is_sparse:
                hid = a["hid"] if a else (sd[pre + "rec_crow"].numel() - 1)
                in_dim = a["in_dim"] if a else 0
                sparse_in = a["sparse_in"] if a else ((pre + "in_val") in sd)
                c = SparseCell(in_dim, hid, rec_fanin=1, in_fanin=1, sparse_in=sparse_in,
                               syn_density=1.0, seed=0)
                self._realloc_sparse(c, sd, pre, in_dim=(a["in_dim"] if a else None))
                cells.append(c)
            else:
                w = sd[pre + "Win.weight"]                        # (out=hid_i, in=in_i)
                cells.append(DenseCell(w.shape[1], w.shape[0]))
        self.cells = nn.ModuleList(cells).to(self.device)
        self.hidden = sd["head.weight"].shape[1]     # head input = top-layer width
        self.head = nn.Linear(self.hidden, self.V).to(self.device)
        self.sparse_cfg = d.get("sparse_cfg", getattr(self, "sparse_cfg", None))
        self.opt = torch.optim.Adam(self.parameters(), lr=d.get("lr", self.lr))
        self.load_state_dict(sd)
        pm = d.get("pmask")
        self._pmask = [m.to(self.device) for m in pm] if pm else None    # keep pruned synapses pruned
        try: self.opt.load_state_dict(d["opt"])
        except Exception: pass
        self.age = d.get("age", 0); self.seen_bytes = d.get("seen", 0); self.lr = d.get("lr", self.lr)
        self.seq = d.get("seq", self.seq)
        self.syn_density = d.get("syn_density", getattr(self, "syn_density", 1.0))
        self.grow_syn_frac = d.get("grow_syn_frac", getattr(self, "grow_syn_frac", 0.15))
        self.prune_frac = d.get("prune_frac", getattr(self, "prune_frac", 0.05))
        for k, v in (d.get("faith") or {}).items():                      # restore every fidelity-axis setting
            setattr(self, k, v)
        self.attention = d.get("attention", 1.0); self.loss_ema = d.get("loss_ema", None)
        if d.get("fb") is not None:   self._fb = [t.to(self.device) for t in d["fb"]]        # aligned feedback
        if d.get("ei") is not None:   self._ei_sign = [t.to(self.device) for t in d["ei"]]   # Dale E/I typing
        if d.get("thr_adapt") is not None: self._thr_adapt = [t.to(self.device) for t in d["thr_adapt"]]
        if getattr(self, "dale", False): self._project_dale()
        return self

    @staticmethod
    def _realloc_sparse(c, sd, pre, in_dim=None):
        """Resize a freshly-built sparse cell's buffers/params to the saved CSR sizes so
        load_state_dict matches (the saved connectome may be a different nnz after growth).
        `in_dim` comes from the authoritative arch descriptor (col.max()+1 undercounts if the last
        input neuron is never wired)."""
        c.hid = sd[pre + "rec_crow"].numel() - 1
        c.register_buffer("rec_crow", torch.zeros_like(sd[pre + "rec_crow"]))
        c.register_buffer("rec_col", torch.zeros_like(sd[pre + "rec_col"]))
        c.register_buffer("rec_mask", torch.zeros_like(sd[pre + "rec_mask"]))
        c.register_buffer("rec_row", torch.zeros_like(sd[pre + "rec_row"]))
        c.rec_val = nn.Parameter(torch.zeros_like(sd[pre + "rec_val"]))
        if (pre + "in_val") in sd:
            c.sparse_in = True
            c.register_buffer("in_crow", torch.zeros_like(sd[pre + "in_crow"]))
            c.register_buffer("in_col", torch.zeros_like(sd[pre + "in_col"]))
            c.register_buffer("in_mask", torch.zeros_like(sd[pre + "in_mask"]))
            c.register_buffer("in_row", torch.zeros_like(sd[pre + "in_row"]))
            c.in_val = nn.Parameter(torch.zeros_like(sd[pre + "in_val"]))
            c.in_bias = nn.Parameter(torch.zeros_like(sd[pre + "in_bias"]))
            c.in_dim = in_dim if in_dim else (sd[pre + "in_col"].max().item() + 1 if sd[pre + "in_col"].numel() else 0)
        else:
            c.sparse_in = False
            c.in_dim = sd[pre + "Win.weight"].shape[1]
