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
from .spiking import LIFCell, ALIFCell, SparseLIFCell, SparseALIFCell, _adapt_intrinsic
from . import synapse
from .stdp import SpikingSTDP
from .predictive_coding import PredictiveCoding


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
        # §MEM2b learned population READ of the top gated slow compartment C: logits += C @ mem_read_w.t(). Gives the
        # byte-predictor DIRECT linear access to the slow bank (τ∈[8,160]) vs only the fixed κ threshold nudge.
        # zero-init (torch.zeros ⇒ NO global-RNG consumed ⇒ byte-identical off) + read_mem gate ⇒ default off is a no-op.
        self.read_mem = False
        self.mem_read_w = nn.Parameter(torch.zeros(self.V, hidden))
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
        self.eprop_lr_scale = 1000.0           # the BASE rate; the EFFECTIVE rate self-adapts via `attention`.
        # 2000 was too hot: on a fresh net the held-out bpb BOUNCES and diverges (measured 4.71, curve 4.84→5.17);
        # 1000 is stable and ~0.4 bpb better (4.28). The self-adapting attention brake alone did not damp the 2000
        # base fast enough early in training — a lower base is the safer operating point. (Diagnostic: e-prop is the
        # convergence ceiling here, not the encoder/decoder: same arch under BPTT+Adam reaches 3.26 and still
        # descends, embedding 64→256 changes nothing — the ~4.3 floor is the price of faithful LOCAL learning.)
        self._fanin_pow = 1.0                  # divide the RECURRENT/input update by N_j^p (p=1 → width-invariant descent)
        # The readout HEAD reads the WHOLE hidden width (fan-in = hidden), so the same p=1 fan-in norm divides its
        # update by ~hidden and STARVES it at scale: at hidden=128000 the head moves ~2.8e-8/step and stays frozen
        # at its random init → the whole net can only reach the byte-frequency baseline (measured: head_w_std≈init,
        # fb_align_cos≈0, understanding≈0). The head is a simple readout, not a width-invariant recurrent map, so it
        # gets a gentler power — live-tunable; 0 = no fan-in norm (fastest, UNSTABLE). A/B (hidden=2000, 300 steps):
        # pow 1.0 → bpb 4.02 head frozen; pow 0.7 → bpb 2.95 acc 0.49 (WIN); pow ≤0.5 → explodes. 0.7 is the sweet spot.
        # BUT the sweet spot DRIFTS with width (0.7@2000 works, freezes@128000; 0.45@128000 explodes) — a pure power
        # law is the wrong model for a readout. head_norm="energy" (NLMS) replaces ÷hidden^pow with ÷‖top_v‖²: since
        # Δlogit_i = μ·err_i·‖v‖²/(‖v‖²+ε) → μ·err_i, the ‖v‖² CANCELS ⇒ the per-step logit move is invariant to BOTH
        # width N and sparsity ρ (‖v‖²≈ρN⟨v²⟩ auto-discounts the ~96% silent membranes). ONE head_lr_scale then
        # transfers from hidden=256 to 256000. Update-only (no forward change), e-prop-local, byte-identical when off.
        self.head_fanin_pow = 0.7
        # OPTIMIZER on the (correct, faithful) local e-prop gradient. The gradient dE/dw=Σ_t L_t·e_t is verified
        # correct (eligibility trace + learning signal + routing audited faithful); the BUG was the apply-tail —
        # ÷(denom·fan_in^p)+±0.02 clamp+fixed-lr, a crude per-tensor mis-scaling that froze the embedding (÷denom·÷emb
        # ≈÷5e4), froze the head WEIGHTS (÷hidden^0.7) while the head BIAS (÷denom only) raced to the byte-marginal
        # ⇒ logits collapse to the prior ⇒ bpb stuck at ~4.5. Correct e-prop (Bellec 2020) feeds the LOCAL gradient
        # to ADAM (per-weight moment normalization) — which reads ONLY each synapse's own gradient history, so it is
        # fully LOCAL/faithful (no backprop, no weight-transport). learn_opt="adam" gives every weight a ~adam_lr
        # step regardless of its raw-gradient magnitude, unfreezing E/head/recurrent together (a single global lr
        # cannot — it just explodes, measured). gate·attention·DA still multiplies the step (mechanisms modulate).
        self.learn_opt = "legacy"             # "legacy"=÷fan_in+clamp+fixed-lr (byte-identical); "adam"=per-weight Adam
        self.adam_lr = 1e-3                    #   Adam step size on the local e-prop gradient
        # §17 glia metaplasticity under Adam: a per-neuron pgain that SCALES THE RAW GRADIENT is a no-op under
        # Adam — m and √u both carry the same per-row factor, so it cancels in the m̂/√û ratio (a chronically
        # over-firing astrocyte domain then fails to throttle its own synapses). glia_pgain_post_adam moves the
        # metaplastic gain from the pre-Adam gradient onto the POST-Adam STEP (a per-post-neuron learning-rate
        # multiplier that Adam's RMS cannot normalise away). OFF ⇒ pgain stays a gradient scale (byte-identical;
        # legacy path unchanged, where the ±clamp makes pre-grad vs post-step genuinely differ).
        self.glia_pgain_post_adam = False
        self._adam_state = {}                  #   id(param) -> [m, u] fp32 first/second-moment EMAs (per-weight, local)
        self._adam_t = 0                       #   Adam bias-correction step counter
        self.head_norm = "power"              # "power"=÷hidden^head_fanin_pow (current, byte-identical); "energy"=NLMS ÷‖top_v‖²
        self.head_lr_scale = 1.0              # NLMS μ (energy mode): μ_eff=μ·gate·attn·(0.5+da)≤~1.3<2 (mean-square stable).
        #   A/B (hidden 256..4096): μ=0.3 bpb~5.5 (too slow), μ=1.0→4.75, μ=2.0→3.9 (below the 4.5 byte floor). Δz_rms
        #   width-ratio 1.05-1.15 at all μ. Raise toward 2.0 live for faster descent (still <2 stable margin).
        self.head_energy_eps = 1e-2           # ε as a fraction of EMA(mean readout energy) — silent-top-layer guard
        self._head_e_ema = 0.0                # persistent EMA of mean readout energy (0 = uninitialised)
        # SELF-ADAPTING plasticity (§15.17): the learning rate is NOT a fixed dial to hand-tune — it is
        # gated by `attention`, which tracks the brain's OWN learning health (loss vs. its running
        # baseline). A loss spike above baseline (a shock / struggling) DROPS attention → the update
        # shrinks → the representation is protected and re-learns gently (self-healing); at/below baseline
        # attention rises to engage. This is the Yerkes–Dodson arousal→plasticity curve; it removes the
        # need to manually chase eprop_lr_scale and would have auto-damped the Dale/lr excursions.
        self.attention = 1.0
        # §15.18 pair-based asymmetric STDP (timing-refined sibling of eps_rec/eps_in). DEFAULT OFF; single
        # source of truth = self.stdp.on (routed by set_faith). Holds only float scalars + device/dtype refs.
        self.stdp = SpikingSTDP(device=device, dtype=dtype)
        self.loss_ema = None
        self.attn_sensitivity = 0.8            # how sharply attention drops with above-baseline loss
        self.attn_rate_sens = 0.25             # how sharply attention drops when firing runs > 2× the homeostatic
                                               # target (the over-excitation brake; loss-blind drift protection)
        self.attn_mem_sens = 0.5               # how sharply attention drops when the TOP-layer membrane magnitude
                                               # |v| runs above mem_target — a REPRESENTATION runaway that the
                                               # rate/loss brakes are blind to (mem_mag inflates while firing sits
                                               # on the homeostatic target). 0 disables (byte-identical to before).
        self.mem_target = 1.0                  # absolute |v| anchor: healthy top-membrane mem_mag < 1; bpb explodes
                                               #   past ~1.5. FIXED (not a self-following EMA) so a slow drift cannot
                                               #   escape it the way glia.auto_target normalises its rate reference.
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
        self.bounded_synapses = False          # Fusi bounded synapses: weights clamped so they can't PUMP the
        self.w_max = 3.0                       #   membrane. Fan-in-RELATIVE by default (bound = w_max/√fanin): the
        self.w_max_relative = True             #   clamp scales like the init (1/√fanin), holding the recurrent map
        #   near its STABLE init spectral radius at ANY width. A fixed ±1 is ~32× the dense init (≈0.031) and ~8× the
        #   sparse init (≈0.125) → NON-binding, so under Dale/coherent learning the recurrence v←β·v·(1−z)+W·z can
        #   reach ρ(W)≫1/β and go unstable, inflating |v| (the everything-on mem_mag runaway) while every weight sits
        #   legally inside ±1. 3× init = ample learning headroom yet bounded; lower toward ~2 for a hard ρ<1/β
        #   guarantee, raise if descent stalls. Absolute ±w_max when w_max_relative=False (byte-identical legacy).
        self.homeostasis = False               # intrinsic firing-rate homeostasis (metaplasticity): each
        self.target_rate = 0.08                #   neuron's threshold drifts to hold a target spike rate
        self.homeo_lr = 0.02                   #   (Turrigiano) — keeps a continual net off silence/saturation
        # MEMBRANE-MAGNITUDE intrinsic homeostasis (§HARM). Threshold homeostasis regulates the spike RATE, but the
        # everything-on collapse is a |v| runaway with the rate pinned ON target, so the rate channel is blind to it;
        # worse, RAISING thr to hold the rate INFLATES the membrane (mem_mag≈0.57·thr) — thr_adapt climbing is a
        # co-symptom, not a brake. This adds a second homeostatic term on the SAME threshold: when a neuron's mean
        # |v| exceeds mem_homeo_target it LOWERS the threshold so the neuron fires and RESETS, discharging v through
        # its own reset (the membrane's natural clamp) — the correct-sign actuator that needs NO change to the leak
        # β / integration τ (a substrate change de-calibrates the readout, measured to break learning). The RATE
        # channel is the restoring force (more firing → it raises thr back), so the two settle at a slightly-above-
        # target rate that pins |v|≈mem_homeo_target. relu(|v|/target−1) ⇒ 0 while |v|≤target ⇒ byte-identical when
        # quiescent; the default target 3.0 sits ABOVE the healthy high-|v| operating band (homeostasis-alone runs
        # stably at |v|≈2–2.5 and still learns) so it engages only on a genuine runaway, not on a stably-elevated
        # membrane. Complements the attention membrane-brake above (that throttles dW; this discharges v).
        self.mem_homeostasis = True            #   membrane-magnitude channel of intrinsic homeostasis (threshold actuator)
        self.mem_homeo_target = 3.0            #   |v| setpoint for the magnitude channel; ABOVE the healthy band (byte-id below it)
        self.mem_homeo_lr = 0.05               #   integral gain of the magnitude→threshold term (drift-coupled via `excess`)
        self.homeo_lr_drift = 0.0              #   optional: scale the RATE homeo gain by |rate−target| (0=off, byte-id)
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
        # §HARM rate-relative metabolic penalty: the constant `+mlam·z` term is a sign-coherent DC bias on the
        # e-prop gradient that Adam (which normalises magnitude but preserves sign) converts into a full-strength
        # downward step EVERY tick → silences the top membrane (mem_mag→0.1-0.2, no learning). Gate the penalty by
        # per-neuron OVER-firing relu(rate/target−1) (like intrinsic homeostasis) so it is ZERO at/below target (no
        # DC for Adam to amplify, stable fixed point at rate=target) and only genuine over-drive gets a
        # proportional, SELF-RELEASING push. DEFAULT OFF ⇒ byte-identical to the constant term.
        self.metab_rate_relative = False       # engage the rate-relative (one-sided) metabolic gate
        self.metab_rate_cap = 4.0              # cap on relu(rate/target−1) so the over-target bias stays a minority of Adam's RMS
        # ── §HARM EXCITATION-PRESSURE BUS P — the unifying coordinator ─────────────────────────────────────
        # The per-mechanism retunes above (metab_rate_relative, glia_pgain_post_adam, mem_homeostasis) each
        # defuse ONE Adam-amplified suppressor, but the everything-on collapse is COLLECTIVE: under Adam the
        # WHOLE suppressive stack (metabolic price, glia brake, rate/mem homeostasis, PV divisive gain) can
        # co-pin a SILENCED membrane at mem_mag≈0.1 — a near-absorbing state that no single control releases
        # (removing any ONE does not recover; the measured `−metabolic→0.15 / −glia→0.11` facts). P is ONE
        # absolute-anchored over-drive signal ∈[0,excite_p_cap] that gates EVERY suppressive control together:
        # each engages ∝min(1,P) and RELEASES to 0 the instant the top representation is calm OR silenced
        # (mem_mag ≤ mem_target AND rate ≤ target_rate ⇒ P=0). So a silenced net sees ZERO suppression and its
        # membrane is free to climb back out; only a genuine runaway (mem_mag/rate ABOVE their FIXED anchors —
        # never a self-following EMA that would calibrate the runaway away) drives P>0 and pushes the brakes
        # on, self-releasing the moment it settles at the anchor (negative feedback, stable fixed point at the
        # anchor). This is the coordination principle in one bus: absolute-anchored PRESSURE, engaged only when
        # drive is genuinely too high, released when calm. DEFAULT OFF ⇒ every gated control keeps its current
        # (nominal) strength ⇒ byte-identical; when ON, the gate can only REDUCE suppression below today's
        # constant level (min(1,P)≤1), never strengthen it past nominal — so it can undo the silence, not add
        # instability. ANCHOR NOTE: under Adam the healthy band is mem_mag≈1.4–2.0, so the Adam everything-on
        # config sets mem_target≈2.0 (this re-anchors BOTH the bus AND the attention mem-brake to one setpoint);
        # left at the legacy 1.0 the bus would wrongly engage inside the healthy band.
        self.excite_pressure = False           # master toggle for the P coordinator (default off ⇒ byte-identical)
        self.excite_p_cap = 3.0                # clamp on P (matches the attention over/over_mem ±3 range)
        self.excite_p_gain = 1.0               # sensitivity: P = excite_p_gain · max(mem_over, rate_over)
        self.homeo_leak = 0.0                  # §HARM P-gated anti-windup leak on _thr_adapt: leaks ∝(1−min(1,P)) so the
        #   threshold integrator UNWINDS toward baseline when the net is calm/silenced (P→0) — the recovery path out of a
        #   homeostasis wind-up silence — and HOLDS (no leak) under genuine over-rate pressure. 0 ⇒ byte-identical.
        self.homeo_clamp = 0.0                 # §HARM hard anti-windup backstop: |_thr_adapt|≤homeo_clamp (0⇒off⇒byte-identical).
        #   Under Adam the RATE integrator winds up ~1e3× faster (proper-strength plant) and can push thr into the surrogate
        #   dead-zone (|v−thr|≫0 ⇒ ψ→0 ⇒ g=Lsig·ψ→0 ⇒ frozen, acc→0.008). A ±0.6 clamp (thr0=1.0) keeps effective thr∈[0.4,1.6],
        #   inside the Adam healthy mem band, so ψ stays alive and the silence point is STRUCTURALLY unreachable — unlike the
        #   leak (which only RELEASES when calm), the clamp bounds even a transient over-rate excursion. Unconditional backstop.
        self._excite_p = 0.0                   # last step's pressure (cached for life.py glia gate + next step's in-loop controls)
        self._mind = None                      # persistent per-layer state = stream of thought
        self._last = None
        # §10: the NEURON count is fixed at birth; the SYNAPSE count is what develops. Seed a
        # sparse connectome (syn_density of connections active) that childhood then densifies.
        self.syn_density = syn_density
        # §PC third learning rule (Rao-Ballard/Friston). DEFAULT OFF via learn_rule='eprop'; pc.on gates the
        # PC EXTRAS (precision + inference). dtype passed through (stats stay fp32 like the e-prop grads).
        self.pc = PredictiveCoding(device, dtype)
        self._init_synapse_mask(syn_density)
        from .synaptic_stp import SpikingSTP
        self.stp = SpikingSTP(self.device)     # §17 short-term synaptic plasticity controller (DEFAULT OFF)

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
                if getattr(self, "stp", None) is not None and self.stp.on:
                    self.stp.reset(self.cells, B)     # §17: fresh context ⇒ (u,x) to rest (continuity across think/generate)
            top_mem = None
            for i, c in enumerate(self.cells):
                spikes, mems, states[i] = c.run_seq(inp, states[i], stp=getattr(self, "stp", None), stp_layer=i)
                inp = spikes; top_mem = mems
            read = self._readout(top_mem, inp)             # (B,T,hid)
            logits = self.head(read)                       # (B,T,V) in one matmul
            if getattr(self, "read_mem", False):           # §MEM2b add the learned read of the top slow compartment
                css = getattr(self.cells[-1], "_css_seq", None)
                if css is not None: logits = logits + F.linear(css, self.mem_read_w)
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
        _rule = getattr(self, "learn_rule", "bptt")
        if _rule == "pc":                                      # §PC predictive-coding route (peer of e-prop)
            return self.learn_pc(text, epochs=epochs, bs=bs, max_steps=max_steps, seq=seq,
                                 on_step=on_step, gate=gate, tone=tone)
        if _rule == "eprop":                                   # faithful forward-in-time route
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

    @torch.no_grad()
    def learn_pc(self, text, epochs=1, bs=16, max_steps=12, seq=None, on_step=None, gate=1.0, tone=None):
        """§PC · train the cortex by hierarchical predictive coding (Rao-Ballard/Friston). Identical windowing
        to learn_eprop; each window runs _pc_step (= _eprop_step with the trace off + precision weighting)."""
        data = text if isinstance(text, list) else self.to_bytes(text)
        seq = seq or self.seq
        if len(data) <= seq + 1:
            return None
        self._ensure_feedback(); self.pc.ensure(self.cells)    # B_l = the PC generative weights; pad sig2/prec
        t = torch.tensor(data, device=self.device); n = t.numel()
        first = last = None
        for _ in range(epochs):
            steps = max(1, min(max_steps, (n - seq) // (bs * seq) + 1))
            for _s in range(steps):
                i = torch.randint(0, n - seq - 1, (bs,), device=self.device)
                x = torch.stack([t[k:k + seq] for k in i])
                y = torch.stack([t[k + 1:k + seq + 1] for k in i])
                loss = self._pc_step(x, y, gate, tone=tone)
                last = loss
                if first is None: first = last
                if on_step is not None: on_step(_s + 1, steps, last)
        self.seen_bytes += n
        return (first if first is not None else 0.0, last if last is not None else 0.0)

    @torch.no_grad()
    def _pc_step(self, x, y, gate=1.0, tone=None):
        """§PC one predictive-coding step — routes through _eprop_step's shared spmm/sddmm/edge_reduce + the
        width-invariant _upd tail with pc=True (eb→0 instantaneous ε, precision-weighted error, optional
        inference relaxation). Single code path with e-prop ⇒ no drift; e-prop is untouched (pc=False)."""
        return self._eprop_step(x, y, gate, tone=tone, pc=True)

    _EP_CHUNK = 1 << 26                                        # cap on any transient (chunk, nnz) buffer

    @torch.no_grad()
    def _eprop_step(self, x, y, gate=1.0, tone=None, pc=False):
        """One e-prop gradient step over a (B,T) window — pure PyTorch, NO O(H²) anywhere. Eligibility
        traces are per-neuron (O(H)); the recurrent grad is accumulated PER SYNAPSE by gather/scatter
        (O(nnz) for a sparse cortex, so it scales to hundreds of thousands of neurons). Timestep-outer
        (== the layer-outer forward), online, gated by the neuromodulator. No autograd, no BPTT."""
        B, T = x.shape
        cells = self.cells; dev = self.device
        if getattr(self, "stp", None) is not None and self.stp.on:
            self.stp.reset(cells, B)                            # §17: each e-prop window starts (u,x) at rest (like v/eps)
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
        # §MEM2 input-gated slow compartment: per-neuron state C (|c|≤1) + per-SYNAPSE eligibility ec (clone of the
        # ALIF ea shapes/recursion). Allocated ONLY where gated_slow is on ⇒ default-off is byte-identical & free.
        gs = [getattr(c, "gated_slow", False) for c in cells]
        if any(gs):
            C = [torch.zeros(B, c.hid, device=dev) if gs[i] else None for i, c in enumerate(cells)]
            ec_rec = [torch.zeros(B, c.rec_val.numel(), device=dev) if (gs[i] and sp(c))
                      else (torch.zeros(B, c.hid, c.hid, device=dev) if gs[i] else None)
                      for i, c in enumerate(cells)]
            ec_in = [None] * len(cells)
            for i, c in enumerate(cells):
                if not gs[i]: continue
                if sp(c) and c.sparse_in: ec_in[i] = torch.zeros(B, c.in_val.numel(), device=dev)
                else: ec_in[i] = torch.zeros(B, c.hid, c.in_dim, device=dev)
        else:
            C = ec_rec = ec_in = [None] * len(cells)
        g_rec = [torch.zeros_like(c.rec_val) if sp(c) else torch.zeros_like(c.Wrec.weight) for c in cells]
        g_in, g_in_b = [], []
        for c in cells:
            if sp(c) and c.sparse_in: g_in.append(torch.zeros_like(c.in_val)); g_in_b.append(torch.zeros_like(c.in_bias))
            else: g_in.append(torch.zeros_like(c.Win.weight)); g_in_b.append(torch.zeros_like(c.Win.bias))
        gHead = torch.zeros_like(self.head.weight); gHead_b = torch.zeros_like(self.head.bias)
        # §MEM2b learned-read gradients (both INSTANTANEOUS — read params are OUTSIDE the c-recurrence ⇒ no trace):
        #   κ_j (per-neuron read gain): thr+=κ·C ⇒ ∂z/∂κ=−ψ·C ; drive+=κ·C ⇒ +ψ·C.  R (head-read): sibling of gHead.
        _lr_read = [gs[i] and getattr(c, "learn_read", False) for i, c in enumerate(cells)]
        g_kappa = [torch.zeros(c.hid, device=dev) if _lr_read[i] else None for i, c in enumerate(cells)]
        # §MEM2c learned write content w=tanh(a·I+d): 2 O(H) per-neuron RTRL traces (= ∂C_j/∂a_j, ∂C_j/∂d_j) + grads.
        _lw = [gs[i] and getattr(c, "learn_write", False) for i, c in enumerate(cells)]
        ea_wa = [torch.zeros(B, c.hid, device=dev) if _lw[i] else None for i, c in enumerate(cells)]
        ea_wd = [torch.zeros(B, c.hid, device=dev) if _lw[i] else None for i, c in enumerate(cells)]
        g_wa = [torch.zeros(c.hid, device=dev) if _lw[i] else None for i, c in enumerate(cells)]
        g_wd = [torch.zeros(c.hid, device=dev) if _lw[i] else None for i, c in enumerate(cells)]
        # §MEM3 fast-Hebbian relational store: F per-edge fast weight (STATE, not param; FIXED Hebbian write, no grad).
        # rho_j learned per-neuron read gain (instantaneous grad, sibling of κ_j). Sparse cortex only (like STDP).
        fm = [sp(c) and getattr(c, "fast_mem", False) for c in cells]
        Ffast = [torch.zeros(B, c.rec_val.numel(), device=dev) if fm[i] else None for i, c in enumerate(cells)]
        recall_now = [None] * len(cells)
        _lr_fm = [fm[i] and getattr(cells[i], "mem_learn_read", False) for i in range(len(cells))]
        g_rho = [torch.zeros(c.hid, device=dev) if _lr_fm[i] else None for i, c in enumerate(cells)]
        read_mem = getattr(self, "read_mem", False)
        gMemRead = torch.zeros_like(self.mem_read_w) if read_mem else None
        gE = torch.zeros_like(self.E.weight)                   # the sensory byte-embedding also learns (e-prop)
        he_on = getattr(self, "head_norm", "power") == "energy"   # §NLMS width/sparsity-invariant readout
        v_energy = [torch.zeros((), device=dev, dtype=torch.float32) for _ in cells] if he_on else None
        learned_fb = getattr(self, "feedback_mode", "learned") == "learned"
        gFB = [torch.zeros_like(self._fb[l]) for l in range(len(cells) - 1)] if learned_fb else None
        dendritic = getattr(self, "dendritic", False)          # apical burst-coded error delivery?
        burst_thr = float(getattr(self, "burst_thr", 0.5))
        burst_frac = 0.0
        btsp = getattr(self, "btsp", False)                    # long (behavioral-timescale) eligibility?
        ebeta = float(getattr(self, "btsp_beta", 0.98))
        homeo = getattr(self, "homeostasis", False)            # intrinsic firing-rate homeostasis?
        mem_homeo = homeo and bool(getattr(self, "mem_homeostasis", True))   # §HARM membrane-magnitude threshold channel
        bounded = getattr(self, "bounded_synapses", False)     # Fusi bounded weights?
        twocomp = getattr(self, "two_compartment", False)      # UNIFIED apical/interneuron/neuromod circuit?
        g_ap = float(getattr(self, "g_ap", 0.15)); beta_ap = float(getattr(self, "beta_ap", 0.9))
        som_b = float(getattr(self, "som_baseline", 0.5)); pv_g = float(getattr(self, "pv_gain", 0.3))
        ap = [torch.zeros(B, c.hid, device=dev) for c in cells] if twocomp else None   # apical dendrite state
        infer_g = float(self.pc.infer_gain) if pc else 0.0     # §PC inference-relaxation gain (0 = learning-only)
        pc_prev = [torch.zeros(B, c.hid, device=dev) for c in cells] if (pc and infer_g > 0.0) else None
        pc_err_out = 0.0                                        # §PC output prediction-error accumulator
        intern = getattr(self, "interneurons", None)           # §17 spiking PV/SOM/VIP pools (controller)
        use_intern = twocomp and intern is not None and getattr(intern, "on", False)
        if use_intern: intern.begin(B, ref=v[0])               # capture device/dtype, reset per-step membranes
        # differentiated neuromodulation: the 4 tones gate 4 distinct pathways (else one scalar `gate`=ACh).
        diffnm = getattr(self, "diff_neuromod", False) and isinstance(tone, dict)
        da = float(tone.get("da", 0.5)) if diffnm else 0.5     # DA → reward-modulated plasticity magnitude
        ne_gain = (0.5 + float(tone.get("ne", 1.0))) if diffnm else 1.0   # NE → somatic gain (surprise/attention)
        ht = float(tone.get("ht", 0.5)) if diffnm else 0.5     # 5-HT → patience (apical + eligibility timescale)
        ht_pat = (0.85 + 0.3 * ht) if diffnm else 1.0          # stretches the eligibility window (works w/o twocomp)
        if diffnm: beta_ap = min(0.99, beta_ap * (0.7 + 0.6 * ht))        # 5-HT → apical patience (twocomp)
        pl = getattr(self, "_plateau", None)                   # §17 NMDA apical plateau controller (live handle)
        plateau = twocomp and pl is not None and pl.on         # gated: needs the two-compartment apical circuit
        if plateau:
            p_thr, p_gain, rho_p, pcpl = float(pl.p_thr), float(pl.p_gain), float(pl.rho_p), float(pl.btsp_couple)
            pdur = max(1.0, float(pl.dur) * (0.7 + 0.6 * ht)) if diffnm else float(pl.dur)   # 5-HT patience → longer plateau
            plat = [torch.zeros(B, c.hid, device=dev, dtype=v[0].dtype) for c in cells]      # per-neuron plateau (rides on ap[l])
            pclk = [torch.zeros_like(p) for p in plat]          # all-or-none refractory window countdown
            plat_rate = 0.0; apd_top_mag = 0.0
        stoch = getattr(self, "stochastic", False)             # probabilistic (noisy) spiking?
        snoise = float(getattr(self, "spike_noise", 0.1))
        metab = getattr(self, "metabolic", False)              # spike-rate energy penalty on the update?
        mlam = float(getattr(self, "metabolic_lambda", 0.01))
        astro_pg = getattr(self, "_astro_pgain", None)         # §17 glia: per-post-neuron metaplastic gain (list[H_l] or None)
        astro_mm = float(getattr(self, "_astro_metab_mult", 1.0) or 1.0)   # glia metabolic-scarcity multiplier on mlam
        intr = [getattr(c, "intrinsic_exc", False) for c in cells]   # §INTRINSIC per-cell excitability (no neuron dies)
        astro_on = bool(getattr(self, "_astro_on", False)) or any(intr)   # intrinsic needs the per-neuron rate accumulator
        # §HARM RATE-RELATIVE metabolic gate (precomputed ONCE per step from the PRIOR step's rate, like glia.a and
        # _thr_adapt): a per-neuron factor relu(rate/target−1)∈[0,cap] so the metabolic penalty is one-sided — zero
        # at/below target (no DC bias for Adam to amplify into silence), proportional only above it. Ratio source:
        # glia's per-neuron activity field a_l (=rate/target, surfaced as _astro_a) when glia is on; else a
        # self-maintained per-neuron rate EMA /target. No rate estimate yet ⇒ factor 0 (safe release, no penalty).
        metab_rel = None
        if metab and bool(getattr(self, "metab_rate_relative", False)):
            _mrcap = float(getattr(self, "metab_rate_cap", 4.0))
            _mratio = getattr(self, "_astro_a", None)          # glia per-neuron activity ratio a_l (a=1 ⇔ at target)
            if _mratio is None:
                _mre = getattr(self, "_metab_rate_ema", None)  # fallback (glia off): our own per-neuron rate EMA / target
                if _mre is not None:
                    _mtr = max(float(getattr(self, "target_rate", 0.08)), 1e-3)
                    _mratio = [r / _mtr for r in _mre]
            metab_rel = []
            for l, c in enumerate(cells):                      # per-neuron relu(ratio−1), width-guarded + capped
                if _mratio is not None and l < len(_mratio) and _mratio[l].numel() >= c.hid:
                    metab_rel.append((_mratio[l][:c.hid].to(dev).float() - 1.0).clamp(0.0, _mrcap))
                else:
                    metab_rel.append(torch.zeros(c.hid, device=dev))   # no/stale rate → release (no DC bias)
        # §HARM excitation-pressure bus: the IN-LOOP suppressive controls (metabolic price @ the Lsig term, PV
        # divisive gain @ the drive) run BEFORE this step's P is known, so they gate on the PRIOR step's pressure
        # (causal; P is a slow controller). _p_prev∈[0,1] is the in-loop gate — OFF ⇒ 1.0 (nominal, byte-identical);
        # ON ⇒ min(1, last P) so both controls release to 0 on a calm/silenced net and reach nominal only under
        # ≥2×-anchor over-drive. Tail controls (glia pgain, homeostasis) use THIS step's fresh P computed below.
        _p_on = bool(getattr(self, "excite_pressure", False))
        _p_prev = min(1.0, float(getattr(self, "_excite_p", 0.0))) if _p_on else 1.0
        if astro_on:                                           # §17 per-neuron accumulators for glia.sense()/sense_mem()
            astro_zsum = [torch.zeros(c.hid, device=dev) for c in cells]   # firing (rate channel)
            astro_vsum = [torch.zeros(c.hid, device=dev) for c in cells]   # |v| (ABSOLUTE membrane-runaway channel)
        if homeo:
            self._ensure_homeo(); spk_sum = [torch.zeros(c.hid, device=dev) for c in cells]
        lr = self.lr * (float(self.pc.pc_lr_scale) if pc else getattr(self, "eprop_lr_scale", 2000.0))
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
        # §15.18 STDP: fast spike-time traces (timing-refined siblings of eps_rec/eps_in) + per-edge deltas.
        # Reallocated every step from current c.hid / g_rec shapes → a mid-life grow() cannot desync it.
        sd = getattr(self, "stdp", None); do_stdp = sd is not None and sd.on
        if do_stdp:
            lam_p, lam_m = sd.decay()
            xr_p = [torch.zeros(B, c.hid, device=dev) for c in cells]      # per-layer PRE trace  (λ₊)
            xr_m = [torch.zeros(B, c.hid, device=dev) for c in cells]      # per-layer POST trace (λ₋)
            xi_p = [torch.zeros(B, c.in_dim, device=dev) for c in cells]   # input-matrix PRE trace (λ₊)
            stdp_rec = [torch.zeros_like(g) for g in g_rec]                # shares g_rec shape/partition (FSDP)
            stdp_in = [torch.zeros_like(g) for g in g_in]
            ltp_acc = ltd_acc = 0.0
        tot_loss = 0.0
        for tt in range(T):
            layer_in = inp[:, tt]; psi = []
            for l, c in enumerate(cells):
                z_prev = z[l]
                z_eff = z_prev * self.stp.transmit(l, z_prev) if (getattr(self, "stp", None) is not None and self.stp.on) else z_prev  # §17
                if sp(c):
                    rec = spmm(c.rec_val * c.eff_rec_mask(), c.rec_col, c.rec_row, z_eff, c.hid)   # §17 laminar-thinned
                    if c.sparse_in:
                        pre = spmm(c.in_val * c.eff_in_mask(), c.in_col, c.in_row, layer_in, c.hid) + c.in_bias
                    else:
                        _ir = getattr(c, "lam_in_row", None)
                        pre = c.Win(layer_in) if _ir is None else c.Win(layer_in) * _ir
                else:
                    _rw = getattr(c, "lam_rec_w", None)
                    rec = (z_eff @ (c.Wrec.weight * _rw).t()) if _rw is not None else c.Wrec(z_eff)
                    _ir = getattr(c, "lam_in_row", None)
                    pre = c.Win(layer_in) if _ir is None else c.Win(layer_in) * _ir
                drive = pre + rec
                if gs[l]:                                      # §MEM2 gate on the RAW drive I=pre+rec (∂I/∂W=z_pre ⇒ local)
                    _ig_l = torch.sigmoid(c.k_g * drive + c.b_g_vec)          # NMDA write conductance (0,1)
                    if _lw[l]:                                                # §MEM2c learned per-neuron write w=tanh(a·I+d)
                        _Iraw_l = drive                                       #   raw I for the write-param traces
                        _wc_l = torch.tanh(c.write_a * drive + c.write_d)
                    else:
                        _wc_l = torch.tanh(drive)                             # bounded write (-1,1)
                    _cprev_l = C[l]                                           # c_{t-1} (used by the eligibility)
                    C[l] = (1.0 - _ig_l) * _cprev_l + _ig_l * _wc_l          # convex gated slow state ⇒ |c|≤1
                    if getattr(c, "gs_read", "threshold") == "drive":
                        drive = drive + c.gs_kappa * C[l]                    # UPGRADE: plateau current into the membrane
                if fm[l]:                                      # §MEM3 content-addressable recall from F_{t-1} (before twocomp ⇒ PV regulates it)
                    _sg = self._ei_sign[l] if (getattr(self, "dale", False) and getattr(self, "_ei_sign", None)) else None
                    _pre_fm = z_eff if _sg is None else z_eff * _sg          # Dale sign at presyn (F itself is unsigned, ∈[0,1])
                    _contrib = Ffast[l] * _pre_fm[:, c.rec_col.long()]       # (B,nnz)
                    recall_now[l] = torch.zeros(B, c.hid, device=dev).index_add_(1, c.rec_row, _contrib) / float(max(1.0, c.rec_fanin))
                    _rho_fm = c.mem_rho_vec if getattr(c, "mem_learn_read", False) else float(c.mem_read_gain)
                    drive = drive + _rho_fm * recall_now[l]                  # drive-read ⇒ +ψ·recall grad for rho_j
                if twocomp:                                    # PV fast divisive gain control + apical→soma feedback
                    apd_fwd = (ap[l] + plat[l]) if plateau else ap[l]   # §17 sustained plateau feeds forward across ticks
                    _ag = getattr(c, "lam_apical_gain", None)           # §17 laminar: apical error → L2/3+L5 tufts, spare L4
                    if _ag is not None: apd_fwd = apd_fwd * _ag
                    _pvg = pv_g * _p_prev                       # §HARM: release the PV divisive gain (a FORWARD suppressor)
                    denom = intern.pv(l, z_prev.mean(1, keepdim=True), _pvg) if use_intern \
                        else (1.0 + _pvg * z_prev.mean(1, keepdim=True))   # when calm/silenced (P→0) → denom→1 → full drive
                    drive = drive / denom + g_ap * apd_fwd
                if diffnm: drive = drive * ne_gain             # NE sets somatic gain (surprise/attention)
                if pc_prev is not None: drive = drive + infer_g * pc_prev[l]   # §PC top-down error relaxes activity
                if intr[l]: drive = drive + c.intrinsic_bias   # §INTRINSIC baseline excitability keeps every neuron alive
                _bt = c.beta_vec if getattr(c, "het_tau", False) else c.beta   # §MEM per-neuron long-memory taus
                v[l] = (_bt * v[l] - c.thr * z_prev if getattr(c, "sub_reset", False)   # §MEM subtractive reset
                        else _bt * v[l] * (1.0 - z_prev)) + drive
                if al[l]:
                    a[l] = c.rho * a[l] + z_prev               # adaptation from the previous spike
                    thr = c.thr0 + c.beta_adapt * a[l]         # adaptive threshold
                else:
                    thr = c.thr
                if homeo: thr = thr + self._thr_adapt[l]       # intrinsic homeostasis offset (metaplasticity)
                if gs[l] and getattr(c, "gs_read", "threshold") != "drive":
                    thr = thr + c.gs_kappa * C[l]              # §MEM2 soma read via threshold (EXACT; membrane v untouched)
                vfire = v[l] + snoise * torch.randn_like(v[l]) if stoch else v[l]   # stochastic (noisy) firing
                psi_l = self._psi(vfire - thr); z[l] = (vfire >= thr).float()
                if fm[l]:                                      # §MEM3 Hebbian EMA write: post(t)·pre(t-1) ⇒ F∈[0,1], no gradient
                    Ffast[l] = c.mem_fast_decay * Ffast[l] + (1.0 - c.mem_fast_decay) * (z[l][:, c.rec_row] * z_eff[:, c.rec_col.long()])
                if homeo: spk_sum[l] = spk_sum[l] + z[l].sum(0)
                if astro_on:                                   # §17 per-neuron accumulators for glia
                    astro_zsum[l] = astro_zsum[l] + z[l].sum(0)               # firing (rate channel)
                    astro_vsum[l] = astro_vsum[l] + v[l].abs().sum(0)         # |v| membrane magnitude (runaway channel)
                eb = ebeta if btsp else c.beta                 # BTSP: eligibility outlives the membrane
                dyn_eb = getattr(self, "_dyn_elig_beta", None)  # §16 P2: attention→FREQUENCY sets the window
                if dyn_eb is not None: eb = float(dyn_eb)       #   (gamma-short when focused, alpha-long when not)
                if diffnm: eb = min(0.995, eb * ht_pat)        # 5-HT stretches the eligibility window (patience)
                if plateau:                                    # §17 BTSP: a live plateau population stretches eligibility → btsp_beta
                    eb = min(0.9999, eb + pcpl * plat_rate * (float(getattr(self, "btsp_beta", 0.98)) - eb))
                if pc: eb = 0.0                                # §PC: NO temporal trace — instantaneous β→0 limit (ε=z_prev)
                eps_rec[l] = eb * eps_rec[l] + z_eff           # §17: eligibility carries the TRANSMITTED spike z⊙g (resource-aware)
                eps_in[l] = eb * eps_in[l] + layer_in
                if do_stdp and sp(c):                          # STDP rides the sparse cortex (edge-wise, O(nnz·B))
                    dr, lp_r, ld_r = sd.edge_delta(z[l], z[l], xr_p[l], xr_m[l], c.rec_row, c.rec_col)
                    stdp_rec[l] += dr; ltp_acc += lp_r; ltd_acc += ld_r
                    if c.sparse_in:                            # post=z[l] (hid), pre=layer_in (in_dim)
                        di, lp_i, ld_i = sd.edge_delta(z[l], layer_in, xi_p[l], xr_m[l], c.in_row, c.in_col)
                        stdp_in[l] += di; ltp_acc += lp_i; ltd_acc += ld_i
                    xr_p[l] = lam_p * xr_p[l] + z[l]           # advance AFTER the delta (t-1 → t)
                    xr_m[l] = lam_m * xr_m[l] + z[l]
                    xi_p[l] = lam_p * xi_p[l] + layer_in
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
                if gs[l]:                                      # §MEM2 ec = g·ec + Φ·z_pre  (LOCAL: clone of ea, decay ρ→g, source ε^v→z)
                    _ip = _ig_l * (1.0 - _ig_l)                                            # i'_j = i(1−i)
                    _one_w2 = 1.0 - _wc_l * _wc_l
                    _Phi = c.k_g * _ip * (_wc_l - _cprev_l) + _ig_l * _one_w2 * (c.write_a if _lw[l] else 1.0)  # Φ_j=∂c_j/∂I_j (·a_j when learn_write)
                    _gret = 1.0 - _ig_l                                                    # retention gate g_j
                    if sp(c):
                        ec_rec[l] = _Phi[:, c.rec_row] * z_eff[:, c.rec_col.long()] + _gret[:, c.rec_row] * ec_rec[l]
                        if c.sparse_in:
                            ec_in[l] = _Phi[:, c.in_row] * layer_in[:, c.in_col.long()] + _gret[:, c.in_row] * ec_in[l]
                        else:
                            ec_in[l] = _Phi.unsqueeze(2) * layer_in.unsqueeze(1) + _gret.unsqueeze(2) * ec_in[l]
                    else:
                        ec_rec[l] = _Phi.unsqueeze(2) * z_eff.unsqueeze(1) + _gret.unsqueeze(2) * ec_rec[l]
                        ec_in[l] = _Phi.unsqueeze(2) * layer_in.unsqueeze(1) + _gret.unsqueeze(2) * ec_in[l]
                    if _lw[l]:                                  # §MEM2c per-neuron write-param traces (diagonal RTRL, O(H))
                        ea_wa[l] = _gret * ea_wa[l] + _ig_l * _one_w2 * _Iraw_l            # ∂C_j/∂a_j
                        ea_wd[l] = _gret * ea_wd[l] + _ig_l * _one_w2                      # ∂C_j/∂d_j
                psi.append(psi_l); layer_in = z[l]
            top_v = v[-1]; logits = self.head(top_v)           # membrane readout → logits
            _rmem = read_mem and gs[-1] and C[-1] is not None  # §MEM2b learned read of the top slow compartment
            if _rmem: logits = logits + F.linear(C[-1], self.mem_read_w)
            p = torch.softmax(logits.float(), 1)
            oh = torch.zeros_like(p); oh.scatter_(1, y[:, tt].long().unsqueeze(1), 1.0)
            err = p - oh                                        # CE gradient wrt logits
            tot_loss += float(-(oh * (p + 1e-9).log()).sum(1).mean())
            if pc: pc_err_out += float(err.abs().mean())        # §PC output prediction error (free-energy proxy)
            gHead += err.t() @ top_v.float(); gHead_b += err.sum(0)   # head grad is LOCAL in time
            if _rmem: gMemRead += err.t() @ C[-1].float()       # §MEM2b R grad = err^T@C (sibling of gHead; NO dL/dC→cortex = no W^T)
            if he_on:                                          # §NLMS: realized membrane energy ‖v_l‖² per layer,
                for l in range(len(cells)):                    #   fp32-accumulated (bf16 length-N sum-of-squares underflows)
                    v_energy[l] += (v[l].float() ** 2).sum()
            if learned_fb:                                     # Kolen-Pollack: B learns the head's grad (top
                for l in range(len(cells) - 1):                # layer) / an error·activity correlation (lower)
                    gFB[l] += err.t() @ v[l].float()
            for l, c in enumerate(cells):
                Lsig = err @ self._fb[l].float()               # top-down learning signal (random or learned B)
                if pc:                                         # §PC local prediction error e_l = err@B_l
                    self.pc.record(l, Lsig)                     #   raw per-layer pred-error (metric)
                    Lsig = self.pc.precision_weight(l, Lsig) * Lsig   # precision-weight Π_l·e_l (mean-norm → width-invariant)
                    if pc_prev is not None: pc_prev[l] = Lsig   #   carry to next-step inference relaxation
                if twocomp:                                    # UNIFIED apical circuit — error runs THROUGH the
                    if use_intern:                             # §17 spiking PV/SOM/VIP pools replace the scalars
                        agate = intern.apical(l, z[l].mean(1, keepdim=True),
                                              (tone.get("ach", gate) if diffnm else gate), som_b)
                    else:
                        vip = float(gate)                      # apical dendrite, not alongside it:
                        som = som_b * z[l].mean(1, keepdim=True)   #  SOM inhibition ∝ population activity, and
                        agate = (vip - som).clamp(min=0.0)     #  VIP (neuromod "learn-now" tone) DISINHIBITS it
                    ap[l] = beta_ap * ap[l] + agate * Lsig     #  → the apical compartment integrates the gated error
                    if plateau:                                #  §17 NMDA plateau: all-or-none, regenerative, latched
                        thr_p = p_thr * (ap[l].abs().mean(1, keepdim=True) + 1e-9)          # RELATIVE (width-invariant)
                        trig = (ap[l].abs() > thr_p).float() * z[l] * (pclk[l] <= 0).float()  # BAC coincidence + refractory
                        pclk[l] = torch.where(trig > 0, torch.full_like(pclk[l], pdur), pclk[l])  # ignite window
                        plat[l] = plat[l] + trig * p_gain * ap[l]                            # supralinear regenerative seed
                        act = (pclk[l] > 0).float(); plat[l] = rho_p * plat[l] * act         # sustain (slow tau_p) then clear
                        pclk[l] = (pclk[l] - 1.0).clamp(min=0.0)
                        apd = ap[l] + plat[l]
                        if l == len(cells) - 1: plat_rate = float(act.mean()); apd_top_mag = float(apd.abs().mean())
                    else:
                        apd = ap[l]
                    thr_b = burst_thr * (apd.abs().mean(1, keepdim=True) + 1e-9)
                    brst = (apd.abs() > thr_b).float() * z[l]  # apical BURST rides a somatic spike (plateau-sustained)
                    Lsig = apd * brst
                    if l == len(cells) - 1: burst_frac = float(brst.mean())
                elif dendritic:                                # standalone apical burst code (Naud/Richards),
                    thr = burst_thr * (Lsig.abs().mean(1, keepdim=True) + 1e-9)   # not yet routed through a
                    brst = (Lsig.abs() > thr).float() * z[l]   # two-compartment neuron — low-bandwidth, noisy
                    Lsig = Lsig * brst
                    if l == len(cells) - 1: burst_frac = float(brst.mean())
                if metab:                                      # metabolic cost (mlam·Σz); §17 glia scales the energy price
                    _mz = z[l] if metab_rel is None else metab_rel[l] * z[l]   # §HARM rate-relative: OVER-target firing only
                    Lsig = Lsig + (mlam * astro_mm * _p_prev) * _mz    # §HARM bus: release the DC price (incl. astro_mm) when calm/silenced
                #   has dLoss/dz=+mlam → ADDS to the per-neuron error, pushing incoming weights DOWN (less firing)
                g = (Lsig * psi[l]).float()                    # g_j = L_j · ψ_j
                if g_kappa[l] is not None:                     # §MEM2b κ_j grad: EXACT per-neuron, no trace, no W^T
                    _krs = 1.0 if getattr(c, "gs_read", "threshold") == "drive" else -1.0
                    g_kappa[l] += _krs * (g * C[l]).sum(0)
                if g_rho[l] is not None:                       # §MEM3 rho_j read-gain grad: ∂z/∂ρ=+ψ·recall (drive-read), instantaneous
                    g_rho[l] += (g * recall_now[l]).sum(0)
                if l == 0 and getattr(c, "Win", None) is not None:   # the byte-embedding learns too: project the
                    gE.index_add_(0, x[:, tt].long(), g @ c.Win.weight)   # layer-0 signal back to the used rows.
                    #   (this input-projection gradient is the ONE weight-transport path in the rule — a
                    #   deliberate, disclosed exception; all cortical credit still flows through _fb, not W^T.)
                ba = c.beta_adapt if al[l] else 0.0            # e_ji = ψ_j(ε^v_i − β_a·ε^a_ji − κ·ε^c_ji); grad = Σ g_j·e_ji
                # §MEM2 compartment sign: threshold-read (thr+=κc ⇒ ∂z/∂c=−κψ) → −κ; drive-read (v+=κc ⇒ +κψ) → +κ.
                _ks = 0.0 if not gs[l] else (c.gs_kappa if getattr(c, "gs_read", "threshold") == "drive" else -c.gs_kappa)
                if _lw[l]:                                     # §MEM2c write-param grads: dL/dC_j=_ks·g_j through the SAME read
                    g_wa[l] += _ks * (g * ea_wa[l]).sum(0)
                    g_wd[l] += _ks * (g * ea_wd[l]).sum(0)
                if sp(c):
                    g_rec[l] += sddmm(g, eps_rec[l], c.rec_row, c.rec_col)     # membrane part, O(nnz)
                    if al[l]: g_rec[l] += -ba * edge_reduce(g, ea_rec[l], c.rec_row)   # adaptation part
                    if gs[l]: g_rec[l] += _ks * edge_reduce(g, ec_rec[l], c.rec_row)   # §MEM2 compartment part, O(nnz)
                    if c.sparse_in:
                        g_in[l] += sddmm(g, eps_in[l], c.in_row, c.in_col)
                        if al[l]: g_in[l] += -ba * edge_reduce(g, ea_in[l], c.in_row)
                        if gs[l]: g_in[l] += _ks * edge_reduce(g, ec_in[l], c.in_row)   # §MEM2
                        g_in_b[l] += g.sum(0)
                    else:
                        g_in[l] += g.t() @ eps_in[l].float()                          # dense in (emb small)
                        if al[l]: g_in[l] += -ba * (g.unsqueeze(2) * ea_in[l]).sum(0)
                        if gs[l]: g_in[l] += _ks * (g.unsqueeze(2) * ec_in[l]).sum(0)   # §MEM2
                        g_in_b[l] += g.sum(0)
                else:
                    g_rec[l] += g.t() @ eps_rec[l].float()
                    if al[l]: g_rec[l] += -ba * (g.unsqueeze(2) * ea_rec[l]).sum(0)    # (h,h) dense adaptation
                    if gs[l]: g_rec[l] += _ks * (g.unsqueeze(2) * ec_rec[l]).sum(0)    # §MEM2 dense compartment
                    g_in[l] += g.t() @ eps_in[l].float()
                    if al[l]: g_in[l] += -ba * (g.unsqueeze(2) * ea_in[l]).sum(0)
                    if gs[l]: g_in[l] += _ks * (g.unsqueeze(2) * ec_in[l]).sum(0)      # §MEM2
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
        # MEMBRANE-MAGNITUDE brake: the over-firing and loss-surprise terms both watch SPIKE statistics, but the
        # everything-on collapse is a REPRESENTATION-magnitude runaway — |v| (the head's input at line 545, and the
        # `mem_mag` diagnostic below) inflates ~10× while the mean firing rate is pinned on the homeostatic target,
        # so `over`≈0 and `surprise`≈0 stay mute until AFTER bpb detonates. mem_mag = mean |v| of the TOP membrane is
        # the leading indicator (climbs before bpb), so fold it into the brake against a FIXED absolute anchor
        # mem_target — NOT a self-following EMA, which would calibrate the runaway away exactly as glia.auto_target
        # does. This pulls plasticity down the moment the representation runs away, throttling the
        # weight→drive→|v|→BTSP-eligibility→weight pump at its growth source. Membrane-magnitude, so it engages even
        # when firing stays on target; live-tunable via attn_mem_sens (0 disables → byte-identical to the old brake).
        mem_mag = float(v[-1].abs().mean())    # top-layer representation magnitude (same as the _diag mem_mag below)
        mem_sens = float(getattr(self, "attn_mem_sens", 0.5))
        over_mem = min(3.0, max(0.0, mem_mag / max(float(getattr(self, "mem_target", 1.0)), 1e-3) - 1.0))
        # ── §HARM EXCITATION-PRESSURE BUS P ── the single absolute-anchored coordinator (see __init__). over_mem is
        # ALREADY relu(mem_mag/mem_target−1) (the membrane channel); add the rate channel relu(rate/target−1) and take
        # the max, ×gain, capped. This is THIS step's fresh P — used by the tail controls (glia pgain, homeostasis,
        # mem-leak) below and cached for life.py's glia gate + the NEXT step's in-loop controls. _pgate∈[0,1] is the
        # tail gate: OFF ⇒ 1.0 (nominal, byte-identical); ON ⇒ min(1,P) so every tail suppressor releases at P=0.
        if _p_on:
            _p_rate = 0.0
            if homeo and spk_sum is not None:
                _p_rate = max(0.0, rate / max(float(getattr(self, "target_rate", 0.08)), 1e-3) - 1.0)
            excite_p = min(float(getattr(self, "excite_p_cap", 3.0)),
                           float(getattr(self, "excite_p_gain", 1.0)) * max(float(over_mem), _p_rate))
        else:
            excite_p = 0.0
        self._excite_p = excite_p              # cache for life.py glia gate + next step's in-loop controls
        _pgate = min(1.0, excite_p) if _p_on else 1.0
        # §HARM: gate the glia per-neuron metaplastic brake by the bus — blend each pgain toward neutral (1.0) by
        # _pgate so the brake RELEASES on a calm/silenced net (P→0 ⇒ pgain→1 ⇒ full plasticity) and reaches its full
        # glia-set value only under pressure. OFF ⇒ _pgate=1 ⇒ pgain unchanged (byte-identical). Feeds BOTH the
        # pre-Adam gradient-scale path and the post-Adam step-scale path (glia_pgain_post_adam) below via astro_pg_g.
        astro_pg_g = astro_pg
        if astro_pg is not None and _p_on:
            astro_pg_g = [1.0 - _pgate * (1.0 - pg) for pg in astro_pg]
        attn_t = min(1.3, max(0.2, 1.0 - sens * surprise - rate_sens * over - mem_sens * over_mem))   # loss-shock OR over-firing OR membrane-runaway → less plasticity
        cur_at = float(getattr(self, "attention", 1.0))
        aw = 0.3 if attn_t < cur_at else 0.1        # ASYMMETRIC: brake FAST (safety-biased), release SLOW — so an
        self.attention = (1.0 - aw) * cur_at + aw * attn_t   # over-firing/loss signal pulls the rate down promptly, not over ~10 steps
        self.loss_ema = 0.98 * self.loss_ema + 0.02 * L             # slow learning-health baseline
        scale = float(gate) * lr * self.attention * ((0.5 + da) if diffnm else 1.0)   # ACh gates; ATTENTION self-adapts; DA reward-modulates
        _adam = getattr(self, "learn_opt", "legacy") == "adam"                # per-weight optimizer on the LOCAL e-prop grad
        # §17 glia metaplasticity survives Adam ONLY as a step scale: reroute pgain from the raw gradient (where
        # Adam's per-weight RMS cancels a pure per-row scale → no-op) onto the applied Adam step. Requires Adam
        # (legacy keeps the ±clamped gradient scale — NOT step-equivalent) and a live glia field (astro_pg).
        _pg_post = bool(getattr(self, "glia_pgain_post_adam", False)) and _adam and (astro_pg is not None)
        if _adam:                                                            # (faithful: reads only each synapse's own g-history)
            self._adam_t = int(getattr(self, "_adam_t", 0)) + 1
            if not hasattr(self, "_adam_state"): self._adam_state = {}
            _b1, _b2, _aeps = 0.9, 0.999, 1e-8; _alr = float(getattr(self, "adam_lr", 1e-3))
            _amult = float(gate) * self.attention * ((0.5 + da) if diffnm else 1.0)   # gate·attn·DA (mechanisms modulate); NO base lr
            _abc1 = 1.0 - _b1 ** self._adam_t; _abc2 = 1.0 - _b2 ** self._adam_t      # bias correction
        def _upd(w, g, fin, pw=p, smul=None):
            if _adam:                                                        # ADAM: per-weight m/û normalization, no ÷fanin, no clamp
                k = id(w); st = self._adam_state.get(k)
                if st is None or st[0].shape != w.shape:
                    st = [torch.zeros_like(w, dtype=torch.float32), torch.zeros_like(w, dtype=torch.float32)]; self._adam_state[k] = st
                m, u = st; gf = g.float()
                m.mul_(_b1).add_(gf, alpha=1.0 - _b1); u.mul_(_b2).addcmul_(gf, gf, value=1.0 - _b2)
                step = (_amult * _alr) * (m / _abc1) / ((u / _abc2).sqrt() + _aeps)
                if smul is not None: step = step * smul   # §17 glia: per-post-neuron LR multiplier on the ADAM STEP (survives RMS)
                w.add_(step.to(w.dtype), alpha=-1.0)
                return
            d = (fin ** pw) if torch.is_tensor(fin) else float(fin) ** pw     # §17 per-neuron eff-fanin (tensor) or scalar
            w.add_((scale * (g / (denom * d))).clamp_(-dmax, dmax).to(w.dtype), alpha=-1.0)
        hpow = float(getattr(self, "head_fanin_pow", 0.5))     # gentler fan-in norm for the wide readout head
        if do_stdp:                                            # §15.18 blend the timing term into the grad BEFORE _upd:
            _lastsp = None; _mag_sum = 0.0; _mag_n = 0          #   −Δw because _upd subtracts (so +Δw potentiates);
            for l, c in enumerate(cells):                      #   masked so silent/pruned synapses stay silent
                if not sp(c): continue
                g_rec[l] = g_rec[l] - sd.mix * stdp_rec[l] * c.rec_mask
                if c.sparse_in:
                    g_in[l] = g_in[l] - sd.mix * stdp_in[l] * c.in_mask
                _t = (sd.mix * stdp_rec[l]).abs()              # blended timing magnitude on THIS layer's edges
                _mag_sum += float(_t.sum()); _mag_n += int((_t > 0).sum())   # active (paired) edges only
                _lastsp = l
            sd._ltp = ltp_acc; sd._ltd = ltd_acc
            # mean over ACTIVE edges across ALL sparse layers (the old code sampled only the top layer, which is
            # the sparsest-firing one — near-silent at depth — so stdp_mag read a flat 0.0 while STDP was healthy).
            sd._mag = (_mag_sum / _mag_n) if _mag_n > 0 else 0.0
        if astro_pg is not None and not _pg_post:              # §17 glia: slow per-POSTsynaptic-neuron metaplastic gain on
            for l, c in enumerate(cells):                      #   the APPLIED update (a 4th factor on Δw, NOT the eligibility).
                pg = astro_pg_g[l].to(g_rec[l].dtype)          #   §HARM bus-gated; _pg_post ⇒ moved onto the Adam STEP (below)
                if sp(c):
                    g_rec[l] = g_rec[l] * pg[c.rec_row]        # sparse: scale each edge by its POSTsynaptic neuron's gain
                    if c.sparse_in:
                        g_in[l] = g_in[l] * pg[c.in_row]; g_in_b[l] = g_in_b[l] * pg
                    else:
                        g_in[l] = g_in[l] * pg.unsqueeze(1); g_in_b[l] = g_in_b[l] * pg
                else:
                    g_rec[l] = g_rec[l] * pg.unsqueeze(1)      # dense: rows = postsynaptic neurons
                    g_in[l] = g_in[l] * pg.unsqueeze(1); g_in_b[l] = g_in_b[l] * pg
        for l, c in enumerate(cells):
            # §17 glia (post-Adam): the per-post-neuron gain rides the STEP, so slice it to each weight's rows —
            # per-EDGE by postsynaptic neuron for sparse (pg[row]), per-ROW for dense (pg[:,None]), per-neuron for
            # biases. None ⇒ no scale (default path / non-glia weights: head, E, feedback are never glia-throttled).
            _pgl = astro_pg_g[l].to(g_rec[l].dtype) if _pg_post else None   # §HARM bus-gated per-post-neuron step scale
            if sp(c):
                _sm_e = _pgl[c.rec_row] if _pgl is not None else None    # per recurrent EDGE, indexed by its post-neuron
                _lrm = getattr(c, "lam_rec_mask", None)
                if _lrm is not None:                                     # §17 laminar-thinned fan-in norm (width-invariant)
                    _upd(c.rec_val, g_rec[l] * (c.rec_mask * _lrm), c.lam_rec_fanin[c.rec_row].to(g_rec[l].dtype), smul=_sm_e)
                else:
                    _upd(c.rec_val, g_rec[l] * c.rec_mask, c.rec_fanin, smul=_sm_e)   # silent synapses get no update
                if c.sparse_in:
                    _upd(c.in_val, g_in[l] * c.eff_in_mask(), c.in_fanin, smul=(_pgl[c.in_row] if _pgl is not None else None))
                    _upd(c.in_bias, g_in_b[l], 1, smul=_pgl)
                else:
                    _upd(c.Win.weight, g_in[l], c.Win.weight.shape[1], smul=(_pgl.unsqueeze(1) if _pgl is not None else None))
                    _upd(c.Win.bias, g_in_b[l], 1, smul=_pgl)
            else:
                _sm_r = _pgl.unsqueeze(1) if _pgl is not None else None  # dense: rows = postsynaptic neurons
                _lrw = getattr(c, "lam_rec_w", None)                     # §17 laminar dense adjacency (test nets)
                _upd(c.Wrec.weight, g_rec[l] if _lrw is None else g_rec[l] * _lrw, c.Wrec.weight.shape[1], smul=_sm_r)
                _upd(c.Win.weight, g_in[l], c.Win.weight.shape[1], smul=_sm_r)
                _upd(c.Win.bias, g_in_b[l], 1, smul=_pgl)
        mu = meanE = epsE = None
        if he_on and not _adam:                                # §NLMS energy-normalized readout (width+sparsity-invariant)
            last = len(cells) - 1
            # μ drops the cortex's 2000× base (self.lr·eprop_lr_scale) — NLMS μ is O(0.3) — but KEEPS gate·attention·DA
            # so the over-excitation brake / ACh gate / reward still modulate the head.
            mu = float(gate) * self.attention * ((0.5 + da) if diffnm else 1.0) * float(getattr(self, "head_lr_scale", 0.3))
            meanE = float(v_energy[last]) / denom              # realized top-layer membrane energy = ‖v‖² (per-sample mean)
            prevE = float(getattr(self, "_head_e_ema", 0.0) or 0.0)
            self._head_e_ema = meanE if prevE <= 0.0 else 0.98 * prevE + 0.02 * meanE
            epsE = float(getattr(self, "head_energy_eps", 1e-2)) * max(self._head_e_ema, 1e-12)   # silent-layer floor
            def _updh(w, g, mE):                               # ΔW = μ·g/(‖v‖²+ε) → Δlogit = μ·err (width/ρ-free)
                w.add_((mu * (g / (denom * (mE + epsE)))).clamp_(-dmax, dmax).to(w.dtype), alpha=-1.0)
            _updh(self.head.weight, gHead, meanE)
            self.head.bias.add_((mu * (gHead_b / denom)).clamp_(-dmax, dmax).to(self.head.bias.dtype), alpha=-1.0)
        else:
            _upd(self.head.weight, gHead, self.head.weight.shape[1], pw=hpow)   # head: gentler norm so it isn't starved at width
            _upd(self.head.bias, gHead_b, 1)
        if gMemRead is not None:                               # §MEM2b R: learned read of the slow bank (head power path)
            _upd(self.mem_read_w, gMemRead, self.mem_read_w.shape[1], pw=hpow)
        for l, c in enumerate(cells):                          # §MEM2b κ_j: per-neuron read-gain update + MANDATORY |κ|<thr bound
            if g_kappa[l] is not None:
                _upd(c.kappa_vec, g_kappa[l], 1)
                c.kappa_vec.data.clamp_(-(float(getattr(c, "thr", 1.0)) - 1e-3), float(getattr(c, "thr", 1.0)) - 1e-3)
            if g_wa[l] is not None:                            # §MEM2c learned write update (a slope, d set-point) + hygiene clamp
                _upd(c.write_a, g_wa[l], 1); _upd(c.write_d, g_wd[l], 1)
                c.write_a.data.clamp_(-8.0, 8.0)               # |c|≤1 holds for any a,d; clamp is numerical hygiene only
            if g_rho[l] is not None:                           # §MEM3 rho_j read-gain update, bounded [0, rho_max]
                _upd(c.mem_rho_vec, g_rho[l], 1)
                c.mem_rho_vec.data.clamp_(0.0, float(getattr(c, "mem_rho_max", 2.0)))
        _upd(self.E.weight, gE, self.E.weight.shape[1])        # sensory byte-embedding update (no longer frozen) — both paths
        if learned_fb:                                         # Kolen-Pollack: mirror the head/error grad into
            fb_dec = float(getattr(self, "fb_decay", 1e-4)); last = len(cells) - 1   # B, then weight-decay → B aligns with W
            for l in range(len(cells)):                         # B must move at the SAME rate as the head, else it can't align
                G = gHead if l == last else gFB[l]
                if he_on and not _adam: _updh(self._fb[l], G, float(v_energy[l]) / denom)   # B_l gets v[l]'s OWN energy → aligns
                else:                   _upd(self._fb[l], G, self._fb[l].shape[1], pw=hpow)
                self._fb[l].mul_(1.0 - fb_dec)
        if bounded:                                            # Fusi bounded synapses: clamp the MEMBRANE-FEEDING
            for c in cells:                                    # weights so the recurrent map can't pump |v| unbounded.
                if sp(c):                                      # Bound is fan-in-RELATIVE (÷√fanin) → width-invariant,
                    rb = self._syn_bound(c.rec_fanin); c.rec_val.data.clamp_(-rb, rb)          # near the stable init.
                    if c.sparse_in:
                        ib = self._syn_bound(c.in_fanin); c.in_val.data.clamp_(-ib, ib)
                else:
                    rb = self._syn_bound(c.Wrec.weight.shape[1]); c.Wrec.weight.data.clamp_(-rb, rb)   # fanin = hidden
                    ib = self._syn_bound(c.Win.weight.shape[1]); c.Win.weight.data.clamp_(-ib, ib)
            if twocomp and getattr(self, "_fb", None) is not None:   # APICAL COUPLING: bound the learned error-
                for B_l in self._fb:                                 # projection B that drives ap[l] — under two_
                    fb_b = self._syn_bound(B_l.shape[0]); B_l.clamp_(-fb_b, fb_b)   # compartment Kolen-Pollack grows
                    #   B unbounded (only 1e-4 decay) → apical-loop gain g_ap·apd runaway; B is (V,hid), init 1/√V.
        if homeo:                                              # intrinsic homeostasis: RATE→threshold(+) + MAGNITUDE→threshold(−)
            _mdrift = float(getattr(self, "homeo_lr_drift", 0.0))
            _mtgt = max(float(getattr(self, "mem_homeo_target", 3.0)), 1e-3)
            _mlr = float(getattr(self, "mem_homeo_lr", 0.05))
            _hleak = float(getattr(self, "homeo_leak", 0.0))                        # §HARM P-gated anti-windup leak
            for l in range(len(cells)):
                rerr = spk_sum[l] / float(B * T) - float(self.target_rate)          # per-neuron rate error
                elr = self.homeo_lr                                                 # RATE channel → threshold (existing)
                if _mdrift > 0.0:                                                   # drift-adaptive gain: raise thr FASTER when the
                    elr = self.homeo_lr * (1.0 + _mdrift * float(rerr.abs().mean()) / max(float(self.target_rate), 1e-3))
                self._thr_adapt[l] += elr * rerr                                    #   rate is far from target (catch a fast runaway)
                if _p_on and _hleak > 0.0:                                          # §HARM: leak the integrator toward baseline
                    self._thr_adapt[l] *= (1.0 - _hleak * (1.0 - _pgate))           #   ∝(1−P) — UNWIND the wind-up when calm/silenced
                    #   (P→0 ⇒ full leak ⇒ threshold returns ⇒ neurons re-fire ⇒ recovery); HOLD (no leak) under over-rate pressure.
                if mem_homeo:
                    # MAGNITUDE channel: the everything-on collapse is a REPRESENTATION-WIDE |v| runaway (the LAYER-
                    # MEAN mem_mag inflates) that the rate channel is blind to (|v| grows with rate on target); worse,
                    # raising thr to hold rate INFLATES |v|. Key on the LAYER MEAN (= the mem_mag runaway metric), NOT
                    # per-neuron: a healthy sparse code has a high-|v| TAIL but a low mean, so per-neuron triggering
                    # would brake healthy neurons — the layer mean fires only when the whole representation runs away.
                    # When it does, LOWER the threshold (uniformly across the layer) so over-driven neurons fire and
                    # RESET, discharging v through its OWN reset (the membrane's natural clamp) — the correct-sign
                    # actuator that needs NO change to the leak β / integration τ (a substrate change de-calibrates
                    # the readout, measured to break learning). The RATE channel is the restoring force (more firing →
                    # it raises thr back), so the two settle at a slightly-above-target rate that pins the mean
                    # |v|≈mem_homeo_target. relu ⇒ 0 while mean|v|≤target ⇒ byte-identical when quiescent; the default
                    # target 3.0 sits ABOVE the healthy high-|v| operating band so it fires only on a genuine runaway.
                    excess = max(0.0, float(v[l].abs().mean()) / _mtgt - 1.0)
                    if excess > 0.0: self._thr_adapt[l] -= _mlr * excess * _pgate   # §HARM: mem-leak engages ∝P, releases at P=0
                _hclamp = float(getattr(self, "homeo_clamp", 0.0))                  # §HARM hard anti-windup backstop (both channels)
                if _hclamp > 0.0: self._thr_adapt[l].clamp_(-_hclamp, _hclamp)      #   keeps thr out of the ψ dead-zone (unconditional)
        if getattr(self, "dale", False):
            self._project_dale()                               # re-impose E/I sign law after the update
        self._burst_frac = burst_frac
        if astro_on:                                           # §17 per-neuron rate r_l=(Σ_{b,t}z)/(B·T) for glia.sense()
            self._spk_rate_vec = [astro_zsum[l] / float(B * T) for l in range(len(cells))]
            for l, c in enumerate(cells):                      # §INTRINSIC adapt each neuron's excitability toward the floor
                if intr[l]: _adapt_intrinsic(c, self._spk_rate_vec[l])
            self._mem_mag_vec = [astro_vsum[l] / float(B * T) for l in range(len(cells))]   # per-neuron |v| for glia.sense_mem()
        if bool(getattr(self, "metab_rate_relative", False)) and getattr(self, "_astro_a", None) is None:
            # §HARM fallback per-neuron rate EMA for the rate-relative metabolic gate when glia is OFF (glia's a_l
            # is preferred and, when present, drives the gate directly). Source: this step's per-neuron rate from
            # astro (if on) or homeostasis' spk_sum. Neither ⇒ no source ⇒ gate stays released (penalty = 0).
            _rv = getattr(self, "_spk_rate_vec", None) if astro_on \
                else ([spk_sum[l] / float(B * T) for l in range(len(cells))] if homeo else None)
            if _rv is not None:
                _re = getattr(self, "_metab_rate_ema", None)
                if _re is None or len(_re) != len(_rv) or any(_re[l].numel() != _rv[l].numel() for l in range(len(_rv))):
                    self._metab_rate_ema = [r.detach().clone() for r in _rv]   # (re)seed on first use / width change
                else:
                    for l in range(len(_rv)): _re[l].mul_(0.9).add_(_rv[l].detach(), alpha=0.1)
        if twocomp: self._apical_mag = float(ap[-1].abs().mean())   # apical-dendrite drive magnitude
        if use_intern: self._intern_rates = intern.state()          # §17 cache per-type spiking rates for weight_stats
        if plateau:                                                 # §17 write back plateau metrics (apical_mag on apd)
            self._plateau_rate = plat_rate; pl._last_rate = plat_rate
            self._apical_mag = apd_top_mag
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
            head_update_mag=(float((mu * (gHead / (denom * (meanE + epsE)))).clamp(-dmax, dmax).abs().mean()) if (he_on and not _adam and meanE is not None)
                             else float((scale * (gHead / (denom * float(self.head.weight.shape[1]) ** hpow))).clamp(-dmax, dmax).abs().mean())),
            surprise=float(surprise),
            rec_w_mag=float(_ct.rec_val.abs().mean() if sp(_ct) else _ct.Wrec.weight.abs().mean()),
        )
        if any(gs):                                            # §MEM2 stability guard: max|c| MUST stay ≤1 (convex bound); mean|c| = register occupancy
            _cvals = [C[l] for l in range(len(cells)) if gs[l] and C[l] is not None]
            self._diag["slow_cmax"] = float(max(t.abs().max() for t in _cvals)) if _cvals else 0.0
            self._diag["slow_cmag"] = float(sum(t.abs().mean() for t in _cvals) / max(1, len(_cvals))) if _cvals else 0.0
        if mem_homeo:                                          # §HARM magnitude-channel engagement: mean thr offset (≤0 = discharging)
            self._diag["mem_thr"] = float(self._thr_adapt[-1].mean())
        if he_on and mu is not None:                           # §NLMS truthful head diagnostics (mu/meanE are None under adam)
            self._diag["head_dlogit"] = float(mu * err.abs().mean())   # BY DESIGN (wrong health signal) — watch head_dlogit
            self._diag["head_energy"] = float(meanE)                   # (~μ·err, width-free) + bpb + fb_align_cos instead
        if pc:                                                 # §PC: surface prediction-error + precision metrics
            self._diag["pred_err_out"] = pc_err_out / max(T, 1)
            self.pc._pred_err_out = pc_err_out / max(T, 1)
            self._pc_err = list(self.pc._err)
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
        if getattr(self.cells[-1], "lamina", None) is not None:            # §17 per-lamina rate + masked-edge fraction
            tot = 0.0; masked = 0.0
            for c in self.cells:
                lm = getattr(c, "lam_rec_mask", None)
                if lm is not None and hasattr(c, "rec_mask"):
                    tot += float(c.rec_mask.sum()); masked += float((c.rec_mask & ~lm).sum())
            out["lam_forbidden_frac"] = (masked / tot) if tot > 0 else 0.0
            for k, val in (getattr(self, "_lam_rates", None) or {}).items():
                out[k] = float(val)                                       # rate_L4/L23/L56 (filled live by laminar.measure)
        out["attention"] = float(getattr(self, "attention", 1.0))              # self-adapting plasticity gate
        out["eff_lr_scale"] = float(getattr(self, "eprop_lr_scale", 2000.0) * getattr(self, "attention", 1.0))
        if getattr(self, "learn_rule", "eprop") == "pc" and hasattr(self, "pc"):   # §PC per-layer prediction error
            out["pred_err_out"] = float((getattr(self, "_diag", {}) or {}).get("pred_err_out", 0.0))
            for _l, _e in enumerate(getattr(self, "_pc_err", [])):
                out[f"pred_err_L{_l}"] = float(_e)
            out["mean_precision"] = float(self.pc.state()["mean_precision"])
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
            if getattr(self, "_plateau", None) is not None and self._plateau.on:
                out["plateau_rate"] = float(getattr(self, "_plateau_rate", 0.0))   # §17 NMDA plateau active fraction
            if getattr(self, "interneurons", None) is not None and self.interneurons.on:
                r = getattr(self, "_intern_rates", {})                        # §17 spiking PV/SOM/VIP population rates
                out["intern_pv"] = r.get("rate_pv", 0.0); out["intern_som"] = r.get("rate_som", 0.0)
                out["intern_vip"] = r.get("rate_vip", 0.0)
        if getattr(self, "homeostasis", False) and getattr(self, "_thr_adapt", None):
            out["homeo_thr_mean"] = float(torch.cat(self._thr_adapt).mean())   # homeostatic threshold drift
        if getattr(self, "bounded_synapses", False):                          # fraction of synapses saturated
            c0 = self.cells[0]                                                 # (vs the EFFECTIVE fan-in-relative bound)
            if self._is_sparse(c0): w0 = c0.rec_val.detach().abs(); bnd = self._syn_bound(c0.rec_fanin)
            else:                   w0 = c0.Wrec.weight.detach().abs(); bnd = self._syn_bound(c0.Wrec.weight.shape[1])
            out["synapse_sat_frac"] = float((w0 >= 0.999 * bnd).float().mean())
        if getattr(self, "stp", None) is not None and self.stp.on:
            out["stp_efficacy"] = float(self.stp.mean_efficacy)   # §17 headline metric (≈1 rest, <1 depress, >1 facil)
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
                   "attn_mem_sens", "mem_target",
                   "feedback_mode", "fb_decay", "dale", "dendritic", "burst_thr", "bounded_synapses", "w_max", "w_max_relative",
                   "homeostasis", "target_rate", "homeo_lr", "mem_homeostasis", "mem_homeo_target", "mem_homeo_lr", "homeo_lr_drift",
                   "btsp", "btsp_beta", "two_compartment", "g_ap",
                   "beta_ap", "som_baseline", "pv_gain", "diff_neuromod", "stochastic", "spike_noise", "metabolic",
                   "metabolic_lambda", "metab_rate_relative", "metab_rate_cap", "stdp", "head_norm", "head_lr_scale", "head_energy_eps",
                   "learn_opt", "adam_lr", "glia_pgain_post_adam",
                   "excite_pressure", "excite_p_cap", "excite_p_gain", "homeo_leak", "homeo_clamp")   # §HARM excitation-pressure bus

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
                if v not in ("eprop", "bptt", "pc"): continue
            elif k == "feedback_mode":
                if v not in ("learned", "random"): continue
            elif k == "head_norm":                         # invalid values must NOT silently change the rule
                if v not in ("power", "energy"): continue
            elif k == "learn_opt":                         # per-weight optimizer on the local e-prop gradient
                if v not in ("legacy", "adam"): continue
            elif k == "stdp":                              # single source of truth = self.stdp.on
                self.stdp.set_params(on=v); applied[k] = self.stdp.on; continue
            else:
                cur = getattr(self, k, None)
                if isinstance(cur, bool):    v = bool(v)
                elif isinstance(cur, float): v = float(v)
                if k == "eprop_lr_scale":                  # clamp numeric hyperparams to sane ranges
                    v = max(0.1, v)
                elif k in ("w_max", "mem_target", "mem_homeo_target"):   # strictly positive (these divide in a brake)
                    v = max(1e-3, v)
                elif k in ("target_rate", "homeo_lr", "pv_gain", "g_ap", "fb_decay", "burst_thr", "head_fanin_pow",
                           "som_baseline", "spike_noise", "metabolic_lambda", "attn_sensitivity", "attn_rate_sens",
                           "attn_mem_sens", "mem_homeo_lr", "homeo_lr_drift", "metab_rate_cap",
                           "head_lr_scale", "head_energy_eps", "adam_lr",
                           "excite_p_cap", "excite_p_gain", "homeo_leak", "homeo_clamp"):   # §HARM bus knobs (≥0)
                    v = max(0.0, v)
                elif k in ("btsp_beta", "beta_ap"):
                    v = min(0.9999, max(0.0, v))
            setattr(self, k, v); applied[k] = getattr(self, k)
        if self.dale:        self._project_dale()          # make weights Dale-compliant immediately
        if self.homeostasis: self._ensure_homeo()
        return applied

    @torch.no_grad()
    def set_mem(self, gated_slow=None, gs_read=None, k_g=None, gs_kappa=None, learn_read=None, read_mem=None, learn_write=None, cells="all"):
        """§MEM2/§MEM2b live control of the input-gated slow compartment + its learned read, broadcast to every cell.
        gated_slow toggles the working-memory register; gs_read∈{'threshold'(exact),'drive'}; k_g the gate slope;
        gs_kappa the fixed soma read gain (κ<0 = excitatory afterdepolarization, |κ|<thr). §MEM2b: learn_read makes
        the per-neuron read gain κ_j learnable; read_mem gives the head a DIRECT learned linear read of the top slow
        compartment (logits += C@mem_read_w.t()). Toggling on is safe live (c starts at 0). Returns the applied config."""
        tgt = self.cells if cells == "all" else [self.cells[i] for i in cells]
        for c in tgt:
            if gated_slow is not None: c.gated_slow = bool(gated_slow)
            if gs_read in ("threshold", "drive"): c.gs_read = gs_read
            if k_g is not None: c.k_g = float(k_g)
            if gs_kappa is not None: c.gs_kappa = float(gs_kappa)
            if learn_read is not None: c.learn_read = bool(learn_read)
            if learn_write is not None: c.learn_write = bool(learn_write)
        if read_mem is not None: self.read_mem = bool(read_mem)
        return {"gated_slow": getattr(tgt[0], "gated_slow", False), "gs_read": getattr(tgt[0], "gs_read", "threshold"),
                "k_g": getattr(tgt[0], "k_g", 1.0), "gs_kappa": getattr(tgt[0], "gs_kappa", -0.3),
                "learn_read": getattr(tgt[0], "learn_read", False), "read_mem": getattr(self, "read_mem", False),
                "learn_write": getattr(tgt[0], "learn_write", False)} if tgt else {}

    @torch.no_grad()
    def set_fastmem(self, fast_mem=None, mem_fast_decay=None, mem_read_gain=None, mem_learn_read=None, mem_rho_max=None, cells="all"):
        """§MEM3 live control of the fast-Hebbian relational store (the variable-binding mechanism), broadcast per cell.
        fast_mem toggles the store; mem_fast_decay=λ sets the working-memory horizon ~1/(1-λ); mem_read_gain the fixed
        recall gain (else per-neuron mem_rho_vec when mem_learn_read). Sparse cortex only. Default off ⇒ byte-identical."""
        tgt = self.cells if cells == "all" else [self.cells[i] for i in cells]
        for c in tgt:
            if not hasattr(c, "rec_val"): continue             # sparse only (like STDP)
            if fast_mem is not None: c.fast_mem = bool(fast_mem)
            if mem_fast_decay is not None: c.mem_fast_decay = min(0.999, max(0.0, float(mem_fast_decay)))
            if mem_read_gain is not None: c.mem_read_gain = float(mem_read_gain)
            if mem_learn_read is not None: c.mem_learn_read = bool(mem_learn_read)
            if mem_rho_max is not None: c.mem_rho_max = float(mem_rho_max)
        return {"fast_mem": getattr(tgt[0], "fast_mem", False), "mem_fast_decay": getattr(tgt[0], "mem_fast_decay", 0.92),
                "mem_read_gain": getattr(tgt[0], "mem_read_gain", 0.5), "mem_learn_read": getattr(tgt[0], "mem_learn_read", False)} if tgt else {}

    def faith_config(self):
        """The current state of every faithfulness toggle/hyperparameter — the fidelity axis settings."""
        return {k: (self.stdp.on if k == "stdp" else getattr(self, k, None)) for k in self._FAITH_KEYS}

    def _syn_bound(self, fanin):
        """Fusi bounded-synapse clamp bound for a weight tensor with the given post-neuron fan-in.
        Fan-in-RELATIVE by default: w_max × (1/√fanin) — the clamp then scales like the init (1/√fanin),
        so the SUMMED recurrent drive Σ_i W_ji·z_i (hence the membrane fixed point v*≈drive/(1−β)) and the
        recurrent operator's spectral radius stay O(1) at ANY width, held near the stable init regime rather
        than the ~32×-init headroom a fixed ±1 permits. Absolute ±w_max when w_max_relative=False."""
        wm = float(getattr(self, "w_max", 3.0))
        if bool(getattr(self, "w_max_relative", True)):
            return wm / max(1.0, float(fanin)) ** 0.5
        return wm

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
            nmr = torch.zeros(self.V, new, device=dev, dtype=self.mem_read_w.dtype)   # §MEM2b widen the learned read (new cols 0 = id-preserving)
            nmr[:, :old] = self.mem_read_w
        self.head = nhead
        self.mem_read_w = nn.Parameter(nmr)
        self.hidden = new
        self._head_e_ema = 0.0                                  # §NLMS: recalibrate the ε energy-floor to the new width
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
                        pc=(dict(cfg={k: getattr(self.pc, k) for k in self.pc._KEYS},
                                 sig2=[t.detach().cpu() for t in self.pc._sig2]) if hasattr(self, "pc") else None),  # §PC state
                        thr_adapt=getattr(self, "_thr_adapt", None),
                        stp=({k: getattr(self.stp, k) for k in self.stp._KEYS} if getattr(self, "stp", None) is not None else None)), path)  # §17 STP params (transient u,x NOT saved)

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
        self.mem_read_w = nn.Parameter(torch.zeros(self.V, self.hidden, device=self.device))   # §MEM2b rebuilt at saved width before load
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
            if k == "stdp":                                              # object-backed toggle: keep the SpikingSTDP,
                if hasattr(self, "stdp"): self.stdp.set_params(on=v)     # route to its controller (don't overwrite it)
                continue
            setattr(self, k, v)
        self.attention = d.get("attention", 1.0); self.loss_ema = d.get("loss_ema", None)
        if d.get("stp") and getattr(self, "stp", None) is not None:
            self.stp.set_params(**d["stp"])                    # §17 restore STP params (u,x rebuilt at rest)
        if d.get("fb") is not None:   self._fb = [t.to(self.device) for t in d["fb"]]        # aligned feedback
        if d.get("ei") is not None:   self._ei_sign = [t.to(self.device) for t in d["ei"]]   # Dale E/I typing
        if d.get("thr_adapt") is not None: self._thr_adapt = [t.to(self.device) for t in d["thr_adapt"]]
        if d.get("pc") is not None and hasattr(self, "pc"):                # §PC: restore cfg + per-neuron variance
            self.pc.set_params(**(d["pc"].get("cfg") or {}))
            self.pc._sig2 = [t.to(self.device).float() for t in (d["pc"].get("sig2") or [])]
            _pr = [1.0 / (s + float(self.pc.eps)) for s in self.pc._sig2]
            self.pc._prec = [(p / (p.mean() + 1e-12)) for p in _pr]
            self.pc._err = [0.0 for _ in self.pc._sig2]
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
