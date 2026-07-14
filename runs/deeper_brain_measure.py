"""§16 measurement (the obligation of breadth): turn each deeper-brain mechanism ON and measure whether it
HELPS, HURTS, or does NOTHING to bits/byte — the same discipline applied to the §15.17 faithfulness stack.
Honest scope: this measures the CORTEX-LEVEL effect each mechanism has on learning (the part that touches
bits/byte). P0's memory benefit is a separate forgetting-resistance measurement (runs/generative_replay_test.py);
P1/P2 also have LIFE-loop behavioural effects (drive/stress regulation, sleep-pressure, selective ignition over
a lifetime) that only a living run measures — noted, not claimed here.

Run: HIP_VISIBLE_DEVICES=1 python runs/deeper_brain_measure.py
"""
import os, sys
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
sys.path.insert(0, "/home/dander/workspace/zk/sapience")
import torch
from brain.spiking_brain import SpikingBrain
from brain.endocrine import SpikingEndocrine
from brain.dynamics import SpikingDynamics

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda": torch.zeros(1, device=DEV)
HID, STEPS = 8000, 220
TXT = (" the quick brown fox jumps over the lazy dog. water runs to the sea. the sun rose over the hills. "
       "bread is made from wheat. trees grow tall. the ocean is deep and blue. fire gives light. ") * 40

def fresh():
    torch.manual_seed(0)
    b = SpikingBrain(DEV, emb=64, hidden=HID, layers=2, cell="lif", seed=0,
                     sparse=True, rec_fanin=32, in_fanin=32, syn_density=0.5)
    b.set_faith(learn_rule="eprop", feedback_mode="learned", two_compartment=True); b.eprop_lr_scale = 2000
    return b

def run(mode):
    b = fresh()
    endo = SpikingEndocrine(); endo.on = (mode in ("endocrine", "both"))
    dyn = SpikingDynamics(); dyn.on = (mode in ("dynamics", "both"))
    bpb0 = b.bits_per_byte(TXT); prev = bpb0
    for i in range(STEPS):
        gate = endo.plasticity_gain() if endo.on else 1.0        # §16 P1: cortisol inverted-U gates plasticity
        if dyn.on:
            b._dyn_elig_beta = dyn.eligibility_beta(float(getattr(b, "attention", 1.0)))  # P2: attention→window
        else:
            b._dyn_elig_beta = None
        r = b.learn_eprop(TXT, epochs=1, bs=16, max_steps=1, seq=48, gate=gate)
        if endo.on:                                              # drive the endocrine from the live learning signal
            dg = getattr(b, "_diag", {}) or {}
            prog = max(0.0, (r[0] - r[1])) if isinstance(r, tuple) else 0.0
            endo.wake_tick(surprise=dg.get("surprise", 0.0),
                           threat=max(0.0, 1.0 - float(getattr(b, "attention", 1.0))), progress=prog, novelty=0.5)
            if (i + 1) % 40 == 0:                                # a "night" every 40 steps: sleep relieves cortisol
                for _ in range(15): endo.sleep_tick()            #   (the wake/sleep cycle the real loop provides)
    return round(b.bits_per_byte(TXT), 3), endo, dyn

print(f"§16 mechanism A/B — cortex-level bits/byte effect ({HID*2//1000}k neurons, {STEPS} steps, identical seed/data)\n")
base = None; rows = []
for mode in ("baseline", "endocrine", "dynamics", "both"):
    bpb, endo, dyn = run(mode)
    if mode == "baseline": base = bpb
    delta = round(bpb - base, 3)
    extra = f"cortisol {endo.C:.2f} g(C) {endo.plasticity_gain():.2f}" if endo.on else ""
    if dyn.on: extra += f"  eff_freq {dyn.eligibility_beta(1.0):.3f}"
    rows.append((mode, bpb, delta, extra))
    print(f"  {mode:10s} bpb {bpb:.3f}   Δ vs baseline {delta:+.3f}   {extra}", flush=True)

print("\nverdict (helps if Δ<-0.05, hurts if Δ>+0.05, else NEUTRAL on bits/byte):")
for mode, bpb, delta, _ in rows[1:]:
    v = "HELPS" if delta < -0.05 else ("HURTS" if delta > 0.05 else "NEUTRAL")
    print(f"  {mode}: {v} ({delta:+.3f})")
print("\nP0 generative replay: measured separately by forgetting-resistance (runs/generative_replay_test.py): "
      "retention generative 0.274 > raw-buffer 0.255 > none 0.232.")
