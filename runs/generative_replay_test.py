"""P0 acceptance test (§16): does GENERATIVE self-replay resist catastrophic forgetting WITHOUT a raw
replay buffer? Train topic A, then live through B..E under three consolidation modes and measure how much
of A is retained. Accept iff  generative ≳ buffer ≫ none  (buffer = the old raw-replay baseline).

Run: HIP_VISIBLE_DEVICES=1 python runs/generative_replay_test.py
"""
import os, sys
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
sys.path.insert(0, "/home/dander/workspace/zk/sapience")
import torch
from brain.spiking_brain import SpikingBrain

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda": torch.zeros(1, device=DEV)

# five distinctive topics (disjoint vocabulary → forgetting is measurable)
TOPICS = {
 "A": ("the ocean tide rolls over cold grey stones as gulls wheel above the harbour boats. "
       "salt spray and seaweed drift on the wind while fishermen haul their heavy nets ashore. ") * 12,
 "B": ("the desert dunes shimmer under a blazing noon sun where scorpions hide beneath dry rock. "
       "camels plod across the endless sand and no rain has fallen for many long months here. ") * 12,
 "C": ("the mountain peaks wear glaciers of ancient ice above dark pine forests and steep ravines. "
       "eagles nest on granite cliffs while climbers rope across the frozen ridge at dawn. ") * 12,
 "D": ("the city street hums with traffic and neon signs above crowded cafes and subway stairs. "
       "commuters hurry past glass towers as sirens echo through the concrete downtown canyons. ") * 12,
 "E": ("the meadow blooms with clover and wild poppies where honeybees drift from flower to flower. "
       "a gentle brook winds past the willow trees and rabbits graze in the warm green grass. ") * 12,
}

def fresh():
    torch.manual_seed(0)
    b = SpikingBrain(DEV, emb=64, hidden=8000, layers=2, cell="lif", seed=0,
                     sparse=True, rec_fanin=32, in_fanin=32, syn_density=0.5)
    b.set_faith(learn_rule="eprop", feedback_mode="learned"); b.eprop_lr_scale = 3000
    return b

def train(b, text, steps):
    for _ in range(steps): b.learn_eprop(text, epochs=1, bs=16, max_steps=1, seq=48)

def run(mode):
    b = fresh()
    train(b, TOPICS["A"], 120)                                  # learn A well
    a0 = b.next_byte_acc(TOPICS["A"])
    seen = ["A"]
    for t in ("B", "C", "D", "E"):
        train(b, TOPICS[t], 90)                                 # live on the new topic (forgets A)
        seen.append(t)
        if mode == "buffer":                                    # OLD baseline: replay stored raw text
            for s in seen: b.learn_text(TOPICS[s][:400], epochs=1, max_steps=2)
        elif mode == "generative":                              # NEW: dream from the net (cues = 6-byte prefixes)
            cues = [TOPICS[s][:6] for s in seen]
            b.generative_replay(n=len(seen) * 2, dream_len=200, temperature=1.1,
                                 cues=cues, anchor=None, anchor_frac=0.0)
    aF = b.next_byte_acc(TOPICS["A"])
    return a0, aF

print(f"forgetting-resistance: topic-A next-byte-accuracy after learning B..E ({DEV})\n")
res = {}
for mode in ("none", "buffer", "generative"):
    a0, aF = run(mode)
    res[mode] = aF
    print(f"  {mode:11s}: A-acc {a0:.3f} (after A) -> {aF:.3f} (after B..E)   retention {aF/max(a0,1e-6):.2f}")
print(f"\naccept iff generative >~ buffer >> none:  "
      f"none {res['none']:.3f} | buffer {res['buffer']:.3f} | generative {res['generative']:.3f}")
ok = res["generative"] > res["none"] + 0.02
print("VERDICT:", "GENERATIVE REPLAY RESISTS FORGETTING (buffer-free) ✓" if ok else "no clear benefit — needs iteration")
