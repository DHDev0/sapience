"""THE capability-vs-fidelity curve — the novel measurement (§15.16).

Each biological faithfulness constraint costs some capability (bits/byte). This measures that cost by
training an otherwise-identical 16k-neuron spiking cortex under each constraint (added one at a time,
per the "measure after each" discipline) and reporting bits/byte. The non-plausible BPTT run is the
capability ceiling; the full faithful stack is the fidelity end. Writes JSON + a markdown table for the paper.

Run: HIP_VISIBLE_DEVICES=1 python runs/fidelity_capability_curve.py   (GPU1, ~15 min)
"""
import os, sys, json, time
sys.path.insert(0, "/home/dander/workspace/zk/sapience")
import torch
from brain.spiking_brain import SpikingBrain

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda": torch.zeros(1, device=DEV)
HID, LAYERS, STEPS, BS, SEQ = 8000, 2, 220, 16, 48          # 16k neurons
SCALE = 3000.0
TXT = (" the quick brown fox jumps over the lazy dog. water runs down to the sea. the sun rose over the "
       "hills as birds sang. a cat slept on the warm mat. rain fell on the green fields. children played "
       "near the old stone bridge. the moon is a round rock in the night sky. bread is made from wheat. "
       "trees grow tall and give shade. the ocean is deep and blue. fire is hot and gives light. ") * 30

CONFIGS = [
    ("BPTT (non-plausible ceiling)", dict(learn_rule="bptt")),
    ("e-prop + random feedback (DFA)", dict(learn_rule="eprop", feedback_mode="random")),
    ("+ learned feedback (Kolen-Pollack)", dict(learn_rule="eprop", feedback_mode="learned")),
    ("+ Dale's law (E/I typing)", dict(learn_rule="eprop", feedback_mode="learned", dale=True)),
    ("+ dendritic / burst error", dict(learn_rule="eprop", feedback_mode="learned", dendritic=True)),
    ("+ bounded synapses (Fusi)", dict(learn_rule="eprop", feedback_mode="learned", bounded_synapses=True)),
    ("+ firing-rate homeostasis", dict(learn_rule="eprop", feedback_mode="learned", homeostasis=True)),
    ("+ BTSP long eligibility", dict(learn_rule="eprop", feedback_mode="learned", btsp=True)),
    ("+ unified two-compartment", dict(learn_rule="eprop", feedback_mode="learned", two_compartment=True)),
    ("+ differentiated neuromod", dict(learn_rule="eprop", feedback_mode="learned", diff_neuromod=True)),
    ("+ stochastic spiking", dict(learn_rule="eprop", feedback_mode="learned", stochastic=True)),
    ("+ metabolic cost", dict(learn_rule="eprop", feedback_mode="learned", metabolic=True)),
    ("FULL faithful stack (all)", dict(learn_rule="eprop", feedback_mode="learned", two_compartment=True,
                                       diff_neuromod=True, stochastic=True, metabolic=True, dale=True,
                                       bounded_synapses=True, homeostasis=True, btsp=True)),
]

def run(cfg):
    torch.manual_seed(0)
    b = SpikingBrain(DEV, emb=64, hidden=HID, layers=LAYERS, cell="lif", seed=0,
                     sparse=True, rec_fanin=32, in_fanin=32, syn_density=0.5)
    b.eprop_lr_scale = SCALE
    b.set_faith(**cfg)
    bpb0 = b.bits_per_byte(TXT); t0 = time.time()
    b.learn_text(TXT, epochs=1, bs=BS, max_steps=STEPS, seq=SEQ,
                 tone=dict(da=0.5, ach=1.0, ne=1.0, ht=0.5))   # wake tone (diff_neuromod uses it)
    bpb1 = b.bits_per_byte(TXT); dt = (time.time() - t0) / STEPS
    ws = b.weight_stats()
    return dict(bpb0=round(bpb0, 3), bpb=round(bpb1, 3), ms_step=round(dt * 1000, 1),
                spike_rate=round(b.spike_rate(TXT), 4),
                metrics={k: round(v, 4) for k, v in ws.items()
                         if k in ("fb_align_cos", "ei_frac_excit", "burst_frac", "homeo_thr_mean", "synapse_sat_frac")})

results = []
base = None
print(f"capability-vs-fidelity: {HID*LAYERS//1000}k neurons, {STEPS} steps, scale {SCALE}\n", flush=True)
for name, cfg in CONFIGS:
    r = run(cfg); r["name"] = name; results.append(r)
    if "learned feedback" in name and base is None: base = r["bpb"]
    print(f"{name:38s} bpb {r['bpb0']:.2f}->{r['bpb']:.2f}  {r['ms_step']:.0f}ms/step  spk {r['spike_rate']}  {r['metrics']}", flush=True)

# fidelity cost = bpb above the learned-feedback plausible baseline
for r in results:
    r["cost_vs_plausible_base"] = round(r["bpb"] - base, 3) if base else None

out = dict(config=dict(hidden=HID, layers=LAYERS, neurons=HID * LAYERS, steps=STEPS, scale=SCALE), results=results)
with open("/home/dander/workspace/zk/sapience/runs/fidelity_capability_curve.json", "w") as f:
    json.dump(out, f, indent=2)

md = ["# Capability vs. fidelity — measured cost of each biological constraint", "",
      f"16k-neuron spiking cortex, {STEPS} e-prop steps, identical seed/data. bits/byte (lower = more capable).",
      "Cost = bits/byte above the plausible learned-feedback baseline.", "",
      "| Configuration | bits/byte | cost vs plausible base | ms/step | spike rate |",
      "|---|---|---|---|---|"]
for r in results:
    md.append(f"| {r['name']} | {r['bpb']:.3f} | {r['cost_vs_plausible_base']:+.3f} | {r['ms_step']:.0f} | {r['spike_rate']} |")
with open("/home/dander/workspace/zk/sapience/runs/fidelity_capability_curve.md", "w") as f:
    f.write("\n".join(md) + "\n")
print("\nwrote runs/fidelity_capability_curve.{json,md}", flush=True)
