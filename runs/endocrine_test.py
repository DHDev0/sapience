"""P1 acceptance test (§16): the SpikingEndocrine drive/cortisol/mood dynamics.
Verifies: (1) satiation → focus (met drive lowers NE-gain, emits homeostatic reward); (2) cortisol
inverted-U on plasticity; (3) graceful degradation — chronic stress raises allostatic load + drops the
plasticity ceiling, and SLEEP recovers it.  Run: python runs/endocrine_test.py"""
import sys, math
sys.path.insert(0, "/home/dander/workspace/zk/sapience")
from brain.endocrine import SpikingEndocrine

ok = True
def check(name, cond):
    global ok; ok = ok and cond
    print(f"  {'✓' if cond else '✗'} {name}")

# (1) satiation → focus + homeostatic reward
print("(1) drive / satiation → focus:")
e = SpikingEndocrine(); e.on = True
for _ in range(150): e.wake_tick(progress=0.0, novelty=0.0)         # STARVED: no learning, no novelty
starved_ne = e.ne_gain(); starved_D = e.D_energy + e.D_novelty
r_meet = e.wake_tick(progress=0.5, novelty=1.0)                     # a big satiation event (learning + novelty)
fed_ne = e.ne_gain(); fed_D = e.D_energy + e.D_novelty
check("starved raises the drive deficit", starved_D > 0.5)
check("satiation emits a homeostatic reward (r_home>0)", r_meet > 0.0)
check("satiation LOWERS NE-gain → focus (Aston-Jones-Cohen)", fed_ne < starved_ne)
print(f"     starved: D={starved_D:.2f} NE={starved_ne:.2f}  ->  fed: D={fed_D:.2f} NE={fed_ne:.2f}  r_home={r_meet:.3f}")

# (2) cortisol one-sided gate on plasticity (calm→optimal full; only chronic-high impairs)
print("(2) cortisol plasticity gain g(C) (one-sided: calm full, chronic-high impairs):")
e2 = SpikingEndocrine()
gains = []
for C in (0.0, 0.15, 0.35, 0.6, 1.0):
    e2.C = C; e2.AL = 0.0; gains.append((C, e2.plasticity_gain()))
check("calm→optimal cortisol = FULL plasticity (unthrottled)", gains[0][1] >= 0.95 and gains[2][1] >= 0.95)
check("chronic-high cortisol impairs plasticity", gains[-1][1] < gains[2][1])
print("     " + "  ".join(f"C={c:.2f}:g={g:.2f}" for c, g in gains))

# (3) chronic stress → allostatic load → impaired; sleep recovers
print("(3) chronic stress → allostatic load, then sleep recovery:")
e3 = SpikingEndocrine(); e3.on = True
for _ in range(400): e3.wake_tick(threat=0.9, surprise=0.5, progress=0.0)   # sustained high threat
stressed_gain = e3.plasticity_gain(); stressed_AL = e3.AL; stressed_C = e3.C
for _ in range(400): e3.sleep_tick()                                        # a long recovery sleep
rested_gain = e3.plasticity_gain(); rested_AL = e3.AL; rested_C = e3.C
check("chronic stress accrues allostatic load", stressed_AL > 0.05)
check("chronic stress DROPS the plasticity ceiling", stressed_gain < 0.6)
check("sleep RELIEVES cortisol", rested_C < stressed_C)
check("sleep recovers the plasticity gain", rested_gain > stressed_gain)
print(f"     stressed: C={stressed_C:.2f} AL={stressed_AL:.2f} g={stressed_gain:.2f}  ->  "
      f"rested: C={rested_C:.2f} AL={rested_AL:.2f} g={rested_gain:.2f}")

print("\nVERDICT:", "ENDOCRINE DYNAMICS CORRECT ✓" if ok else "FAILED — needs fix")
sys.exit(0 if ok else 1)
