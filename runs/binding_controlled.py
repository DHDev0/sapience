"""Controlled binding proof (critic pivot #4): does the MEM3 fast-Hebbian store actually BIND variables, isolated
from the LM confound? Two synthetic tasks where binding MUST show if it works and CANNOT be faked by n-gram stats:
  COPY   : [S random bytes] SEP [same S bytes]      — recall the whole span (tests content storage)
  RECALL : (k v) pairs, DELAY filler, then a key k   — predict its value v (tests associative binding at a delay)
fast_mem OFF (byte-identical control) vs ON, sweeping span/delay past the ~4-byte membrane horizon. Success (pre-
registered): ON beats OFF by >2·std beyond ~4 bytes; curves converge below it. Usage: binding_controlled.py [task] [steps]"""
import sys, random, time
sys.path.insert(0, "/home/dander/workspace/zk/sapience")
import torch, torch.nn.functional as F
from brain.spiking_brain import SpikingBrain

TASK = sys.argv[1] if len(sys.argv) > 1 else "copy"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
HID, LAY, BS = 512, 2, 32
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = list(range(97, 123))                              # 26 lowercase letters as the token alphabet
SEP, PAD = 32, 46                                         # ' ' and '.'


def copy_batch(rng, S):
    seqs = []
    for _ in range(BS):
        span = [rng.choice(VOCAB) for _ in range(S)]
        seqs.append(span + [SEP] + span)                 # predict the 2nd copy from the 1st (needs storage)
    x = torch.tensor(seqs, device=DEV)
    return x[:, :-1], x[:, 1:], S                         # copy region = targets[S+1 : 2S+1]


def recall_batch(rng, delay, npair=4):
    seqs = []
    for _ in range(BS):
        keys = rng.sample(VOCAB, npair); vals = [rng.choice(VOCAB) for _ in range(npair)]
        s = []
        for k, v in zip(keys, vals): s += [k, v]         # kv pairs
        s += [rng.choice(VOCAB) for _ in range(delay)]   # filler
        qi = rng.randrange(npair); s += [keys[qi], vals[qi]]   # query key → its value (to predict)
        seqs.append(s)
    x = torch.tensor(seqs, device=DEV)
    return x[:, :-1], x[:, 1:], None                     # recall target = the very last byte (the value)


def make(fast):
    torch.manual_seed(0)
    b = SpikingBrain(DEV, dtype=torch.float32, emb=48, hidden=HID, layers=LAY, seq=64, seed=0,
                     sparse=True, rec_fanin=48, in_fanin=48)
    b.to(DEV); b._ensure_feedback(); b.set_faith(learn_rule="eprop", learn_opt="adam", adam_lr=2e-3, feedback_mode="learned")
    if fast:
        b.set_fastmem(fast_mem=True, mem_fast_decay=0.96, mem_learn_read=True)   # the binding mechanism under test
        b.set_mem(gated_slow=True, read_mem=True)                                # + the store readout
    return b


@torch.no_grad()
def eval_copy(b, S, n=20):
    rng = random.Random(999); acc = 0.0
    for _ in range(n):
        x, y, _ = copy_batch(rng, S); lo, _ = b._run(x)
        pred = lo.argmax(-1)                             # accuracy on the 2nd-copy region only
        reg = pred[:, S+1:S+1+S].eq(y[:, S+1:S+1+S]).float().mean()
        acc += float(reg)
    return round(acc / n, 3)


@torch.no_grad()
def eval_recall(b, delay, n=30):
    rng = random.Random(999); acc = 0.0
    for _ in range(n):
        x, y, _ = recall_batch(rng, delay); lo, _ = b._run(x)
        acc += float(lo.argmax(-1)[:, -1].eq(y[:, -1]).float().mean())   # last byte = the recalled value
    return round(acc / n, 3)


def run(fast, task):
    b = make(fast); rng = random.Random(0); t0 = time.time()
    sweep = [2, 4, 8, 16, 32] if task == "copy" else [2, 4, 8, 16, 32]
    for s in range(STEPS):
        if task == "copy": x, y, _ = copy_batch(rng, rng.choice([2, 4, 8, 16]))
        else: x, y, _ = recall_batch(rng, rng.choice([2, 4, 8, 16]))
        b._eprop_step(x, y, gate=1.0)
    res = {k: (eval_copy(b, k) if task == "copy" else eval_recall(b, k)) for k in sweep}
    print(f"  [{task} fast={fast}] {res}  ({time.time()-t0:.0f}s)", flush=True)
    return res


print(f"=== BINDING PROOF task={TASK} steps={STEPS} (chance≈{1/26:.3f}) ===", flush=True)
off = run(False, TASK)
on = run(True, TASK)
print(f"[{TASK}] OFF={off}", flush=True)
print(f"[{TASK}] ON ={on}", flush=True)
delta = {k: round(on[k] - off[k], 3) for k in off}
print(f"[{TASK}] ON-OFF={delta}  (binding works if ON≫OFF for larger spans/delays past ~4)", flush=True)
print("BINDING DONE", flush=True)
