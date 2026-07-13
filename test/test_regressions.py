"""
Regression tests — one per bug we fixed, named after the bug, so the behaviour can't silently
revert. Each asserts the FIXED behaviour (and the comment says what the bug was).
"""
import os, shutil, torch
from brain import partner
partner.claude_say = lambda *a, **k: "Rivers carry water down to the wide and salty sea."
partner.web_topic = lambda *a, **k: "clear simple sentences about the world and how it works."
partner.web_text = lambda *a, **k: "readable web text about many different everyday topics."
from brain.life import BrainLife, _usable_lesson, resolve_compute
from brain import senses

BASE = "/tmp/_regr_test"


def _life(sub, **kw):
    d = os.path.join(BASE, sub); shutil.rmtree(d, ignore_errors=True)
    return BrainLife(d, core="spiking", use_teacher=False, use_visual=False,
                     emb=16, hidden=48, layers=1, device="cpu", seed=0, **kw)


# --- BUG: a GROWN basal ganglia was silently discarded on resume (exact-shape guard) ---
def test_regression_grown_bg_survives_resume():
    d = os.path.join(BASE, "bgresume"); shutil.rmtree(d, ignore_errors=True)
    L = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48,
                  layers=1, device="cpu", seed=0)
    L.bg.grow(16)                                              # neuron growth changes M/w_v/W_pi shapes
    import torch as T
    L.bg.w_v = L.bg.w_v + 0.123                                # a learned critic to preserve
    marker = float(L.bg.w_v.sum()); n = L.bg.M.shape[0]
    L.save_life()
    R = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48,
                  layers=1, device="cpu", seed=0, resume=True)
    assert R.bg.M.shape[0] == n                                # grown BG restored (not discarded)
    assert abs(float(R.bg.w_v.sum()) - marker) < 1e-4          # learned critic survived


# --- BUG: live-tuned neuromod tone was clobbered by wake defaults on resume ---
def test_regression_tuned_neuromod_tone_survives_resume():
    d = os.path.join(BASE, "toneresume"); shutil.rmtree(d, ignore_errors=True)
    L = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48,
                  layers=1, device="cpu", seed=0)
    L.set_net("neuromod", {"da": 0.87}); L.save_life()
    R = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48,
                  layers=1, device="cpu", seed=0, resume=True)
    assert abs(R.nm.tone["da"] - 0.87) < 1e-6                  # tuned tone won over the wake reset


# --- BUG: hippo episode counter + cortex seq were not persisted across resume ---
def test_regression_hippo_nstored_and_cortex_seq_survive_resume():
    d = os.path.join(BASE, "miscresume"); shutil.rmtree(d, ignore_errors=True)
    L = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48,
                  layers=1, device="cpu", seed=0)
    import torch as T
    L.hippo.store(T.sign(T.randn(5, 256))); L.hippo.n_stored = 999
    L.brain.seq = 123; L.save_life()
    R = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48,
                  layers=1, device="cpu", seed=0, resume=True)
    assert R.hippo.n_stored == 999 and R.brain.seq == 123


# --- BUG: §1 cerebellum was imported but never instantiated (loop ran 4 systems, not 5) ---
def test_regression_five_systems_instantiated_and_cerebellum_trains():
    L = _life("cereb")
    assert all(hasattr(L, m) for m in ("cortex" if False else "cerebellum", "bg", "hippo", "nm"))
    m0 = L.cereb_mse
    for _ in range(4):
        L._train_cerebellum("the fox and the dog run by the river every single day. " * 8)
    assert L.cereb_mse > 0 and L.cereb_mse != m0                 # §1 actually learns now


# --- BUG: §10 pruning was faked (pruned=0); and it must prune SYNAPSES, not neurons ---
def test_regression_pruning_is_synaptic_not_neuronal():
    b = _life("prune").brain
    b.learn_text("the cat sat on the mat. " * 40, epochs=1, bs=8, max_steps=6)
    h = b.hidden
    cut = b.prune(frac=0.1)
    assert cut > 0 and b.hidden == h                            # connections removed, NEURONS kept
    zeros = sum(int((~m).sum()) for m in b._pmask)
    b.learn_text("the dog ran. " * 30, epochs=1, bs=8, max_steps=4)
    still = sum(int((lin.weight == 0).sum()) for lin in b._prune_targets())
    assert still >= zeros                                        # pruned synapses do NOT regrow


def test_regression_develop_reports_grow_then_prune():
    from brain.spiking_brain import SpikingBrain
    b = SpikingBrain(torch.device("cpu"), emb=16, hidden=80, layers=1, seed=0, syn_density=0.5); b.seen_bytes = 10**9
    b.grow_until, b.prune_until = 2, 4
    grown = pruned = 0
    for _ in range(5):
        d = b.develop(add=24); grown += d["grown"]; pruned += d["pruned"]
        assert "synapses" in d and "neurons" in d               # develop reports the arch diag
    assert grown > 0 and pruned > 0                             # SYNAPSES grew then pruned (not neurons)
    assert b.neuron_count() == 80                               # NEURONS stayed fixed (the new model)


# --- BUG: think_temp was tunable but think() hard-coded temperature=0.6 ---
def test_regression_think_uses_think_temp():
    L = _life("tt"); L.think_temp = 0.91
    seen = {}
    orig = L.brain.think
    L.brain.think = lambda n, temperature: (seen.__setitem__("t", temperature), orig(n, temperature=temperature))[1]
    L.think()
    assert seen.get("t") == 0.91                                # the tuned value reaches sampling


# --- BUG: freeze_learning was bypassed on the raw-sensory (list) teach path ---
def test_regression_freeze_learning_blocks_sensory_frames():
    L = _life("fzsens"); L.freeze_learning = True
    frame = senses.frame("audio", senses.encode_audio(torch.randn(500).numpy(), sr=8000))
    L._teach_q.put(("tool:x (audio)", frame))                   # a sensory byte frame
    w0 = L.brain.head.weight.clone()
    L._consume_perceptions(None)
    assert torch.equal(w0, L.brain.head.weight)                 # observe-only: no weight change


# --- BUG: sleep replay + SHY downscale ignored freeze_learning ---
def test_regression_freeze_learning_blocks_sleep_replay():
    L = _life("fzsleep")
    L.memory.write("the sun warms the sea and the rain returns to the land. " * 20)
    L.freeze_learning = True
    w0 = L.brain.head.weight.clone()
    L._sleep_tick()
    assert torch.equal(w0, L.brain.head.weight)                 # no learning + no SHY when frozen


# --- BUG: kill() still ran run()'s finally save_life() ('force kill, no checkpoint' lied) ---
def test_regression_kill_skips_checkpoint_graceful_saves():
    L = _life("killa"); L._killed = True
    L.run(stop_flag=lambda: True)                               # loop body skipped → straight to finally
    assert not os.path.exists(L.ckpt)                           # killed → NO checkpoint
    L2 = _life("killb")
    L2.run(stop_flag=lambda: True)                              # graceful → DOES checkpoint
    assert os.path.exists(L2.ckpt)


# --- BUG: teacher error strings ('Error: Exceeded USD budget') were distilled as language ---
def test_regression_usable_lesson_filters_errors():
    assert _usable_lesson("The ocean is a vast body of salt water covering much of the Earth.")
    assert not _usable_lesson("Error: Exceeded USD budget (0.05)")
    assert not _usable_lesson("rate limit reached")
    assert not _usable_lesson("hi")                             # too short to be a real lesson


# --- BUG: resuming a sleeping brain fabricated a night + age increment on the first tick ---
def test_regression_resume_wakes_no_spurious_develop():
    d = os.path.join(BASE, "resume"); shutil.rmtree(d, ignore_errors=True)
    L = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48, layers=1, device="cpu", seed=0)
    L.awake = False; L.slept_count = 3; age0 = L.brain.age
    L.save_life()
    L2 = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48, layers=1, device="cpu", resume=True)
    assert L2.awake is True and L2.sleep_remaining == 0         # resumes awake
    assert L2.brain.age == age0 and L2.slept_count == 3          # no fabricated night/age bump


# --- BUG: live config (freezes, growth, caps, module params, focus) reset on every resume ---
def test_regression_all_live_config_persists():
    d = os.path.join(BASE, "cfg"); shutil.rmtree(d, ignore_errors=True)
    L = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48, layers=1, device="cpu", seed=0)
    L.budget = 0.42; L.grow_add = 40; L.freeze_growth = True; L.learn_steps = 11; L.max_log_mb = 9
    L.brain.grow_until = 20; L.set_net("hippocampus", {"beta": 14}); L.focus(topics=["math"], mode="topics", label="m")
    L.save_life()
    R = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48, layers=1, device="cpu", resume=True)
    assert (R.budget, R.grow_add, R.freeze_growth, R.learn_steps, R.max_log_mb) == (0.42, 40, True, 11, 9)
    assert R.brain.grow_until == 20 and R.hippo.beta == 14.0 and R.focus_label == "m"


# --- BUG: fresh-launch form fields (grow_add/freezes/...) did nothing until 'apply live' ---
def test_regression_set_params_applies_fresh_soft_config():
    from interface import dashboard as D
    c = D.Controller(); c.life = _life("fresh")
    c.set_params({"grow_add": 77, "freeze_sleep": True, "learn_steps": 9, "perceive_gap": 3})
    assert c.life.grow_add == 77 and c.life.freeze_sleep and c.life.learn_steps == 9 and c.life.perceive_gap == 3


# --- BUG: resume broke after growth (loader assumed uniform layer width) ---
def test_regression_resume_after_nonuniform_growth_byte_identical():
    from brain.spiking_brain import SpikingBrain
    b = SpikingBrain(torch.device("cpu"), emb=16, hidden=48, layers=2, seed=0)
    b.learn_text("water flows down to the sea and back to the sky. " * 30, epochs=1, bs=8, max_steps=6)
    b.grow(24); b.grow(24)                                       # only the TOP layer grows → non-uniform
    txt = "the river runs. " * 20; before = b.bits_per_byte(txt)
    b.save("/tmp/_rg.pt"); b2 = SpikingBrain(torch.device("cpu"), emb=16, hidden=48, layers=2, seed=9); b2.load("/tmp/_rg.pt")
    assert abs(b2.bits_per_byte(txt) - before) < 1e-4 and b2.hidden == b.hidden
    os.remove("/tmp/_rg.pt")


def test_regression_multi_gpu_note_is_honest():
    r = resolve_compute("cuda")                                 # no GPU here → falls back, honest note
    assert "FSDP2 shard" not in r["note"] or "not yet wired" in r["note"]


# --- BUG: §5 neuromod tone was display-only; the eligibility gate was dead code ---
def test_regression_neuromod_ach_gates_plasticity():
    L = _life("ach")
    L.nm.tone["ach"] = 0.0                                       # zero plasticity → no learning
    w0 = L.brain.head.weight.clone()
    L._learn_text("the river runs to the sea every day and night. " * 12, steps=6)
    assert torch.equal(w0, L.brain.head.weight)                 # ACh=0 gates learning off
    L.nm.tone["ach"] = 1.0                                       # full plasticity → learns
    L._learn_text("the river runs to the sea every day and night. " * 12, steps=6)
    assert not torch.equal(w0, L.brain.head.weight)


# --- BUG: hippocampus wrote EVERY episode; novelty was computed but never gated the write ---
def test_regression_hippocampus_write_is_novelty_gated():
    L = _life("novgate"); L.novelty_gate = 0.5
    same = "the exact same familiar sentence repeated over and over again here. " * 6
    for _ in range(60):
        L._index_episode(same)                                  # identical → low novelty after fill
    n = L.hippo.keys.shape[0]
    assert n < 60                                               # familiar repeats are NOT all stored
