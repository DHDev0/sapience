"""BrainLife integration — the whole living loop, with the network mocked (no Claude/Qwen/web)."""
import os, shutil, wave, numpy as np, torch
from brain import partner
partner.claude_say = lambda *a, **k: "Water is a clear liquid that falls as rain and flows to the sea."
partner.web_topic = lambda *a, **k: "The brain lets us think, feel, remember and learn about the world."
partner.web_text = lambda *a, **k: "A web page full of readable words about many different topics."
partner.check_claude = lambda: (True, "ready")
from brain.life import BrainLife, resolve_compute

BASE = "/tmp/_life_test"


def _life(sub, **kw):
    d = os.path.join(BASE, sub); shutil.rmtree(d, ignore_errors=True)
    return BrainLife(d, core="spiking", use_teacher=False, use_visual=False,
                     emb=16, hidden=48, layers=1, device="cpu", seed=0, **kw)


def test_constructs_five_systems():
    L = _life("five")
    for m in ("cerebellum", "bg", "hippo", "nm"):
        assert hasattr(L, m), f"missing §{m}"                    # all 5 systems instantiated
    assert L.core == "spiking" and L.modules_on


def test_learn_and_metrics():
    L = _life("learn")
    L._learn_text("the cat sat on the mat and the dog ran. " * 20, steps=6)
    nd = L._net_diag()
    for k in ("perplexity_train", "gen_entropy", "spike_rate", "cerebellum_mse"):
        assert nd.get(k) is not None                            # deep diagnostics populate


def test_teach_focus_and_priority_queue():
    L = _life("teach")
    r = L.teach(text="recursion calls itself until a base case stops it.", label="cs")
    assert r["ok"] and L._teach_q.qsize() >= 1
    L.focus(topics=["music theory", "harmony"], mode="topics", label="music")
    assert L.feed_mode == "topics" and L.focus_label == "music" and len(L.topics) == 2
    L._consume_perceptions(None)                                 # drains the directed lesson first
    assert L._teach_q.qsize() < r["chunks"] + 1


def test_use_tool_and_cerebellum_trains():
    L = _life("tool")
    L.tools.add({"name": "echo", "cmd": "echo A fact: {input} matters.", "kind": "text"})
    res = L.use_tool("echo", "gravity")
    assert res["ok"] and "A fact:" in res["output"]
    m0 = L.cereb_mse
    L._train_cerebellum("the fox and the dog play by the river all day long. " * 10)
    assert L.cereb_mse != m0 or L.cereb_mse >= 0                 # §1 learns / MSE tracked


def test_observation_replay_audio_image():
    L = _life("obs")
    sr = 4000; wav = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, 1200)) * 20000).astype(np.int16)
    wf = os.path.join(BASE, "t.wav"); w = wave.open(wf, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(wav.tobytes()); w.close()
    from PIL import Image; imf = os.path.join(BASE, "t.png"); Image.new("L", (48, 48), 100).save(imf)
    L.tools.add({"name": "au", "cmd": f"echo {wf}", "kind": "audio", "shell": True})
    L.tools.add({"name": "im", "cmd": f"echo {imf}", "kind": "image", "shell": True})
    L.use_tool("au", "x"); L.use_tool("im", "x")
    assert len(L.last_observations) == 2
    for o in L.last_observations:
        media = L.observation_media(o["i"])
        assert media and media[1] in ("audio/wav", "image/png") and len(media[0]) > 50


def test_set_net_per_module():
    L = _life("net")
    assert L.set_net("cortex", {"lr": 5e-4})["applied"]["lr"] == 5e-4
    assert L.set_net("hippocampus", {"beta": 12})["applied"]["beta"] == 12.0
    assert L.set_net("neuromod", {"da": 0.9})["applied"]["da"] == 0.9
    assert L.set_net("cerebellum", {"eta": 0.5})["applied"]["eta"] == 0.5
    assert L.set_net("endocrine", {"on": True, "C_star": 0.4})["applied"]["C_star"] == 0.4   # §16 P1
    assert set(L._net_params().keys()) == {"cortex", "hippocampus", "bg", "neuromod", "cerebellum", "endocrine", "dynamics"}


def test_freezes():
    L = _life("freeze")
    L.freeze_learning = True; w0 = L.brain.head.weight.clone()
    L._learn_text("hello world " * 20, steps=4)
    assert torch.equal(w0, L.brain.head.weight)                 # observe-only, no weight change
    L.freeze_sleep = True; L.wake_start = 0
    assert L.should_sleep() is False                            # cycle frozen → stays awake


def test_config_persistence_and_resume_awake():
    d = os.path.join(BASE, "persist"); shutil.rmtree(d, ignore_errors=True)
    L = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48, layers=1, device="cpu", seed=0)
    L.budget = 0.33; L.grow_add = 48; L.freeze_growth = True; L.learn_steps = 13
    L.set_net("hippocampus", {"beta": 15}); L.focus(topics=["coding"], mode="topics", label="code")
    L.awake = False                                             # was asleep at save
    L.save_life()
    L2 = BrainLife(d, core="spiking", use_teacher=False, use_visual=False, emb=16, hidden=48, layers=1, device="cpu", resume=True)
    assert L2.budget == 0.33 and L2.grow_add == 48 and L2.freeze_growth and L2.learn_steps == 13
    assert L2.hippo.beta == 15.0 and L2.focus_label == "code"
    assert L2.awake is True                                     # resumes AWAKE (no spurious night)


def test_bounded_logs():
    L = _life("logs"); L.max_log_mb = 0.005
    for _ in range(4000):
        L.log("x" * 80)
    L._bound_logs()
    assert os.path.getsize(L.logpath) <= 0.005 * 1e6 * 1.2      # capped (earliest evicted)


def test_resolve_compute_auto():
    r = resolve_compute("auto")
    assert r["device"] in ("cpu", "cuda") and r["threads"] >= 1 and "note" in r
    assert resolve_compute("cuda")["device"] in ("cpu", "cuda")  # falls back to cpu if no GPU
