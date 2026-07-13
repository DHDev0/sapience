"""Unified sensory code (senses.py) + internal clock, and the tiered episodic memory."""
import os, shutil, numpy as np, torch
from brain import senses
from brain.memory import EpisodicMemory


def test_encoders_produce_byte_levels():
    t = senses.encode_text("hi")
    assert t == [104, 105]
    a = senses.encode_audio(np.sin(np.linspace(0, 6, 500)).astype(np.float32), sr=8000)
    assert a and all(0 <= x <= 255 for x in a)                   # cochlea → 8-bit
    from PIL import Image
    im = senses.encode_image(Image.new("L", (10, 10), 128), size=8)
    assert len(im) == 64 and all(0 <= x <= 255 for x in im)      # retina → size×size


def test_nerve_tagged_frames():
    f = senses.sense("text", "x")
    assert f[:3] == list(senses.NERVE["text"]) and f[-1] == senses.FRAME_END[0]


def test_clock_tells_and_predicts_time():
    c = senses.Clock(t0=0.0)
    for k in range(50):
        c.tick(now=float(k))
    assert c.events == 50
    st = c.stamp(now=50.0)
    assert len(st) > 0 and all(0 <= x <= 255 for x in st)        # time as byte-levels
    assert "events" in c.tell()


def test_memory_write_compress_evict():
    d = "/tmp/_mem_test"; shutil.rmtree(d, ignore_errors=True)
    m = EpisodicMemory(d, hot_mb=0.02, hard_gb=0.001, segment_mb=0.01, level=6)
    for i in range(200):
        m.write("the brain remembers its whole life. " * 20)
    s = m.stats()
    assert s["lived_chars"] > 0
    assert s["disk_mb"] <= 0.001 * 1000 + 0.05                   # bounded by the hard cap (evicted)
    chunk = m.sample(500)
    assert chunk is None or isinstance(chunk, str)
    shutil.rmtree(d, ignore_errors=True)
