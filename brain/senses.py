"""
senses.py — the universal sensory (and motor) code: everything becomes "electricity".

Like the brain turning sight, sound and speech into the SAME electrical spikes, every
modality here is converted to ONE common stream of byte-levels (0-255 = a quantised
"voltage"). The byte-brain then learns from any sense uniformly, exactly as it learns
text — because it is all just a stream of levels.

  text   -> utf-8 bytes                                    (language)
  image  -> downscaled grayscale pixels, row-major bytes   (retina / optic nerve)
  audio  -> resampled, 8-bit quantised samples             (cochlea)

Each sensory frame is tagged with a short "nerve" marker so the brain can tell which
sense fired (different afferents), and frames are concatenated into one life-stream.
The same code runs in reverse for MOTOR output: the brain emits bytes, which the motor
system reads as text to type/say or as a command to act on (see motor.py).
"""
from __future__ import annotations
import math
import time as _time
import numpy as np

# distinctive multi-byte "nerve" markers (rare in natural text/pixels)
NERVE = {
    "text":  bytes([0x02, 0x54, 0x02]),   # STX 'T'
    "image": bytes([0x02, 0x49, 0x02]),   # STX 'I'
    "audio": bytes([0x02, 0x41, 0x02]),   # STX 'A'
    "self":  bytes([0x02, 0x53, 0x02]),   # STX 'S' — the brain's own output (proprioception)
    "time":  bytes([0x02, 0x43, 0x02]),   # STX 'C' — the internal clock (sense of time)
}
FRAME_END = bytes([0x03])                 # ETX between sensory frames


class Clock:
    """The internal clock — a sense of time from the SEQUENCE OF EVENTS (and real wall
    time), the way the brain keeps time with a pacemaker-accumulator plus a bank of
    oscillators at many periods (striatal beat-frequency interval timing + circadian
    rhythm). At each moment the combined PHASE of the oscillators encodes how much time
    has passed; two slow accumulators give absolute age. `stamp()` renders that state
    into byte-levels so it flows in the SAME sensory code as sight and sound — the brain
    perceives time. Because the oscillators are regular, a matured brain learns to
    PREDICT the clock, i.e. it can tell time.
    """
    EVENT_PERIODS = [8, 32, 128, 512, 2048, 8192]        # oscillator periods, in events
    TIME_PERIODS = [10, 60, 600, 3600, 86400]            # oscillator periods, in seconds

    def __init__(self, t0=None):
        self.events = 0
        self.t0 = t0 if t0 is not None else _time.time()
        self.last_event_t = self.t0
        self.last_interval = 0.0

    def tick(self, now=None):
        """Advance the clock by one event (a beat of the pacemaker)."""
        now = now if now is not None else _time.time()
        self.events += 1
        self.last_interval = now - self.last_event_t
        self.last_event_t = now
        return self

    @staticmethod
    def _phase_bytes(value, periods):
        out = []
        for p in periods:
            ph = (value % p) / p                          # position on this 'hand'
            out.append(int((math.sin(2 * math.pi * ph) * 0.5 + 0.5) * 255))
            out.append(int((math.cos(2 * math.pi * ph) * 0.5 + 0.5) * 255))
        return out

    def stamp(self, now=None):
        """Byte-encoded time: event-sequence oscillators + wall-clock oscillators +
        two accumulator ramps (coarse absolute age)."""
        now = now if now is not None else _time.time()
        ev = self._phase_bytes(self.events, self.EVENT_PERIODS)
        rt = self._phase_bytes(now - self.t0, self.TIME_PERIODS)
        acc = [min(255, self.events // 16), min(255, int((now - self.t0) / 60))]
        return ev + rt + acc

    def elapsed(self, now=None):
        now = now if now is not None else _time.time()
        return dict(events=self.events, seconds=round(now - self.t0, 1))

    def tell(self, now=None):
        e = self.elapsed(now)
        return f"~{e['events']} events / {e['seconds']:.0f}s since birth"


def encode_text(text: str) -> list:
    return list(text.encode("utf-8", errors="replace"))


def encode_image(img, size: int = 48) -> list:
    """PIL image -> size×size grayscale, row-major byte stream (the 'retina')."""
    g = img.convert("L").resize((size, size))
    return list(np.asarray(g, dtype=np.uint8).reshape(-1))


def encode_audio(wav, sr: int, target_sr: int = 4000, max_samples: int = 4000) -> list:
    """Waveform (np array in [-1,1] or int) -> resampled, 8-bit quantised samples (the 'cochlea')."""
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return []
    if np.issubdtype(np.asarray(wav).dtype, np.integer):
        x = x / 32768.0
    if sr != target_sr:                              # cheap linear resample (no librosa)
        n = int(x.size * target_sr / sr)
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x)
    x = x[:max_samples]
    q = np.clip((x * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return list(q)


def frame(modality: str, data) -> list:
    """Wrap an encoded stream with its nerve marker + frame terminator."""
    body = data if isinstance(data, list) else encode_text(str(data))
    return list(NERVE[modality]) + body + list(FRAME_END)


def sense(modality: str, raw, **kw) -> list:
    """Encode a raw sensory input of `modality` into a tagged byte frame.
    For `time`, raw is a Clock (its current stamp is used)."""
    if modality == "text":
        enc = encode_text(raw)
    elif modality == "image":
        enc = encode_image(raw, **kw)
    elif modality == "audio":
        enc = encode_audio(raw, **kw)
    elif modality == "time":
        enc = raw.stamp() if isinstance(raw, Clock) else list(raw)
    else:
        raise ValueError(f"unknown modality {modality}")
    return frame(modality, enc)


class SensoryStream:
    """Accumulate multimodal frames into one continuous life-stream of byte-levels."""

    def __init__(self):
        self.buf = []

    def add(self, modality, raw, **kw):
        f = sense(modality, raw, **kw)
        self.buf.extend(f)
        return f

    def add_frame(self, modality, encoded):
        f = frame(modality, encoded)
        self.buf.extend(f)
        return f

    def drain(self):
        s = self.buf
        self.buf = []
        return s
