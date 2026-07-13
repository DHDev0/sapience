"""
memory.py — EpisodicMemory: the brain's tiered life-history of experienced TEXT.

A three-tier store with automatic eviction, so it lives within fixed RAM + disk caps:

  RAM  (hot)     recent experience, uncompressed, instantly replayable
  SSD  (cold)    older experience as HIGH-RATIO compressed segments (zstd-19, ~100×)
  evict          two phases: (1) at the RAM/soft limit, flush+COMPRESS the oldest hot
                 text to a disk segment; (2) at the global/hard disk limit, DELETE the
                 oldest compressed segments.

Sleep/consolidation replays random text chunks from hot + a random decompressed old
segment, so the brain rehearses its whole life, not just the recent past. Text is what
the recurrent cortex learns from (BPTT on chunks), and text compresses enormously — so
a long life fits in a small, bounded footprint.
"""
from __future__ import annotations
import os, json, random

try:
    import zstandard as _zstd
    _HAVE_ZSTD = True
except Exception:
    import lzma as _lzma
    _HAVE_ZSTD = False


class EpisodicMemory:
    def __init__(self, disk_dir, hot_mb=64.0, hard_gb=10.0, segment_mb=8.0, level=19):
        self.dir = disk_dir
        os.makedirs(disk_dir, exist_ok=True)
        self.hot = []                      # recent text pieces (RAM)
        self.hot_chars = 0
        self.hot_cap = int(hot_mb * 1e6)
        self.seg_cap = int(segment_mb * 1e6)
        self.hard = int(hard_gb * 1e9)
        self.level = level
        self.index_path = os.path.join(disk_dir, "segments.json")
        self.segments = self._load_index()
        self._t = len(self.segments)

    # ---- compression (max ratio) ------------------------------------- #
    def _comp(self, b):
        return _zstd.ZstdCompressor(level=self.level).compress(b) if _HAVE_ZSTD else _lzma.compress(b, preset=9)

    def _decomp(self, b):
        return _zstd.ZstdDecompressor().decompress(b) if _HAVE_ZSTD else _lzma.decompress(b)

    # ---- write experience -------------------------------------------- #
    def write(self, text):
        if not text:
            return
        self.hot.append(text); self.hot_chars += len(text)
        while self.hot_chars > self.hot_cap and self.hot:
            self._flush_segment()          # PHASE 1: compress oldest hot → SSD segment
        self._enforce_hard()               # PHASE 2: delete oldest segments over the hard cap

    def _flush_segment(self):
        buf, size = [], 0
        while self.hot and size < self.seg_cap:
            p = self.hot.pop(0); buf.append(p); size += len(p); self.hot_chars -= len(p)
        raw = "".join(buf).encode("utf-8", "replace")
        if not raw:
            return
        comp = self._comp(raw)
        self._t += 1
        path = os.path.join(self.dir, f"seg_{self._t:06d}.zst")
        with open(path, "wb") as f:
            f.write(comp)
        self.segments.append(dict(path=path, comp=len(comp), chars=len(raw), t=self._t))
        self._save_index()

    def _enforce_hard(self):
        total = sum(s["comp"] for s in self.segments)
        changed = False
        while total > self.hard and self.segments:
            old = self.segments.pop(0)     # oldest, already-compressed → delete
            try: os.remove(old["path"])
            except Exception: pass
            total -= old["comp"]; changed = True
        if changed:
            self._save_index()

    # ---- replay a random chunk of the whole life --------------------- #
    def sample(self, n_chars=2000):
        if self.segments and (not self.hot or random.random() < 0.4):
            s = random.choice(self.segments)
            try:
                txt = self._decomp(open(s["path"], "rb").read()).decode("utf-8", "replace")
            except Exception:
                txt = ""
        else:
            txt = "".join(self.hot[-80:])
        if len(txt) <= n_chars:
            return txt
        i = random.randint(0, len(txt) - n_chars)
        return txt[i:i + n_chars]

    def stats(self):
        comp = sum(s["comp"] for s in self.segments)
        seg_chars = sum(s["chars"] for s in self.segments)
        return dict(hot_mb=round(self.hot_chars / 1e6, 3), disk_mb=round(comp / 1e6, 3),
                    segments=len(self.segments), lived_chars=self.hot_chars + seg_chars,
                    compression=round(seg_chars / max(1, comp), 1))

    def _load_index(self):
        if os.path.exists(self.index_path):
            try: return json.load(open(self.index_path))
            except Exception: return []
        return []

    def _save_index(self):
        try: json.dump(self.segments, open(self.index_path, "w"))
        except Exception: pass
