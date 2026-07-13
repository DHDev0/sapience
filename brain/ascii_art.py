"""
ascii_art.py — render what the brain SEES as ASCII, so a human can watch its vision
in the terminal. The brain's retina is a small grayscale grid (senses.encode_image);
here we map those light-levels onto a character ramp.
"""
from __future__ import annotations

RAMP = " .:-=+*#%@"          # dark -> light


def image_to_ascii(img, width=64, height=28, invert=True):
    """PIL image -> ASCII string. Terminal cells are ~2:1 tall, so height ~ width/2.
    invert=True suits dark terminals (bright pixels -> dense glyphs)."""
    g = img.convert("L").resize((max(4, width), max(2, height)))
    px = list(g.getdata())
    ramp = RAMP[::-1] if invert else RAMP
    n = len(ramp)
    rows = []
    for r in range(height):
        row = px[r * width:(r + 1) * width]
        rows.append("".join(ramp[min(n - 1, v * n // 256)] for v in row))
    return "\n".join(rows)
