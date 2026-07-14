"""
brain — the living brain: a growable SPIKING neural net faithful to the field guide
"The Learning Brain as a system of equations", born from a frozen teacher and then living —
thinking, learning from Claude + the web + registered tools, on an autonomous wake/sleep
rhythm, growing over a lifetime.

Entry points live at the repo root: `dashboard.py` (the unified web board + HTTP API),
`run_life.py` (headless), `tui.py` (terminal UI). The whole life is `brain.life.BrainLife`.
"""
from .ops import sg

__all__ = ["sg"]
