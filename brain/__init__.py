"""
brain — the living brain: a growable SPIKING neural net faithful to the field guide
"The Learning Brain as a system of equations", born from a frozen teacher and then living —
thinking, learning from Claude + the web + registered tools, on an autonomous wake/sleep
rhythm, growing over a lifetime.

Entry points live at the repo root: `dashboard.py` (the unified web board + HTTP API),
`run_life.py` (headless), `tui.py` (terminal UI). The whole life is `brain.life.BrainLife`.
"""
from .device import select_gpu, get_device, device_report, set_perf_flags
from .ops import sg, sign_ste, spike_ste, prune_mask_ste, ternary_ste

__all__ = [
    "select_gpu", "get_device", "device_report", "set_perf_flags",
    "sg", "sign_ste", "spike_ste", "prune_mask_ste", "ternary_ste",
]
