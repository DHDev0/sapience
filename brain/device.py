"""
device.py — pin the whole system to the SECOND GPU and keep it there.

The user's machine has two AMD Radeon RX 7900 XTX (gfx1100) exposed through
ROCm as `cuda` devices. We use the *second* physical GPU only, and we refuse
to silently fall back to the CPU (the user explicitly wants GPU/VRAM only, with
near-zero CPU/RAM traffic).

Usage contract
--------------
`select_gpu(1)` MUST run before `import torch` for the visible-device mask to
take effect. The entrypoints in this package set the environment variable on
their very first line; `select_gpu` here is idempotent and will only warn if
torch is already imported.
"""
from __future__ import annotations
import os
import sys
import warnings

_GPU_ENV_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def select_gpu(index: int = 1) -> None:
    """Make only physical GPU `index` visible. Call before importing torch."""
    val = str(index)
    if "torch" in sys.modules:
        # Too late to change the mask for an already-initialised runtime.
        cur = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
        if cur != val:
            warnings.warn(
                f"select_gpu({index}) called after torch import; visible-device "
                f"mask is fixed at '{cur}'. Set HIP_VISIBLE_DEVICES={val} before "
                f"launching python to guarantee the second GPU.",
                RuntimeWarning,
            )
        return
    for var in _GPU_ENV_VARS:
        os.environ[var] = val


def get_device(require_gpu: bool = True):
    """Return the torch device. Raises if no GPU and require_gpu (default)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    msg = (
        "No ROCm/HIP GPU is available to this process. The 7900 XTX HSA runtime "
        "was wedged (hsa_init -> HSA_STATUS_ERROR_OUT_OF_RESOURCES) in testing; "
        "reset it with:  sudo modprobe -r amdgpu && sudo modprobe amdgpu  "
        "(display is on the ASPEED chip, so this is safe), then re-run."
    )
    if require_gpu:
        raise RuntimeError(msg)
    warnings.warn(msg + "  Falling back to CPU (SLOW, against the GPU-only policy).", RuntimeWarning)
    return torch.device("cpu")


def device_report() -> str:
    import torch
    if not torch.cuda.is_available():
        return "device: CPU (no GPU visible)"
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    hip = getattr(torch.version, "hip", None)
    mask = os.environ.get("HIP_VISIBLE_DEVICES", "<unset>")
    return (f"device: cuda:{i} = {name} | {total:.1f} GB VRAM | ROCm/HIP {hip} | "
            f"HIP_VISIBLE_DEVICES={mask} (physical GPU {mask})")


def set_perf_flags() -> None:
    """Enable throughput-oriented backends; no correctness impact for our matmul core."""
    import torch
    torch.backends.cudnn.benchmark = True          # autotune MIOpen conv algos (if any conv is used)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True   # harmless on ROCm; helps where supported
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
