"""
ops.py — the stop-gradient ("gradient cut") used across the field guide.

  sg[x]   stop-gradient / the "gradient cut"   §0 notation, §6, §13.2(b)

Forward-exact, backward-defined, so it can sit inside a normally-backpropagated graph (e.g. the router
head, or the backprop baseline the predictive-coding / e-prop core is compared against) without breaking
autograd. The spiking core itself runs under no_grad and computes its surrogate spike derivatives inline in
`spiking.py` (the earlier straight-through primitives sign_ste/spike_ste/prune_mask_ste/ternary_ste were
never wired into the live path and were removed).
"""
from __future__ import annotations
import torch
from torch.autograd import Function


# --------------------------------------------------------------------------- #
#  §6 / §13.2(b)  the gradient cut:  forward identity, zero derivative
# --------------------------------------------------------------------------- #
class _StopGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return None  # dsg[x]/dx = 0  -> no gradient crosses this edge


def sg(x: torch.Tensor) -> torch.Tensor:
    """Stop-gradient. Activations pass forward; gradients do not (the module cut)."""
    return _StopGradient.apply(x)


__all__ = ["sg"]
