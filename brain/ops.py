"""
ops.py — custom derivatives and gradient skipping (straight-through / pass-through).

The field guide is built on operations that are non-differentiable or that
deliberately CUT the gradient. Those are exactly the places autograd cannot help
us, so we supply the derivatives by hand:

  sg[x]         stop-gradient / the "gradient cut"      §0 notation, §6, §13.2(b)
  sign_ste      Hopfield recall  s <- sign(W s)          §4  (non-differentiable)
  spike_ste     theta-phase binary spike code            §7.4 (threshold)
  prune_ste     use-it-or-lose-it structural mask        §8.5, §10.3 (threshold)
  ternary_ste   signed sparse code (DG separation aid)   §4  (sparsify)

Every Function is forward-exact and backward-defined so the ops can sit inside a
normally-backpropagated graph (e.g. the router head, or the backprop baseline we
compare predictive coding against) without breaking autograd. The predictive-
coding core itself runs under no_grad and uses these as plain forward maps.

Design of the straight-through gradient: for a hard nonlinearity h(x) whose ideal
"soft" surrogate is s(x), the STE replaces dh/dx by ds/dx. We use the standard
clipped identity surrogate (hardtanh): pass the gradient where |x| <= 1 and kill
it outside, which prevents the runaway that a raw identity STE causes when
pre-activations drift far from the threshold.
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


# --------------------------------------------------------------------------- #
#  §4  Hopfield recall  s = sign(z)   —  straight-through (clipped identity)
# --------------------------------------------------------------------------- #
class _SignSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # sign with a deterministic tie-break to +1 (Hopfield states are +/-1)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1.0).to(g.dtype)  # hardtanh surrogate


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """{-1,+1} sign for attractor recall, differentiable via a clipped STE."""
    return _SignSTE.apply(x)


# --------------------------------------------------------------------------- #
#  §7.4  theta-phase spike code:  binary {0,1} threshold — straight-through
# --------------------------------------------------------------------------- #
class _SpikeSTE(Function):
    @staticmethod
    def forward(ctx, x, thr):
        ctx.save_for_backward(x)
        ctx.thr = thr
        return (x > thr).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        # surrogate derivative of a spike: a triangular bump around threshold
        return g * (x - ctx.thr).abs().le(1.0).to(g.dtype), None


def spike_ste(x: torch.Tensor, thr: float = 0.0) -> torch.Tensor:
    """Binary spike/encode (theta phase). Forward hard threshold, STE backward."""
    return _SpikeSTE.apply(x, thr)


# --------------------------------------------------------------------------- #
#  §8.5 / §10.3  use-it-or-lose-it pruning mask  1[keep] — straight-through
# --------------------------------------------------------------------------- #
class _PruneMaskSTE(Function):
    """Forward: hard survival mask 1[|w| > w_c]. Backward: identity on the weight.

    Lets a magnitude-pruned weight still receive a learning signal for its
    magnitude (so a just-pruned synapse can be regrown), matching the graded
    survival law P_keep(w)=1-exp(-|w|/w_c) whose hard limit this is.
    """
    @staticmethod
    def forward(ctx, w, w_c):
        return (w.abs() > w_c).to(w.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, None


def prune_mask_ste(w: torch.Tensor, w_c: float) -> torch.Tensor:
    return _PruneMaskSTE.apply(w, w_c)


# --------------------------------------------------------------------------- #
#  §4 pattern separation aid: signed ternary sparsify — straight-through
# --------------------------------------------------------------------------- #
class _TernarySTE(Function):
    @staticmethod
    def forward(ctx, x, thr):
        ctx.save_for_backward(x)
        ctx.thr = thr
        out = torch.zeros_like(x)
        out[x > thr] = 1.0
        out[x < -thr] = -1.0
        return out

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1.0).to(g.dtype), None


def ternary_ste(x: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return _TernarySTE.apply(x, thr)


__all__ = ["sg", "sign_ste", "spike_ste", "prune_mask_ste", "ternary_ste"]
