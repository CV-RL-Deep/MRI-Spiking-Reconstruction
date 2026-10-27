import math
import torch


class ArcTanSurrogate(torch.autograd.Function):
    """
    ArcTan surrogate gradient for non-differentiable Heaviside step firing.
    Forward:
        S = Theta(U - V_th) in {0.0, 1.0}
    Backward:
        dS/dU = alpha / (pi * (1 + alpha^2 * (U - V_th)^2))
    """

    @staticmethod
    def forward(ctx, u_minus_vth: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(u_minus_vth)
        ctx.alpha = alpha
        return (u_minus_vth >= 0.0).to(u_minus_vth.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (u_minus_vth,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_u = grad_output * (alpha / (math.pi * (1.0 + (alpha * u_minus_vth) ** 2)))
        return grad_u, None


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    FastSigmoid surrogate gradient function.
    Forward:
        S = Theta(U - V_th) in {0.0, 1.0}
    Backward:
        dS/dU = 1 / (1 + alpha * |U - V_th|)^2
    """

    @staticmethod
    def forward(ctx, u_minus_vth: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
        ctx.save_for_backward(u_minus_vth)
        ctx.alpha = alpha
        return (u_minus_vth >= 0.0).to(u_minus_vth.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (u_minus_vth,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_u = grad_output / ((1.0 + alpha * torch.abs(u_minus_vth)) ** 2)
        return grad_u, None


def surrogate_atan(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return ArcTanSurrogate.apply(x, alpha)


def surrogate_fast_sigmoid(x: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    return FastSigmoidSurrogate.apply(x, alpha)

