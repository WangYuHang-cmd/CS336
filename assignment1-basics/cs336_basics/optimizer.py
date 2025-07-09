from collections.abc import Callable, Iterable
import torch, math, torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce
import torch.optim as optim
from typing import Optional
import math


class MySGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, weight_decay = (
                group["lr"],
                group["momentum"],
                group["weight_decay"],
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1

                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add_(p.data, alpha=weight_decay)

                # Apply momentum
                if momentum != 0:
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-lr)
        return loss


class MyAdamW(optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, betas, eps, weight_decay = (  # α, (β₁, β₂), ϵ, λ
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )

            for p in group["params"]:  # Iterate over parameters
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["step"] = 0

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                adjusted_lr = lr / (1 - betas[0] ** t) * math.sqrt(1 - betas[1] ** t)

                p.data.add_(m / (v.sqrt() + eps), alpha=-adjusted_lr)

                p.data.add_(p.data, alpha=-weight_decay * lr)

        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it > cosine_cycle_iters:
        return min_learning_rate

    if it < warmup_iters:
        return max_learning_rate / warmup_iters * it

    if warmup_iters <= it <= cosine_cycle_iters:
        cosine_param = math.cos(
            math.pi / (cosine_cycle_iters - warmup_iters) * (it - warmup_iters)
        )
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + cosine_param
        )

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return

    total_norm_sq = torch.zeros(1, device=params[0].grad.device)
    for p in params:
        total_norm_sq += p.grad.pow(2).sum()

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            p.grad.mul_(scale)

# def gradient_clipping(
#     parameters: Iterable[torch.nn.Parameter], 
#     max_l2_norm: float
# ) -> None:
#     eps = 1e-6
#     grad_tensor = torch.cat([p.grad.view(-1) for p in parameters if p is not None and p.grad is not None])
#     if not grad_tensor.numel():
#         return  
#     grad_norm2 = grad_tensor.norm(2)
#     if grad_norm2 <= max_l2_norm:
#         return
#     scale = max_l2_norm / (grad_norm2 + eps)
#     for p in parameters:
#         if p is not None and p.grad is not None:
#             p.grad.mul_(scale)



