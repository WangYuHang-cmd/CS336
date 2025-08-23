# adapters.py（只贴关键差异）
from __future__ import annotations
import math
from typing import Type
import torch
from cs336_systems.FlashAttention import (
    FlashAttentionPytorch as _FA_PT,
    FlashAttentionTriton as _FA_TRITON,
)

def _compute_L_for_test(q: torch.Tensor, k: torch.Tensor, is_causal: bool) -> torch.Tensor:
    d = q.shape[-1]
    S = torch.einsum("... q d, ... k d -> ... q k", q, k) * (1.0 / math.sqrt(d))
    if is_causal:
        Lq, Lk = q.shape[-2], k.shape[-2]
        mask = (torch.arange(Lq, device=q.device)[:, None] >= torch.arange(Lk, device=q.device)[None, :])
        S = S.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(S, dim=-1)  # (B, L)

class _FA_PyTorch_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal: bool = False):
        q4, k4, v4 = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        o4 = _FA_PT.apply(q4, k4, v4, is_causal)
        o = o4.squeeze(1)
        L = _compute_L_for_test(q, k, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, L)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, _L = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        with torch.enable_grad():
            q1 = q.detach().requires_grad_(True)
            k1 = k.detach().requires_grad_(True)
            v1 = v.detach().requires_grad_(True)
            o1 = _FA_PT.apply(q1.unsqueeze(1), k1.unsqueeze(1), v1.unsqueeze(1), ctx.is_causal).squeeze(1)
            loss = (o1 * grad_o).sum()
            loss.backward()
            dq, dk, dv = q1.grad, k1.grad, v1.grad
        return dq, dk, dv, None

class _FA_Triton_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal: bool = False):
        q4, k4, v4 = q.unsqueeze(1).contiguous(), k.unsqueeze(1).contiguous(), v.unsqueeze(1).contiguous()
        o4 = _FA_TRITON.apply(q4, k4, v4, is_causal)
        o = o4.squeeze(1)
        L = _compute_L_for_test(q, k, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, L)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, _L = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        with torch.enable_grad():
            q1 = q.detach().requires_grad_(True)
            k1 = k.detach().requires_grad_(True)
            v1 = v.detach().requires_grad_(True)
            o1 = _FA_TRITON.apply(q1.unsqueeze(1), k1.unsqueeze(1), v1.unsqueeze(1), ctx.is_causal).squeeze(1)
            (o1 * grad_o).sum().backward()
            dq, dk, dv = q1.grad, k1.grad, v1.grad
        return dq, dk, dv, None

def get_flashattention_autograd_function_pytorch() -> Type:
    return _FA_PyTorch_Adapter

def get_flashattention_autograd_function_triton() -> Type:
    return _FA_Triton_Adapter



def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
