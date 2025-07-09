import torch, math, torch.nn as nn
import os, sys
from pathlib import Path
from typing import IO, Any, BinaryIO
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce
import numpy.typing as npt
# import einsum

# from .softmax import softmax

def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    if dim < 0:
        dim += x.ndim

    in_type = x.dtype
    x = x.to(torch.float64)
    
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_moved = x.permute(*perm)

    x_max = reduce(x_moved, "... n -> ... 1", "max")
    x_exp = (x_moved - x_max).exp()

    x_enom = reduce(x_exp, "... n -> ... 1", "sum")

    out_moved = x_exp / x_enom

    out = out_moved.permute(*perm)
    
    return out.to(in_type)

def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size_vocab_size"],
    targets: Int[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
    assert (
        inputs.shape[-2] == targets.shape[-1]
    ), f"inputs.shape[-2] {inputs.shape[-2]} != targets.shape[-1] {targets.shape[-1]}"
    log_probs = inputs.float()    
    log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)

    batch_idx = torch.arange(inputs.size(0), device=inputs.device)
    loss = -log_probs[batch_idx, targets].mean()
    
    return loss.to(inputs.dtype)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[npt.NDArray, npt.NDArray]:
    B, T = batch_size, context_length
    data_t = torch.as_tensor(dataset, dtype=torch.long, device=device)
    N = data_t.numel()
    
    # starts = torch.randint(0, N - T, (B,), device=device)
    starts = torch.randperm(N - T, device=device)[:B]  # 无放回采样
    offsets   = rearrange(torch.arange(T + 1, device=device), 'n -> 1 n')  # [1, T+1]
    positions = rearrange(starts, 'b -> b 1') + offsets          
    tokens = data_t[positions]          # [B, T+1]
    x, y   = tokens[:, :-1], tokens[:, 1:]   # Next token prediction [B, T]
    return x, y
    
    
class EpochSampler:
    def __init__(self, num_positions: int, device: torch.device):
        self.N = num_positions            
        self.device = device
        self._shuffle()                   

    def _shuffle(self):
        self.perm = torch.randperm(self.N, device=self.device)
        self.cursor = 0                   

    def next(self, k: int) -> torch.Tensor:
        if self.cursor + k > self.N: 
            self._shuffle()
        idx = self.perm[self.cursor : self.cursor + k]
        self.cursor += k
        return idx

def get_batch_without_same(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    sampler: EpochSampler,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = batch_size, context_length
    data_t = torch.as_tensor(dataset, dtype=torch.long, device=device)   # [N_total]
    N = data_t.numel()

    starts = sampler.next(B)                    # shape (B,)

    # offsets: [1, T+1]，数值 0‥T
    offsets = torch.arange(T + 1, device=device).unsqueeze(0)            # (1, T+1)
    # positions: broadcast → (B, T+1)
    positions = starts.unsqueeze(1) + offsets

    tokens = data_t[positions]                  # (B, T+1)
    x, y = tokens[:, :-1], tokens[:, 1:]        # (B, T)

    return x, y


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    try:
        device = next(model.parameters()).device
    except StopIteration:             
        device = torch.device("cpu")

    if isinstance(src, (str, os.PathLike, Path)):
        ckpt = torch.load(src, map_location=device)   
    else:
        ckpt = torch.load(src, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["iteration"])

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    ckpt : dict[str, object] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    if isinstance(out, (str, os.PathLike, Path)):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out) 
    else: 
        torch.save(ckpt, out)



