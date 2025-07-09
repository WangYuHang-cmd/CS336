import torch, math, torch.nn as nn
import os, sys
from pathlib import Path
from typing import IO, Any, BinaryIO
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce, repeat
import numpy.typing as npt
# import einsum
import torch.nn.functional as F 
from typing import Optional, Tuple
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



def get_eot_id(tokenizer, eot_text: str = "<|endoftext|>") -> int:
    eot_bytes =  eot_text.encode("utf-8")
    return tokenizer.stoi.get(eot_bytes, None)

@torch.no_grad()
def decode(
        model, 
        tokenizer, 
        ptompt:str|list[int],
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        device: str = "cuda",
        eot_ids: Optional[list[int]] = None
    ):
    if eot_id is None:
        eot_id = get_eot_id(tokenizer, "<|endoftext|>")
    
    model.eval() # no dropout, RMSNorm
    
    if isinstance(ptompt, str):
        prompt_ids: list[int] = tokenizer.encode(ptompt, add_eot=False)
    else:
        prompt_ids = list(ptompt)
        
    tokens: Int[torch.Tensor, "1 seq"] = torch.tensor(
        prompt_ids,  dtypoe = torch.long, device=device
    ).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        logits: Float[torch.Tensor, "1 seq vocab"] = model(tokens)
        
        last_logits: Float[Tensor, "1 vocab"] = rearrange(
            logits[:, -1, :], "b v -> b v"
        )
        
        probs: Float[Tensor, "1 vocab"] = F.softmax(last_logits / temperature, dim=-1)
        
        if temperature <= 0:
            next_token: Int[Tensor, "1 1"] = last_logit.argmax(dim=-1, keepdim=True) #choose the largest logit directly
        else:
            probs: Float[Tensor, "1 vocab"] = softmax(last_logits / temperature, dim=-1)
            
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf = torch.cumsum(sorted_p, dim=-1)   
                keep = cdf <= top_p
                keep[..., 0] = True  # Ensure at least one token is kept
                nucleus_p = torch.where(keep, sorted_p, 0.0)                        # 核外置 0
                nucleus_p = nucleus_p / nucleus_p.sum(dim=-1, keepdim=True)
                sampled_in_sorted = torch.multinomial(nucleus_p, num_samples=1)
                next_token = sorted_idx.gather(dim=-1, index=sampled_in_sorted)
            else:
                next_token: Int[Tensor, "1 1"] = torch.multinomial(probs, num_samples=1)
            tokens: Int[torch.Tensor, "1 new_seq"] = torch.cat([tokens, next_token], dim=-1)
            
            if eot_id is not None and next_token.item() == eot_id:
                break

    out_ids: list[int] = tokens[0].tolist()
    # 用你的 tokenizer.decode 把 token 序列转回可读文本
    out_txt: str = tokenizer.decode(out_ids)
    # 返回（token_id 序列, 文本）
    return out_ids, out_txt





# @torch.no_grad()
# def decode(model, tokenizer, prompt, *,
#            max_new_tokens: int = 100,
#            temperature: float = 1.0,
#            top_p: float | None = None,
#            device: str = "cuda") ->  tuple[list[int], str]:
#     ids : list[int] = (
#         tokenizer.encode(prompt, add_eot =  False)
#         if isinstance(prompt, str) 
#         else list(prompt)
#     )
    
#     # (batch=1, seq) -> Int["1 seq"]
#     tokens: Int[torch.Tensor, "1 seq"] = torch.as_tensor(
#         ids, 
#         dtype = torch.long, 
#         device = device
#     ).unsqueeze(0)
    
    
#     for _ in range(max_new_tokens):
#         logits: Float[torch.Tensor, "1 seq vocab"] = model(tokens)
        
#         last_logits: Float[torch.Tensor, "1 vocab"] = rearrange(
#             logits[:, -1, :], 
#             "b v -> b v"
#         )
        
#         probs: Float[torch.Tensor, "1 vocab"] = F.softmax(last_logits, dim=-1)
        
#         if top_p is not None and top_p >= 0.0 and top_p < 1.0:
#             sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
#             cdf = torch.cumsum(sorted_p, dim=-1)
#             keep = cdf <= top_p
#             keep[..., 0] = True  # Ensure at least one token is kept
            
#             nucleus_p = torch.where(keep, sorted_p, sorted_idx)
#             nucleus_p = nucleus_p / nucleus_p.sum(dim=-1, keepdim=True)
#             sample_idx = torch.multinomial(nucleus_p, num_samples=1)
#             next_token: Int[torch.Tensor, "1 1"] = sorted_idx.gather(-1, sample_idx)
#         else:
#             next_token: Int[torch.Tensor, "1 1"] = torch.multinomial(probs, num_samples=1)
        
#         tokens = torch.cat([tokens, next_token], dim=-1)  # Append to sequence
        
#         if next_token.item() == tokenizer.eot_token_id:
#             break
        
#         out_ids: list[int] = ids + [next_token.item()]
#         out_text: str = tokenizer.decode(ids)
#         return out_ids, out_text
            
        
    