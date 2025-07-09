import torch, math, torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce


class MyBatch(nn.Module):
    def __init__(
        self, batch_size: int, seq_len: int, d_model: int, device=None, dtype=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

    def get_batch(self) -> Float[Tensor, "batch seq_len d_model"]:
        """
        Returns a batch of random data with the specified shape.
        """
        return torch.randn(
            (self.batch_size, self.seq_len, self.d_model),
            device=self.device,
            dtype=self.dtype,
        )
