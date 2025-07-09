"""
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) 
    Construct an embedding module. This function should accept the following parameters:
    num_embeddings: int Size of the vocabulary
    embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
def forward(self, token_ids: torch.Tensor) -> torch.Tensor 
    Lookup the embedding vectors for the given token IDs.
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from einops import rearrange, reduce, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)
        
    def forward(self, token_ids):
        return self.weight[token_ids]
