import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange, reduce, einsum


# class MyLayerList(nn.Module):

#     def __init__(self, modules):
#         super().__init__()
#         for idx, m in enumerate(modules):
#             self.add_module(str(idx), m)  # 这一步会把 m 存进 self._modules

#     def __iter__(self):
#         return iter(self._modules.values())

#     def __len__(self):
#         return len(self._modules)

#     def __getitem__(self, idx):
#         return (
#             list(self._modules.values())[idx]
#             if isinstance(idx, slice)
#             else self._modules[str(idx)]
#         )

class MyLayerList(nn.ModuleList):
    """
    Drop-in replacement for nn.ModuleList with optional extra utilities.
    All speed / registration benefits of nn.ModuleList are preserved.
    """
    def __init__(self, modules=()):
        super().__init__(modules)

    def forward_sequential(
        self,
        x: Float[Tensor, "..."],
        *args,
        **kwargs,
    ) -> Float[Tensor, "..."]:
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x