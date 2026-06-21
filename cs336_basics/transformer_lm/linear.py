from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):
    """A linear transformation y = x W^T without a bias term.

    The weight is stored as (d_out, d_in) so that the parameter layout matches
    the reference state dicts used in the tests.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Truncated normal init with variance 2 / (d_in + d_out), clipped at +/-3 sigma.
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
