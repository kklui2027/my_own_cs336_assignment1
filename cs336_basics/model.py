import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(
                    self.vocab,
                    mean=0.0,
                    std=1,
                    a=-3,
                    b=3,
                )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.vocab[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        
        mean_square = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)

        return (normalized * self.g).to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dff: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(dff, d_model))
        self.W3 = nn.Parameter(torch.empty(dff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, dff))
        std = math.sqrt(2.0 / (d_model + dff))
        nn.init.trunc_normal_(self.W1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.W2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.W3, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        silu = x @ self.W1.T
        t = x @ self.W3.T
        return (silu * torch.sigmoid(silu) * t) @ self.W2.T

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even number")

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float64)
        dimensions = torch.arange(0, d_k, 2, device=device, dtype=torch.float64)
        angles = positions[:, None] / (theta ** (dimensions[None, :] / d_k))

        self.d_k = d_k
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected last dimension {self.d_k}, got {x.shape[-1]}")

        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        cos = self.cos_cached[token_positions].to(dtype=compute_dtype)
        sin = self.sin_cached[token_positions].to(dtype=compute_dtype)

        even = x[..., 0::2].to(compute_dtype)
        odd = x[..., 1::2].to(compute_dtype)

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    maxm = x.max(dim = dim, keepdim = True).values
    x = x - maxm

    exp_x = torch.exp(x)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # Q: (..., n, d_k)
    # K: (..., m, d_k)
    # V: (..., m, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores)
    return weights @ V
