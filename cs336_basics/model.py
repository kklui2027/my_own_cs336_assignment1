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

class multihead_self_attention(nn.Module):
    def __init__(self,  d_model: int, num_heads: int, theta:int | None = None, max_seq_len: int | None = None,):
        # d_k = d_v = d_model / h
        super().__init__()
        self.h = num_heads
        self.d_t = d_model // num_heads
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))
        std = 1 / math.sqrt(d_model)
        nn.init.trunc_normal_(self.Wq,mean=0.0,std=std,a=-3 * std,b=3 * std,)
        nn.init.trunc_normal_(self.Wk,mean=0.0,std=std,a=-3 * std,b=3 * std,)
        nn.init.trunc_normal_(self.Wv,mean=0.0,std=std,a=-3 * std,b=3 * std,)
        nn.init.trunc_normal_(self.Wo,mean=0.0,std=std,a=-3 * std,b=3 * std,)
        self.rpe = RotaryPositionalEmbedding(theta, self.d_t, max_seq_len) if theta is not None and max_seq_len is not None else None


    def forward(self, in_features, token_positions=None):
        ## Q, K, V [batch, seq_len, h * d_k]
        ## 按头拆分
        batch, seq_len, d_m = in_features.shape[0], in_features.shape[1], in_features.shape[2]
        Q = in_features @ self.Wq.T
        K = in_features @ self.Wk.T
        V = in_features @ self.Wv.T

        ## Q, K, V - > [batch, h, seq_len, d_k(d_v)]
        Q = Q.reshape(batch, seq_len, self.h, self.d_t).transpose(1, 2)
        K = K.reshape(batch, seq_len, self.h, self.d_t).transpose(1, 2)
        V = V.reshape(batch, seq_len, self.h, self.d_t).transpose(1, 2) 

        ## 加位置编码
        if self.rpe is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=in_features.device)
            if token_positions.ndim > 1:
                token_positions = token_positions.unsqueeze(-2)
            Q = self.rpe(Q, token_positions)
            K = self.rpe(K, token_positions)

        ## 缩放点积注意力 Causual masking  
        # Query i 只能 attend 满足 j <= i 的 Key j，
        # 因而最终只能读取当前位置及之前位置对应的 Value
        positions = torch.arange(seq_len, device=in_features.device)
        mask = positions[None, :] <= positions[:, None]
        res = scaled_dot_product_attention(Q, K, V, mask)

        res = res.transpose(1, 2).reshape(batch, seq_len, d_m)
        
        return res @ self.Wo.T
    