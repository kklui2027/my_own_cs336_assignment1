import torch
from torch import nn
from cs336_basics.model import  RMSNorm, multihead_self_attention, positionwise_feedforward, Embedding, Linear, softmax
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_modle = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.mha = multihead_self_attention(d_model, num_heads, theta,max_seq_len).to(device=device, dtype=dtype)
        self.ffn = positionwise_feedforward(d_model, d_ff).to(device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        ## 因果编码多头注意力层： 
        x = x + self.mha(self.ln1(x))
        ## FFN (SwiGLU)层
        x = x + self.ffn(self.ln2(x))

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.d_modle = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype))
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln = Linear(d_model, vocab_size,  device=device, dtype=dtype)

    def forward(self, tokens_id: torch.Tensor):
        x = self.emb(tokens_id)
        for layer in self.layers:
            x = layer(x)
        return self.ln(self.norm(x))  ## 返回legits