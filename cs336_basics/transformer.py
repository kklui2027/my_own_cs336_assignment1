import torch
from torch import nn
from cs336_basics.model import  RMSNorm, multihead_self_attention, positionwise_feedforward, Embedding, Linear, softmax

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, using_attenres=False, device=None, dtype=None):
        super().__init__()
        self.d_modle = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.using_attenres=using_attenres
        if using_attenres:
            self.w1 = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
            self.w2 = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.mha = multihead_self_attention(d_model, num_heads, theta, max_seq_len).to(device=device, dtype=dtype)
        self.ffn = positionwise_feedforward(d_model, d_ff).to(device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def res_norm(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        
        mean_square = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + 1e-5)

        return normalized.to(input_dtype)
    
    def attenres(self, w: torch.Tensor, history: list[torch.Tensor], dvice=None):
        history =torch.stack([st for st in history], dim=0) 
        q = self.res_norm(history)
        ## w补一个维度，走aten.mm，可以被统计到Floats运算量
        ## 把weights导出，可以查看对
        weights = softmax(q @ w.unsqueeze(-1), dim=0)
        return (weights * history).sum(dim=0)

    def forward(self, x: torch.Tensor, history: list[torch.Tensor] | None=None):
        if self.using_attenres:
            if history is None:
              raise ValueError("未传入history")
        
            history.append(self.mha(self.ln1(x)))
            x = self.attenres(self.w1, history)

            history.append(self.ffn(self.ln2(x)))
            x = self.attenres(self.w2, history)
            return x
        ## 因果编码多头注意力层：   
        x = x + self.mha(self.ln1(x))
        ## FFN (SwiGLU)层
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, using_attenres=False,device=None, dtype=None):
        super().__init__()
        self.d_modle = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.using_attenres=using_attenres
        self.emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, using_attenres, device=device, dtype=dtype))
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln = Linear(d_model, vocab_size,  device=device, dtype=dtype)

    def forward(self, tokens_id: torch.Tensor):
        x = self.emb(tokens_id)

        if self.using_attenres:
            history = [x]
            
            for layer in self.layers:
                x = layer(x, history)
            return self.ln(self.norm(x))  ## 返回legits

        
        for layer in self.layers:
            x = layer(x)
        return self.ln(self.norm(x))  ## 返回legits