import torch

def cross_entropy(logits : torch.Tensor, targets: torch.Tensor):
    ## 全部flatten，方便计算
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)

    maxm = logits.max(dim=-1, keepdim=True).values
    logits = logits - maxm

    rows = torch.arange(logits.shape[0], device=logits.device)
    target_logits = logits[rows, targets]

    log_normalizer = torch.log(torch.exp(logits).sum(dim=1))
    logits = torch.log(torch.exp(logits).sum(dim = 1))
    return (log_normalizer - target_logits).mean()

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss


import math
from typing import Optional, Callable
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                # 初始化该参数的状态
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # 更新迭代次数，t 从 1 开始
                t = ++ state["step"]

                # 计算偏差修正后的学习率
                alpha_t = ( lr * math.sqrt(1 - beta2 ** t ) / (1 - beta1 ** t))

                # Decoupled Weight Decay
                p.mul_(1 - lr * weight_decay)

                # 更新第一矩
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新第二矩
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Adam 参数更新
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

        return loss













    
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e3)

# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.

#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.

#     print(loss.cpu().item())

