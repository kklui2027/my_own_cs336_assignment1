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
                state["step"] += 1
                t = state["step"]

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


def learning_rate_schedule(    
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    ):
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
        return lr
    if warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))/ 2 * (max_learning_rate - min_learning_rate)
        return lr
    if it > cosine_cycle_iters:
        return min_learning_rate
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]

    total_norm = math.sqrt( sum(grad.square().sum() for grad in grads) )

    if total_norm > max_l2_norm:
        epsilon = 1e-6
        scale = max_l2_norm / (total_norm + epsilon)
        for grad in grads:

            grad.mul_(scale)
import numpy.typing as npt
import numpy as np
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    ## 随机生成 0 ~ n - m 之间的索引
    ranidx = np.random.randint(0, len(dataset) - context_length, batch_size)

    ## 利用广播和高级索引直接得到对应形状的下标数组
    offset = np.arange(context_length)
    ranidx = ranidx[:,None] + offset

    inputs = dataset[ranidx]
    labels = dataset[ranidx + 1]

    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(labels, dtype=torch.long, device=device)
    )

def save_checkpoint(model, optimizer, iteration, out):
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(check_point, out)

def load_checkpoint(src, model, optimizer):
    check_point = torch.load(src)

    model.load_state_dict(check_point["model"])
    optimizer.load_state_dict(check_point["optimizer"])

    return check_point["iteration"]

# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e3)

# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.

#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.

#     print(loss.cpu().item())

