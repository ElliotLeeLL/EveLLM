import torch
import torch.nn as nn
from sympy.testing.pytest import tooslow


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x  * torch.rsqrt(mean + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)

# # Test code
# torch.set_printoptions(sci_mode=False)
# batch_example = torch.randn(2, 3, 5)
# ln = LayerNorm(batch_example.shape[-1])
# ln_out = ln(batch_example)
# mean_ln = ln_out.mean(dim=-1, keepdim=True)
# var_ln = ln_out.var(dim=-1, keepdim=True, unbiased=False)
# print(mean_ln)
# print(var_ln)

# torch.manual_seed(123)
# example_batch = torch.randn(2, 3, 4)
# rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
# rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)
# assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))