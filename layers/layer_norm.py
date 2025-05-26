import torch
import torch.nn as nn


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

# # Test code
# torch.set_printoptions(sci_mode=False)
# batch_example = torch.randn(2, 3, 5)
# ln = LayerNorm(batch_example.shape[-1])
# ln_out = ln(batch_example)
# mean_ln = ln_out.mean(dim=-1, keepdim=True)
# var_ln = ln_out.var(dim=-1, keepdim=True, unbiased=False)
# print(mean_ln)
# print(var_ln)