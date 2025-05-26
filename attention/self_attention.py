import torch
import torch.nn as nn


# class SelfAttention(nn.Module):
#     def __init__(self, d_in, d_out):
#         super().__init__()
#         self.d_in = d_in
#         self.d_out = d_out
#         self.W_query = nn.Parameter(torch.rand(d_in, d_out))
#         self.W_key = nn.Parameter(torch.rand(d_in, d_out))
#         self.W_value = nn.Parameter(torch.rand(d_in, d_out))

#     def forward(self, x):
#         queries = x @ self.W_query
#         keys = x @ self.W_key
#         values = x @ self.W_value
#         attn_scores = queries @ keys.T
#         att_weights = torch.softmax(
#             attn_scores / keys.shape[-1] ** 0.5, dim=-1
#         )
#         context_vec = att_weights @ values
#         return context_vec

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        att_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = att_weights @ values
        return context_vec

# # Test code
# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],
#     [0.55, 0.87, 0.66],
#     [0.57, 0.85, 0.64],
#     [0.22, 0.58, 0.33],
#     [0.77, 0.25, 0.10],
#     [0.05, 0.80, 0.55]
# ])
#
# model = SelfAttention(3, 2)
# print(model(inputs))