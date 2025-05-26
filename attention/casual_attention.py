import torch
import torch.nn as nn


class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
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
# batch = torch.stack((inputs, inputs), dim=0)
#
# model = CasualAttention(
#     d_in=3,
#     d_out=2,
#     context_length=batch.shape[1],
#     dropout=0.0
# )
#
# res = model(batch)
# print(res)