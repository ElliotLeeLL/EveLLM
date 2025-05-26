import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.dropout = dropout
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose maxtrixes
        # from -> (batch_size, num_tokens, num_heads, head_dim)
        # to -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)

        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.output(context_vec)

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
# model = MultiHeadAttention(d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2)
# output = model(batch)
#
# print(output)
