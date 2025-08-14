import torch
import torch.nn as nn
from utils.model_utils import compute_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.group_size = num_heads // num_kv_groups
        self.num_kv_groups = num_kv_groups

        if head_dim is not None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        # self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.output = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask=None, cos=None, sin=None):
        batch_size, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose maxtrixes
        # from -> (batch_size, num_tokens, num_heads, head_dim)
        # to -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        keys = compute_rope(keys, cos, sin)
        queries = compute_rope(queries, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)

        return self.output(context)