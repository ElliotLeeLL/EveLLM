from functools import cache

import torch
import torch.nn as nn

from layers.layer_norm import RMSNormQwen, RMSNormGemma
from utils.model_utils import compute_rope


# class GroupedQueryAttention(nn.Module):
#     def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
#         super().__init__()
#         assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
#
#         self.num_heads = num_heads
#         self.num_kv_groups = num_kv_groups
#         self.group_size = num_heads // num_kv_groups
#
#         if head_dim is None:
#             assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
#             head_dim = d_in // num_heads
#
#         self.head_dim = head_dim
#         self.d_out = num_heads * head_dim
#
#         self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
#         self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
#         self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
#
#         self.output = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
#
#         if qk_norm:
#             self.q_norm = RMSNormQwen(head_dim, eps=1e-6)
#             self.k_norm = RMSNormQwen(head_dim, eps=1e-6)
#         else:
#             self.q_norm = self.k_norm = None
#
#     def forward(self, x, mask=None, cos=None, sin=None):
#         batch_size, num_tokens, _ = x.shape
#
#         queries = self.W_query(x)
#         keys = self.W_key(x)
#         values = self.W_value(x)
#
#         queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
#         keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
#         values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
#
#         if self.q_norm:
#             queries = self.q_norm(queries)
#         if self.k_norm:
#             keys = self.k_norm(keys)
#
#         queries = compute_rope(queries, cos, sin)
#         keys = compute_rope(keys, cos, sin)
#
#         keys = keys.repeat_interleave(self.group_size, dim=1)
#         values = values.repeat_interleave(self.group_size, dim=1)
#
#         attn_scores = queries @ keys.transpose(2, 3)
#         attn_scores = attn_scores.masked_fill(mask, -torch.inf)
#         attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
#
#         context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
#         return self.output(context)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None,dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)

        self.output = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNormGemma(head_dim, eps=1e-6)
            self.k_norm = RMSNormGemma(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar ** -0.5
        else:
            self.scaling = head_dim ** -0.5

    def forward(self, x, mask=None, cos=None, sin=None, start_pos=0, cache=None):
        batch_size, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys_new = self.W_key(x)
        values_new = self.W_value(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys_new.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values_new.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Load cache before applying rope
        prev_len = 0
        if cache is not None:
            prev_k, prev_v = cache
            if prev_k is not None:
                prev_len = prev_k.size(2)
                keys_cat_raw = torch.cat([prev_k, keys_new], dim=2)
                values_cat_raw = torch.cat([prev_v, values_new], dim=2)
            else:
                keys_cat_raw = keys_new
                values_cat_raw = values_new
        else:
            keys_cat_raw = keys_new
            values_cat_raw = values_new

        queries = compute_rope(queries, cos, sin, offset=start_pos)
        keys = compute_rope(keys_cat_raw, cos, sin, offset=start_pos - prev_len)

        queries = queries * self.scaling

        # Update cache
        if cache is not None and cache[0] is not None:
            next_cache = (
                torch.cat([cache[0], keys_new], dim=2),
                torch.cat([cache[1], values_new], dim=2)
            )
        else:
            next_cache = (keys_new, values_new)


        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values_cat_raw.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        # attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        out = self.output(context)

        return out, next_cache