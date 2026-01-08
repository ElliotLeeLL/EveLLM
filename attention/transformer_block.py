import torch
import torch.nn as nn

from layers.feed_forward import GatedFeedForward
from layers.layer_norm import RMSNormQwen, RMSNormGemma
from attention.grouped_query_attention import GroupedQueryAttention


# class TransformerBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention_layer = MultiHeadAttention(
#             d_in=config["emb_dim"],
#             d_out=config["emb_dim"],
#             context_length=config["context_length"],
#             dropout=config["drop_rate"],
#             num_heads=config["n_heads"],
#             qkv_bias=config["qkv_bias"],
#         )
#         self.ff = FeedForward(config)
#         self.norm1 = LayerNorm(config["emb_dim"])
#         self.norm2 = LayerNorm(config["emb_dim"])
#         self.dropout = nn.Dropout(config["drop_rate"])
#
#     def forward(self, x):
#         shotcut = x
#         x = self.norm1(x)
#         x = self.attention_layer(x)
#         x = self.dropout(x)
#         x = shotcut + x
#
#         shotcut = x
#         x = self.norm2(x)
#         x = self.ff(x)
#         x = self.dropout(x)
#         x = shotcut + x
#         return x

# class TransformerBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention_layer = MultiHeadAttention(
#             d_in=config["emb_dim"],
#             d_out=config["emb_dim"],
#             context_length=config["context_length"],
#             dropout=config["drop_rate"],
#             num_heads=config["n_heads"],
#             qkv_bias=config["qkv_bias"],
#         )
#         self.ff = GatedFeedForward(config)
#         self.norm1 = RMSNorm(config["emb_dim"])
#         self.norm2 = RMSNorm(config["emb_dim"])
#
#     def forward(self, x):
#         shotcut = x
#         x = self.norm1(x)
#         x = self.attention_layer(x)
#         x = shotcut + x
#
#         shotcut = x
#         x = self.norm2(x)
#         x = self.ff(x)
#         x = shotcut + x
#         return x

# class TransformerBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention_layer = GroupedQueryAttention(
#             d_in=config["emb_dim"],
#             d_out=config["emb_dim"],
#             num_heads=config["n_heads"],
#             num_kv_groups=config["n_kv_groups"],
#             dtype=config["dtype"],
#         )
#         self.ff = GatedFeedForward(config)
#         self.norm1 = RMSNormQwen(config["emb_dim"])
#         self.norm2 = RMSNormQwen(config["emb_dim"])
#
#     def forward(self, x, mask=None, cos=None, sin=None):
#         shotcut = x
#         x = self.norm1(x)
#         x = self.attention_layer(
#             x.to(torch.bfloat16), mask=mask, cos=cos, sin=sin
#         )
#         x = shotcut + x
#
#         shotcut = x
#         x = self.norm2(x)
#         x = self.ff(x)
#         x = shotcut + x
#         return x

class TransformerBlock(nn.Module):
    def __init__(self, config, attn_type):
        super().__init__()
        self.config = config
        self.attn_type = attn_type
        self.sliding_window = config["sliding_window"]

        self.attention_layer = GroupedQueryAttention(
            d_in=config["emb_dim"],
            num_heads=config["n_heads"],
            head_dim=config["head_dim"],
            num_kv_groups=config["n_kv_groups"],
            qk_norm=config["qk_norm"],
            query_pre_attn_scalar=config["query_pre_attn_scalar"],
            dtype=config["dtype"],
        )
        self.ff = GatedFeedForward(config)
        self.input_layernorm = RMSNormGemma(config["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNormGemma(config["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNormGemma(config["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNormGemma(config["emb_dim"], eps=1e-6)

    def forward(
            self,
            x,
            mask_global,
            mask_local,
            cos_global,
            sin_global,
            cos_local,
            sin_local,
            start_pos=0,
            cache=None
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            if cache is not None and isinstance(cache, tuple):
                prev_k, _ = cache
                eff_kv_len = prev_k.size(2) + x.size(1)
            else:
                eff_kv_len = x.size(1)
            # Take the last `eff_kv_len` columns so mask width equals K length
            attn_mask = mask_local[..., -eff_kv_len:]
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn, next_cache = self.attention_layer(x, attn_mask, cos, sin, start_pos=start_pos, cache=cache)
        if next_cache is not None and self.attn_type == "sliding_attention":
            k, v = next_cache
            if k.size(2) > self.sliding_window:
                k = k[:, :, -self.sliding_window:, :]
                v = v[:, :, -self.sliding_window:, :]
            next_cache = (k, v)

        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x, next_cache