import torch
import torch.nn as nn

from layers.feed_forward import GatedFeedForward
from layers.layer_norm import RMSNormQwen
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
    def __init__(self, config):
        super().__init__()
        self.attention_layer = GroupedQueryAttention(
            d_in=config["emb_dim"],
            num_heads=config["n_heads"],
            head_dim=config["head_dim"],
            num_kv_groups=config["n_kv_groups"],
            qk_norm=config["qk_norm"],
            dtype=config["dtype"],
        )
        self.ff = GatedFeedForward(config)
        self.norm1 = RMSNormQwen(config["emb_dim"], eps=1e-6)
        self.norm2 = RMSNormQwen(config["emb_dim"], eps=1e-6)

    def forward(self, x, mask=None, cos=None, sin=None):
        shotcut = x
        x = self.norm1(x)
        x = self.attention_layer(
            x, mask=mask, cos=cos, sin=sin
        )
        x = shotcut + x

        shotcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = shotcut + x
        return x