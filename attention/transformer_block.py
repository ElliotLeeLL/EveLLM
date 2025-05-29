import torch
import torch.nn as nn

from layers.feed_forward import FeedForward
from layers.layer_norm import LayerNorm
from attention.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_layer = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shotcut = x
        x = self.norm1(x)
        x = self.attention_layer(x)
        x = self.dropout(x)
        x = shotcut + x

        shotcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = shotcut + x
        return x