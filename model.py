import torch
import torch.nn as nn
from attention.transformer_block import TransformerBlock
from layers.layer_norm import LayerNorm


class EveLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_embedding(in_idx)
        position_embeds = self.position_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embeds + position_embeds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits