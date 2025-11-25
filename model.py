import torch
import torch.nn as nn
from attention.transformer_block import TransformerBlock
from layers.layer_norm import RMSNormQwen
from utils.model_utils import compute_rope_params


# class EveLLMModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
#         self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
#         self.dropout = nn.Dropout(config["drop_rate"])
#         self.transformer_blocks = nn.Sequential(
#             *[
#                 TransformerBlock(config)
#                 for _ in range(config["n_layers"])
#             ]
#         )
#         self.final_norm = LayerNorm(config["emb_dim"])
#         self.out_head = nn.Linear(
#             config["emb_dim"],
#             config["vocab_size"],
#             bias=False
#         )
#
#     def forward(self, in_idx):
#         batch_size, seq_len = in_idx.shape
#         token_embeds = self.token_embedding(in_idx)
#         position_embeds = self.position_embedding(
#             torch.arange(seq_len, device=in_idx.device)
#         )
#         x = token_embeds + position_embeds
#         x = self.dropout(x)
#         x = self.transformer_blocks(x)
#         x = self.final_norm(x)
#         logits = self.out_head(x)
#         return logits

# class EveLLMModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])
#         self.transformer_blocks = nn.Sequential(
#             *[
#                 TransformerBlock(config)
#                 for _ in range(config["n_layers"])
#             ]
#         )
#         self.final_norm = RMSNorm(config["emb_dim"])
#         self.out_head = nn.Linear(
#             config["emb_dim"],
#             config["vocab_size"],
#             bias=False,
#             dtype=config["dtype"]
#         )
#
#     def forward(self, in_idx):
#         token_embeds = self.token_embedding(in_idx)
#         x = token_embeds
#         x = self.transformer_blocks(x)
#         x = self.final_norm(x)
#         logits = self.out_head(x)
#         return logits

# class EveLLMModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])
#         self.transformer_blocks = nn.Sequential(
#             *[
#                 TransformerBlock(config)
#                 for _ in range(config["n_layers"])
#             ]
#         )
#         self.final_norm = RMSNorm(config["emb_dim"])
#         self.out_head = nn.Linear(
#             config["emb_dim"],
#             config["vocab_size"],
#             bias=False,
#             dtype=config["dtype"]
#         )
#         cos, sin = precompute_for_rope_params(
#             head_dim=config["emb_dim"] // config["n_heads"],
#             theta_base=config["rope_base"],
#             context_length=config["context_length"],
#             freq_config=config["rope_freq"],
#         )
#         self.register_buffer("cos", cos, persistent=False)
#         self.register_buffer("sin", sin, persistent=False)
#         self.config = config
#
#     def forward(self, in_idx):
#         token_embeds = self.token_embedding(in_idx)
#         x = token_embeds
#
#         num_tokens = x.shape[1]
#         mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
#
#         for transformer_block in self.transformer_blocks:
#             x = transformer_block(x, mask=mask, cos=self.cos, sin=self.sin)
#         x = self.final_norm(x)
#         logits = self.out_head(x.to(self.config["dtype"]))
#         return logits

class EveLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(config) for _ in range(config["n_layers"])
            ]
        )
        self.final_norm = RMSNormQwen(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
            dtype=config["dtype"]
        )
        if config["head_dim"] is None:
            head_dim = config["emb_dim"] // config["n_heads"]
        else:
            head_dim = config["head_dim"]
            cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=config["rope_base"],
            context_length=config["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.config = config

    def forward(self, in_idx):
        token_embeds = self.token_embedding(in_idx)
        x = token_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask, cos=self.cos, sin=self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.config["dtype"]))
        return logits