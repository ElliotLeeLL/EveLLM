import torch
import torch.nn as nn
from requests_toolbelt.multipart.encoder import total_len

from attention.transformer_block import TransformerBlock
from layers.layer_norm import RMSNormQwen, RMSNormGemma
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

# class EveLLMModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])
#         self.transformer_blocks = nn.ModuleList(
#             [
#                 TransformerBlock(config) for _ in range(config["n_layers"])
#             ]
#         )
#         self.final_norm = RMSNormQwen(config["emb_dim"])
#         self.out_head = nn.Linear(
#             config["emb_dim"],
#             config["vocab_size"],
#             bias=False,
#             dtype=config["dtype"]
#         )
#         if config["head_dim"] is None:
#             head_dim = config["emb_dim"] // config["n_heads"]
#         else:
#             head_dim = config["head_dim"]
#             cos, sin = compute_rope_params(
#             head_dim=head_dim,
#             theta_base=config["rope_base"],
#             context_drxrxdrfxxdrfxxecrdxcedyhry c xwelength=config["context_length"],
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
        assert config["layer_types"] is not None and len(config["layer_types"]) == config["n_layers"]

        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(config, attn_type) for attn_type in config["layer_types"]
            ]
        )

        self.final_norm = RMSNormGemma(config["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
            dtype=config["dtype"]
        )
        self.config = config
        self.current_pos = 0
        cos_local, sin_local = compute_rope_params(
            head_dim=config["head_dim"],
            theta_base=config["rope_local_base"],
            context_length=config["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=config["head_dim"],
            theta_base=config["rope_base"],
            context_length=config["context_length"],
            dtype=torch.float32,
        )

        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)
        self.config = config

    def _create_masks(
            self,
            cur_len,
            device,
            pos_start=0,
            pos_end=None
    ):
        if pos_end is None:
            pos_end = cur_len
        total_len = pos_end

        ones = torch.ones((total_len, total_len), dtype=torch.bool, device=device)

        mask_global_full = torch.triu(ones, diagonal=1)

        far_past_full = torch.triu(ones, diagonal=self.config["sliding_window"]).T

        mask_local_full = mask_global_full | far_past_full

        row_slice = slice(pos_start, pos_end)
        mask_global = mask_global_full[row_slice, :pos_end][None, None, :, :]
        mask_local = mask_local_full[row_slice, :pos_end][None, None, :, :]
        return mask_global, mask_local

    def forward(
            self,
            x,
            input_ids,
            cache=None
    ):
        b, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) * (self.config["emb_dim"] ** 0.5)

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + seq_len
            self.current_pos = pos_end
            mask_global, mask_local = self._create_masks(
                cur_len=seq_len, device=x.device, pos_start=pos_start, pos_end=seq_len
            )
        else:
            pos_start = 0
            mask_global, mask_local = self._create_masks(
                cur_len=seq_len, device=x.device, pos_start=0, pos_end=seq_len
            )
        for i, block in enumerate(self.transformer_blocks):
            blk_cache = cache.get(i) if cache is not None else None
            x, new_blk_cache = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
                start_pos=pos_start,
                cache=blk_cache,
            )
            if cache is not None:
                cache.update(i, new_blk_cache)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.config["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0