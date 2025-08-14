import torch

config = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model_configs = {
    "eve-llm-124M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "model_name": "eve_llm_124M",
    },
    "eve-llm-355M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "model_name": "eve_llm_355M",
    },
    "eve-llm-774M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "model_name": "eve_llm_774M",
    },
    "eve-llm-1558M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "model_name": "eve_llm_1558M",
    }
}

model_configs_llama = {
    "eve-llm-llama-1B": {
        "model_name": "eve_llm_llama-1B",
        "vocab_size": 128_256,      # Vocabulary size
        "context_length": 131_072,  # Context length
        "emb_dim": 2048,            # NEW: Half the embedding dimension
        "n_heads": 32,              # Number of attention heads
        "n_layers": 16,             # NEW: Half the number of layers
        "hidden_dim": 8192,         # NEW: Almost half the size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,     # The base in RoPE's "theta"
        "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
        "rope_freq": {              # RoPE frequency scaling
            "factor": 32.0,         # NEW: Adjustment of the rescaling factor
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }
}

model_configs_qwen3 = {
    "eve-llm-qwen3-0P6B": {
        "vocab_size": 151_936,           # Vocabulary size
        "context_length": 40_960,        # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 28,                  # Number of layers
        "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
        "head_dim": 128,                 # Size of the heads in GQA
        "qk_norm": True,                 # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
        "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
        "model_name": "eve-llm-qwen3-0P6B",
    },
    "eve-llm-qwen3-1P7B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2048,                 # 2x larger than above
        "n_heads": 16,
        "n_layers": 28,
        "hidden_dim": 6144,              # 2x larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        "model_name": "eve-llm-qwen3-1P7B",
    },
    "eve-llm-qwen3-4B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2560,                 # 25% larger than above
        "n_heads": 32,                   # 2x larger than above
        "n_layers": 36,                  # 29% larger than above
        "hidden_dim": 9728,              # ~3x larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        "model_name": "eve-llm-qwen3-4B",
    },
    "eve-llm-qwen3-8B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 4096,                 # 60% larger than above
        "n_heads": 32,
        "n_layers": 36,                  # 26% larger than above
        "hidden_dim": 12288,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        "model_name": "eve-llm-qwen3-8B",
    },
    "eve-llm-qwen3-14B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,                 # 25% larger than above
        "n_heads": 40,                   # 25% larger than above
        "n_layers": 40,                  # 11% larger than above
        "hidden_dim": 17408,             # 42% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        "model_name": "eve-llm-qwen3-14B",
    },
    "eve-llm-qwen3-32B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,                   # 60% larger than above
        "n_layers": 64,                  # 60% larger than above
        "hidden_dim": 25600,             # 47% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
        "model_name": "eve-llm-qwen3-32B",
    },
}