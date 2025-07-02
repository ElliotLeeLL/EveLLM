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
    "LLAMA32_CONFIG_1B":{
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