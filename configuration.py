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
    },
    "eve-llm-355M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "eve-llm-774M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "eve-llm-1558M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }
}