class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers