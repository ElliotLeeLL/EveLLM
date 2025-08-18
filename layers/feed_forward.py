import torch
import torch.nn as nn
from layers.gelu import GELU
from layers.silu import SiLU


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class GatedFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.fc2 = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.fc3 = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)