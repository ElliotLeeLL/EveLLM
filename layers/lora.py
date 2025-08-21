import math
import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, config):
        super(LoRALayer, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank)).to(device=device, dtype=config["dtype"])
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)).to(device=device, dtype=config["dtype"])
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, config):
        super(LinearWithLoRA, self).__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            config
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)