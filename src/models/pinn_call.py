from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PINNConfig:
    input_dim: int = 2
    hidden_dim: int = 64
    num_hidden_layers: int = 4
    activation: str = "tanh"


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()


class PINNCall(nn.Module):

    def __init__(self, config: PINNConfig):
        super().__init__()

        self.config = config
        act = get_activation(config.activation)
        layers = []

        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(get_activation(config.activation))

        for _ in range(config.num_hidden_layers):

            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(act)

        layers.append(nn.Linear(config.hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t, S):

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        if S.dim() == 1:
            S = S.unsqueeze(-1)

        x = torch.cat([t, S], dim=1)
        V = self.net(x)

        return V.squeeze(-1)
