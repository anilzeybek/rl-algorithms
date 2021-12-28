from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_limits: Tuple[float, float]) -> None:
        super(PolicyNetwork, self).__init__()
        self.action_limits = torch.Tensor(action_limits)

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return torch.tanh(x) * self.action_limits[1]


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
