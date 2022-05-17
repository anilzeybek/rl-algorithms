import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        x = self.net(obs)
        return x * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)

        x1 = self.net1(obs_action)
        x2 = self.net2(obs_action)

        return x1, x2


class RandomScalarNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(RandomScalarNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        return self.net(obs)
