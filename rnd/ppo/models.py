import torch.nn as nn
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs, action=None):
        x = self.net(obs)

        distribution = Categorical(logits=x)
        a = distribution.sample()

        log_prob = None
        if action is not None:
            log_prob = distribution.log_prob(action)

        return a.numpy(), log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs)


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
