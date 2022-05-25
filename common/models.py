import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F


class DiscreteQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class ContinuousQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        return self.net(obs_action)


class VNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.init_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU()
        )

        self.v_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.a_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs):
        x = self.init_net(obs)
        v = self.v_net(x)
        a = self.a_net(x)

        Q = v + a - a.mean(dim=1, keepdim=True)
        return Q


class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)

        x1 = self.net1(obs_action)
        x2 = self.net2(obs_action)

        return x1, x2


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        x = self.net(obs)
        return x * self.max_action


class LogProbActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

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


class SoftActor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.init_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = self.init_net(obs)

        mu = self.mu(x)

        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        logp_pi = False
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= 2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action)).sum(axis=1)
            logp_pi = logp_pi.unsqueeze(1)

        pi_action = self.max_action * torch.tanh(pi_action)

        return pi_action, logp_pi
