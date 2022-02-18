import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

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
            logp_pi -= 2*(np.log(2) - pi_action - F.softplus(-2*pi_action)).sum(axis=1)
            logp_pi = logp_pi.unsqueeze(1)

        pi_action = self.max_action * torch.tanh(pi_action)

        return pi_action, logp_pi


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc4 = nn.Linear(obs_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)

        q1 = F.relu(self.fc1(obs_action))
        q1 = F.relu(self.fc2(q1))
        output1 = self.fc3(q1)

        q2 = F.relu(self.fc4(obs_action))
        q2 = F.relu(self.fc5(q2))
        output2 = self.fc6(q2)

        return output1, output2
