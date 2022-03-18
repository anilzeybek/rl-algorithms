import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs, action=None):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        distribution = Categorical(logits=x)
        a = distribution.sample()

        log_prob = None
        if action is not None:
            log_prob = distribution.log_prob(action)

        return a.numpy(), log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
