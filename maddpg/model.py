import torch
import torch.nn as nn
import torch.nn.functional as F

# env used for this algorithm has discrete actions!!!
# this is why QNetwork only takes n_agents


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_agents):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size*n_agents, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.out = nn.Sigmoid()


    def forward(self, obs):
        x = self.fc1(obs)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # TODO: check here!!!
        return self.out(x)
