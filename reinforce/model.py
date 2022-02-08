import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
