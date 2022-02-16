import torch.nn as nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DuelingNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 64)

        self.v1 = nn.Linear(64, 64)
        self.v2 = nn.Linear(64, 1)

        self.a1 = nn.Linear(64, 64)
        self.a2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        v = F.relu(self.v1(x))
        v = self.v2(v)

        a = F.relu(self.a1(x))
        a = self.a2(a)

        Q = v + a - a.mean(dim=1, keepdim=True)
        return Q
