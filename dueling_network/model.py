import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        self.v1 = nn.Linear(64, 32)
        self.v2 = nn.Linear(32, 1)

        self.a1 = nn.Linear(64, 32)
        self.a2 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.v1(x))
        v = self.v2(v)

        a = F.relu(self.a1(x))
        a = self.a2(a)

        Q = v + a - a.mean(dim=1, keepdim=True)
        return Q
