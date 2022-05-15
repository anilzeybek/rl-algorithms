import torch.nn as nn


class DuelingNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DuelingNetwork, self).__init__()

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
