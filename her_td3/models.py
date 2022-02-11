import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, action_bounds):
        super(Actor, self).__init__()
        self.max_action = max(action_bounds["high"])

        self.fc1 = nn.Linear(obs_dim + goal_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return torch.tanh(x) * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim + goal_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)

        self.fc6 = nn.Linear(obs_dim + action_dim + goal_dim, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 64)
        self.fc9 = nn.Linear(64, 64)
        self.fc10 = nn.Linear(64, 1)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)

        q1 = F.relu(self.fc1(obs_action))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = F.relu(self.fc4(q1))
        output1 = self.fc5(q1)

        q2 = F.relu(self.fc6(obs_action))
        q2 = F.relu(self.fc7(q2))
        q2 = F.relu(self.fc8(q2))
        q2 = F.relu(self.fc9(q2))
        output2 = self.fc10(q2)

        return output1, output2
