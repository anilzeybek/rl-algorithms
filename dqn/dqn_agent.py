import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer

from model import QNetwork


class DQNAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 env_name,
                 buffer_size,
                 lr,
                 batch_size,
                 gamma,
                 tau,
                 eps_start,
                 eps_end,
                 eps_decay):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.buffer_size = buffer_size
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.Q_network = QNetwork(obs_dim, action_dim)
        self.target_network = deepcopy(self.Q_network)

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.lr)

        self.eps = self.eps_start
        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })

    def act(self, obs, train_mode=True):
        if train_mode and np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        else:
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            action_values = self.Q_network(obs)
            return torch.argmax(action_values).item()

    def step(self, obs, action, reward, next_obs, done):
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self._learn()

        if done:
            self.eps = max(self.eps_end, self.eps_decay * self.eps)
            self.rb.on_episode_end()

    def save(self):
        os.makedirs(f"checkpoints/dqn/{self.env_name}", exist_ok=True)
        torch.save({
            "Q_network": self.Q_network.state_dict(),
            "eps": self.eps
        }, f"checkpoints/dqn/{self.env_name}/Q_network.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/dqn/{self.env_name}/Q_network.pt")

        self.Q_network.load_state_dict(checkpoint["Q_network"])
        self.target_network = deepcopy(self.Q_network)

        self.eps = checkpoint["eps"]

    def _learn(self):
        sample = self.rb.sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action']).long()
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        Q_current = self.Q_network(obs).gather(1, action)
        with torch.no_grad():
            Q_target_next = self.target_network(next_obs).max(1)[0].unsqueeze(1)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

        loss = F.mse_loss(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for param, target_param in zip(self.Q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
