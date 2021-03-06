import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.normalizer import Normalizer
from common.models import DuelingQNetwork


class DuelingDQNAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 env_name,
                 start_timesteps,
                 buffer_size,
                 lr,
                 batch_size,
                 gamma,
                 tau,
                 eps_init,
                 eps_last,
                 eps_decay):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.start_timesteps = start_timesteps
        self.buffer_size = buffer_size
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eps = eps_init
        self.eps_last = eps_last
        self.eps_decay = eps_decay

        self.dueling_network = DuelingQNetwork(obs_dim, action_dim)
        self.target_network = deepcopy(self.dueling_network)

        self.optimizer = optim.Adam(self.dueling_network.parameters(), lr=self.lr)
        self.normalizer = Normalizer(self.obs_dim)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })
        self.t = 0

    def act(self, obs, train_mode=True):
        if train_mode and np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        else:
            normalized_obs = torch.Tensor(self.normalizer.normalize(obs)).unsqueeze(0)
            action_values = self.dueling_network(normalized_obs)
            return torch.argmax(action_values).item()

    def step(self, obs, action, reward, next_obs, done):
        self.t += 1
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self.normalizer.update(obs)

        if self.t >= self.start_timesteps:
            self._learn()
            self.eps = max(self.eps_last, self.eps - self.eps_decay)

        if done:
            self.rb.on_episode_end()

    def save(self):
        os.makedirs(f"checkpoints/dueling_dqn/{self.env_name}", exist_ok=True)
        torch.save({
            "dueling_network": self.dueling_network.state_dict(),
            "eps": self.eps,
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.std,
            "normalizer_running_sum": self.normalizer.running_sum,
            "normalizer_running_sumsq": self.normalizer.running_sumsq,
            "normalizer_running_count": self.normalizer.running_count,
            "t": self.t
        }, f"checkpoints/dueling_dqn/{self.env_name}/dueling_network.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/dueling_dqn/{self.env_name}/dueling_network.pt")

        self.dueling_network.load_state_dict(checkpoint["dueling_network"])
        self.target_network = deepcopy(self.dueling_network)

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.std = checkpoint["normalizer_std"]
        self.normalizer.running_sum = checkpoint["normalizer_running_sum"]
        self.normalizer.running_sumsq = checkpoint["normalizer_running_sumsq"]
        self.normalizer.running_count = checkpoint["normalizer_running_count"]

        self.t = checkpoint["t"]
        self.eps = checkpoint["eps"]

    def _learn(self):
        sample = self.rb.sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action']).long()
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        normalized_obs = self.normalizer.normalize(obs).float()
        normalized_next_obs = self.normalizer.normalize(next_obs).float()

        Q_current = self.dueling_network(normalized_obs).gather(1, action)
        with torch.no_grad():
            a = self.dueling_network(normalized_next_obs).argmax(1).unsqueeze(1)
            Q_target_next = self.target_network(normalized_next_obs).gather(1, a)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

        loss = F.mse_loss(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dueling_network.parameters(), 10)
        self.optimizer.step()

        for param, target_param in zip(self.dueling_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
