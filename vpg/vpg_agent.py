import os

import numpy as np
import torch
from torch.optim import Adam

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.normalizer import Normalizer
from common.models import LogProbActor


class VPGAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 env_name,
                 actor_lr,
                 gamma):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_lr = actor_lr
        self.gamma = gamma

        self.actor = LogProbActor(obs_dim, action_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        self.normalizer = Normalizer(self.obs_dim)
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.return_buffer = []

    def act(self, obs, **_):
        normalized_obs = self.normalizer.normalize(obs)
        return self.actor(torch.Tensor(normalized_obs))[0]

    def step(self, obs, action, reward, _, done):
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        self.normalizer.update(obs)

        if done:
            self.action_buffer = np.array(self.action_buffer)
            self.return_buffer = np.zeros_like(self.reward_buffer)
            for i, r in enumerate(reversed(self.reward_buffer)):
                self.return_buffer[-(i + 1)] = r + self.gamma * self.return_buffer[-i]

            self._learn()

            self.obs_buffer = []
            self.action_buffer = []
            self.reward_buffer = []

    def save(self):
        os.makedirs(f"checkpoints/vpg/{self.env_name}", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.std,
            "normalizer_running_sum": self.normalizer.running_sum,
            "normalizer_running_sumsq": self.normalizer.running_sumsq,
            "normalizer_running_count": self.normalizer.running_count,
        }, f"checkpoints/vpg/{self.env_name}/actor.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/vpg/{self.env_name}/actor.pt")
        self.actor.load_state_dict(checkpoint["actor"])

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.std = checkpoint["normalizer_std"]
        self.normalizer.running_sum = checkpoint["normalizer_running_sum"]
        self.normalizer.running_sumsq = checkpoint["normalizer_running_sumsq"]
        self.normalizer.running_count = checkpoint["normalizer_running_count"]

    def _learn(self):
        normalized_obs = self.normalizer.normalize(torch.Tensor(self.obs_buffer)).float()

        self.actor_optimizer.zero_grad()
        log_prob = self.actor(normalized_obs, torch.Tensor(self.action_buffer))[1]
        actor_loss = -(log_prob * torch.Tensor(self.return_buffer)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
