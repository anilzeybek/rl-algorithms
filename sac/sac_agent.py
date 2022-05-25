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
from common.models import SoftActor, TwinQNetwork


class SACAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 env_name,
                 alpha,
                 start_timesteps,
                 buffer_size,
                 actor_lr,
                 critic_lr,
                 batch_size,
                 gamma,
                 tau):

        self.max_action = max(action_bounds["high"])

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.env_name = env_name
        self.alpha = alpha
        self.start_timesteps = start_timesteps
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = SoftActor(obs_dim, action_dim, self.max_action)

        self.critic = TwinQNetwork(obs_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.normalizer = Normalizer(self.obs_dim)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })
        self.t = 0

    def act(self, obs, train_mode=True):
        normalized_obs = self.normalizer.normalize(obs)
        if not train_mode:
            with torch.no_grad():
                action = self.actor(torch.Tensor(normalized_obs), deterministic=True, with_logprob=False)[0].numpy()
        elif train_mode and self.t < self.start_timesteps:
            action = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                       size=self.action_dim)
        else:
            with torch.no_grad():
                action = self.actor(torch.Tensor(normalized_obs), with_logprob=False)[0].numpy()

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, reward, next_obs, done):
        self.t += 1
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self.normalizer.update(obs)

        if self.t >= self.start_timesteps:
            self._learn()

        if done:
            self.rb.on_episode_end()

    def save(self):
        os.makedirs(f"checkpoints/sac/{self.env_name}", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.std,
            "normalizer_running_sum": self.normalizer.running_sum,
            "normalizer_running_sumsq": self.normalizer.running_sumsq,
            "normalizer_running_count": self.normalizer.running_count,
            "t": self.t
        }, f"checkpoints/sac/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/sac/{self.env_name}/actor_critic.pt")

        self.actor.load_state_dict(checkpoint["actor"])

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = deepcopy(self.critic)

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.std = checkpoint["normalizer_std"]
        self.normalizer.running_sum = checkpoint["normalizer_running_sum"]
        self.normalizer.running_sumsq = checkpoint["normalizer_running_sumsq"]
        self.normalizer.running_count = checkpoint["normalizer_running_count"]

        self.t = checkpoint["t"]

    def _learn(self):
        sample = self.rb.sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        normalized_obs = self.normalizer.normalize(obs).float()
        normalized_next_obs = self.normalizer.normalize(next_obs).float()

        Q_current1, Q_current2 = self.critic(normalized_obs, action)
        with torch.no_grad():
            next_action, logp_next_action = self.actor(normalized_next_obs)

            Q1_target_next, Q2_target_next = self.critic_target(normalized_next_obs, next_action)
            Q_target_next = torch.min(Q1_target_next, Q2_target_next)
            Q_target = reward + self.gamma * (Q_target_next - self.alpha * logp_next_action) * (1 - done)

        critic_loss = F.mse_loss(Q_current1, Q_target) + F.mse_loss(Q_current2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ############

        actor_action, logp_actor_action = self.actor(normalized_obs)
        Q_actor1, Q_actor2 = self.critic(normalized_obs, actor_action)
        Q_actor = torch.min(Q_actor1, Q_actor2)

        actor_loss = (self.alpha * logp_actor_action - Q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
