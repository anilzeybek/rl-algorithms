import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import Actor, Critic
from normalizer import Normalizer


class PPOBuffer:
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._clear()

    def _clear(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []

        self.advantage_buffer = []
        self.ret_buffer = []

    def store(self, obs, action, reward, value, log_prob):
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.log_prob_buffer.append(log_prob)

    def end_of_episode(self):
        self.value_buffer.append(0)

        self.obs_buffer = np.array(self.obs_buffer)
        self.action_buffer = np.array(self.action_buffer)
        self.reward_buffer = np.array(self.reward_buffer)
        self.value_buffer = np.array(self.value_buffer)
        self.log_prob_buffer = np.array(self.log_prob_buffer)

        deltas = self.reward_buffer + self.gamma * self.value_buffer[1:] - self.value_buffer[:-1]

        self.advantage_buffer = np.zeros_like(deltas)
        for i, delta in enumerate(reversed(deltas)):
            self.advantage_buffer[-(i+1)] = delta * self.gamma * self.gae_lambda

        self.ret_buffer = np.zeros_like(self.reward_buffer)
        for i, r in enumerate(reversed(self.reward_buffer)):
            self.ret_buffer[-(i+1)] = r + self.gamma * self.ret_buffer[-i]

        self.ret_buffer = np.expand_dims(self.ret_buffer, axis=1)

    def get(self):
        data = dict(obs=self.obs_buffer, action=self.action_buffer,
                    ret=self.ret_buffer, advantage=self.advantage_buffer, log_prob=self.log_prob_buffer)
        self._clear()

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 env_name,
                 actor_lr,
                 critic_lr,
                 gamma,
                 clip_ratio,
                 target_kl,
                 gae_lambda,
                 train_actor_iters,
                 train_critic_iters):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.normalizer = Normalizer(self.obs_dim)
        self.buffer = PPOBuffer(gamma, gae_lambda)

    def act(self, obs):
        normalized_obs = self.normalizer.normalize(obs)
        return self.actor(torch.Tensor(normalized_obs))[0]

    def step(self, obs, action, reward, done):
        normalized_obs = self.normalizer.normalize(obs)
        with torch.no_grad():
            _, log_prob = self.actor(torch.Tensor(normalized_obs), torch.from_numpy(action))
            value = self.critic(torch.Tensor(normalized_obs)).item()

        self.buffer.store(obs, action, reward, value, log_prob)
        self.normalizer.update(obs)

        if done:
            self.buffer.end_of_episode()
            self._learn()

    def save(self):
        os.makedirs(f"checkpoints/ppo/{self.env_name}", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.std,
            "normalizer_running_sum": self.normalizer.running_sum,
            "normalizer_running_sumsq": self.normalizer.running_sumsq,
            "normalizer_running_count": self.normalizer.running_count,
        }, f"checkpoints/ppo/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/ppo/{self.env_name}/actor_critic.pt")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.std = checkpoint["normalizer_std"]
        self.normalizer.running_sum = checkpoint["normalizer_running_sum"]
        self.normalizer.running_sumsq = checkpoint["normalizer_running_sumsq"]
        self.normalizer.running_count = checkpoint["normalizer_running_count"]

    def _learn(self):
        data = self.buffer.get()
        obs, action, advantage, ret, old_log_prob = data['obs'], data['action'], data['advantage'], data['ret'], data['log_prob']

        normalized_obs = self.normalizer.normalize(obs).float()

        for _ in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()

            log_prob = self.actor(normalized_obs, action)[1]
            ratio = torch.exp(log_prob - old_log_prob)
            clipped_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantage
            actor_loss = -(torch.min(ratio*advantage, clipped_adv)).mean()

            with torch.no_grad():
                kl = (old_log_prob - log_prob).mean().item()
                if kl > 1.5 * self.target_kl:
                    break

            actor_loss.backward()
            self.actor_optimizer.step()

        for _ in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()

            critic_loss = F.mse_loss(self.critic(normalized_obs), ret)
            critic_loss.backward()

            self.critic_optimizer.step()
