import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import Actor, Critic, RandomScalarNetwork
from normalizer import Normalizer


class RND_PPOBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.97):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._clear()

    def _clear(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.ext_reward_buffer = []
        self.int_reward_buffer = []
        self.ext_value_buffer = []
        self.int_value_buffer = []
        self.log_prob_buffer = []

        self.ext_advantage_buffer = []
        self.int_advantage_buffer = []

        self.ext_ret_buffer = []
        self.int_ret_buffer = []

    def store(self, obs, action, ext_reward, int_reward, ext_value, int_value, log_prob):
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.ext_reward_buffer.append(ext_reward)
        self.int_reward_buffer.append(int_reward)
        self.ext_value_buffer.append(ext_value)
        self.int_value_buffer.append(int_value)
        self.log_prob_buffer.append(log_prob)

    def end_of_episode(self):
        self.ext_value_buffer.append(0)
        self.int_value_buffer.append(0)

        self.obs_buffer = np.array(self.obs_buffer)
        self.action_buffer = np.array(self.action_buffer)
        self.ext_reward_buffer = np.array(self.ext_reward_buffer)
        self.ext_value_buffer = np.array(self.ext_value_buffer)
        self.int_value_buffer = np.array(self.int_value_buffer)
        self.log_prob_buffer = np.array(self.log_prob_buffer)

        ###

        ext_deltas = self.ext_reward_buffer + self.gamma * self.ext_value_buffer[1:] - self.ext_value_buffer[:-1]

        self.ext_advantage_buffer = np.zeros_like(ext_deltas)
        for i, delta in enumerate(reversed(ext_deltas)):
            self.ext_advantage_buffer[-(i+1)] = delta * self.gamma * self.gae_lambda

        self.ext_ret_buffer = np.zeros_like(self.ext_reward_buffer)
        for i, r in enumerate(reversed(self.ext_reward_buffer)):
            self.ext_ret_buffer[-(i+1)] = r + self.gamma * self.ext_ret_buffer[-i]

        self.ext_ret_buffer = np.expand_dims(self.ext_ret_buffer, axis=1)

        ###

        int_deltas = self.int_reward_buffer + self.gamma * self.int_value_buffer[1:] - self.int_value_buffer[:-1]

        self.int_advantage_buffer = np.zeros_like(int_deltas)
        for i, delta in enumerate(reversed(int_deltas)):
            self.int_advantage_buffer[-(i+1)] = delta * self.gamma * self.gae_lambda

        self.int_ret_buffer = np.zeros_like(self.int_reward_buffer)
        for i, r in enumerate(reversed(self.int_reward_buffer)):
            self.int_ret_buffer[-(i+1)] = r + self.gamma * self.int_ret_buffer[-i]

        self.int_ret_buffer = np.expand_dims(self.int_ret_buffer, axis=1)

        ###

        self.combined_advantage_buffer = self.ext_advantage_buffer + self.int_advantage_buffer

    def get(self):
        data = dict(obs=self.obs_buffer, action=self.action_buffer, ext_ret=self.ext_ret_buffer,
                    int_ret=self.int_ret_buffer, advantage=self.combined_advantage_buffer, log_prob=self.log_prob_buffer)
        self._clear()

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class RND_PPOAgent:
    def __init__(self, obs_dim, action_dim, env_name, actor_lr=1e-4, critic_lr=1e-4, predictor_lr=1e-4, gamma=0.99, gae_lambda=0.97, clip_ratio=0.2, target_kl=0.01, train_actor_iters=80, train_critic_iters=80, train_predictor_iters=80):
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
        self.train_predictor_iters = train_predictor_iters

        self.actor = Actor(obs_dim, action_dim)
        self.ext_critic = Critic(obs_dim)
        self.int_critic = Critic(obs_dim)

        self.target_network = RandomScalarNetwork(self.obs_dim)
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.predictor_network = RandomScalarNetwork(self.obs_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.ext_critic_optimizer = Adam(self.ext_critic.parameters(), lr=critic_lr)
        self.int_critic_optimizer = Adam(self.int_critic.parameters(), lr=critic_lr)
        self.predictor_optimizer = Adam(self.predictor_network.parameters(), lr=predictor_lr)

        self.buffer = RND_PPOBuffer(gamma, gae_lambda)

        self.obs_normalizer = Normalizer(self.obs_dim)
        self.int_reward_normalizer = Normalizer(1, use_mean=False)

    def act(self, obs):
        obs = self.obs_normalizer.normalize(obs)
        return self.actor(torch.Tensor(obs))[0]

    def step(self, obs, action, ext_reward, next_obs, done):
        obs = self.obs_normalizer.normalize(obs)
        next_obs = self.obs_normalizer.normalize(next_obs)

        with torch.no_grad():
            _, log_prob = self.actor(torch.Tensor(obs), torch.from_numpy(action))
            ext_value = self.ext_critic(torch.Tensor(obs)).item()
            int_value = self.int_critic(torch.Tensor(obs)).item()

            int_reward = (self.predictor_network(torch.Tensor(next_obs)) -
                          self.target_network(torch.Tensor(next_obs))).item() ** 2

        self.buffer.store(obs, action, ext_reward, int_reward, ext_value, int_value, log_prob)

        if done:
            self.obs_normalizer.update(np.array(self.buffer.obs_buffer))
            self.int_reward_normalizer.update(np.array(self.buffer.int_reward_buffer))

            self.buffer.int_reward_buffer = self.int_reward_normalizer.normalize(
                self.buffer.int_reward_buffer)

            self.buffer.end_of_episode()
            self._learn()

    def save(self):
        os.makedirs(f"checkpoints/rnd_ppo/{self.env_name}", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "ext_critic": self.ext_critic.state_dict(),
                    "int_critic": self.int_critic.state_dict(),
                    "obs_normalizer_mean": self.obs_normalizer.mean,
                    "obs_normalizer_std": self.obs_normalizer.std},
                   f"checkpoints/rnd_ppo/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/rnd_ppo/{self.env_name}/actor_critic.pt")
        self.actor.load_state_dict(checkpoint["actor"])
        self.ext_critic.load_state_dict(checkpoint["ext_critic"])
        self.int_critic.load_state_dict(checkpoint["int_critic"])
        self.obs_normalizer.mean = checkpoint["obs_normalizer_mean"]
        self.obs_normalizer.std = checkpoint["obs_normalizer_std"]

    def _learn(self):
        data = self.buffer.get()
        obs, action, advantage, ext_ret, int_ret, old_log_prob = data['obs'], data[
            'action'], data['advantage'], data['ext_ret'], data['int_ret'], data['log_prob']

        for _ in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()

            log_prob = self.actor(obs, action)[1]
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
            self.ext_critic_optimizer.zero_grad()

            ext_critic_loss = F.mse_loss(self.ext_critic(obs), ext_ret)
            ext_critic_loss.backward()

            self.ext_critic_optimizer.step()

            ###

            self.int_critic_optimizer.zero_grad()

            int_critic_loss = F.mse_loss(self.int_critic(obs), int_ret)
            int_critic_loss.backward()

            self.int_critic_optimizer.step()

        for _ in range(self.train_predictor_iters):
            self.predictor_optimizer.zero_grad()

            predictor_loss = F.mse_loss(self.predictor_network(obs), self.target_network(obs))
            predictor_loss.backward()

            self.predictor_optimizer.step()
