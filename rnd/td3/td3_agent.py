import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer

from models import Actor, Critic, RandomScalarNetwork
from normalizer import Normalizer


class TD3Agent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 env_name,
                 expl_noise,
                 start_timesteps,
                 buffer_size,
                 actor_lr,
                 critic_lr,
                 predictor_lr,
                 batch_size,
                 gamma,
                 tau,
                 policy_noise,
                 noise_clip,
                 policy_freq):

        self.max_action = max(action_bounds["high"])

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.env_name = env_name
        self.expl_noise = expl_noise
        self.start_timesteps = start_timesteps
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.predictor_lr = predictor_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.policy_freq = policy_freq

        self.actor = Actor(obs_dim, action_dim, self.max_action)
        self.actor_target = deepcopy(self.actor)

        self.ext_critic = Critic(obs_dim, action_dim)
        self.ext_critic_target = deepcopy(self.ext_critic)

        self.int_critic = Critic(obs_dim, action_dim)
        self.int_critic_target = deepcopy(self.int_critic)

        self.target_network = RandomScalarNetwork(self.obs_dim)
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.predictor_network = RandomScalarNetwork(self.obs_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.ext_critic_optimizer = optim.Adam(self.ext_critic.parameters(), lr=self.critic_lr)
        self.int_critic_optimizer = optim.Adam(self.int_critic.parameters(), lr=self.critic_lr)
        self.predictor_optimizer = optim.Adam(self.predictor_network.parameters(), lr=self.predictor_lr)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "ext_reward": {},
            "int_reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })
        self.t = 0

        self.obs_normalizer = Normalizer(self.obs_dim)
        self.int_reward_normalizer = Normalizer(1, use_mean=False)

        self.this_ep_observations = []
        self.this_ep_int_rewards = []

    def act(self, obs, train_mode=True):
        obs = self.obs_normalizer.normalize(obs)

        with torch.no_grad():
            if not train_mode:
                action = self.actor(torch.Tensor(obs)).numpy()
            else:
                if self.t < self.start_timesteps:
                    action = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
                else:
                    action = (
                        self.actor(torch.Tensor(obs)).numpy()
                        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                    )

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, ext_reward, next_obs, done):
        self.t += 1

        obs = self.obs_normalizer.normalize(obs)
        next_obs = self.obs_normalizer.normalize(next_obs)

        with torch.no_grad():
            int_reward = (self.predictor_network(torch.Tensor(next_obs)) -
                          self.target_network(torch.Tensor(next_obs))).item() ** 2

            int_reward = self.int_reward_normalizer.normalize(int_reward)

        self.this_ep_observations.append(obs)
        self.this_ep_int_rewards.append(int_reward)

        self.rb.add(obs=obs, action=action, ext_reward=ext_reward, int_reward=int_reward, next_obs=next_obs, done=done)

        if self.t >= self.start_timesteps:
            self._learn()

        if done:
            self.obs_normalizer.update(np.array(self.this_ep_observations))
            self.int_reward_normalizer.update(np.array(self.this_ep_int_rewards))

            self.this_ep_observations = []
            self.this_ep_int_rewards = []

            self.rb.on_episode_end()

    def save(self):
        os.makedirs(f"checkpoints/rnd/td3/{self.env_name}", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "ext_critic": self.ext_critic.state_dict(),
                    "int_critic": self.int_critic.state_dict(),
                    "obs_normalizer_mean": self.obs_normalizer.mean,
                    "obs_normalizer_std": self.obs_normalizer.std,
                    "t": self.t
                    }, f"checkpoints/rnd/td3/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/rnd/td3/{self.env_name}/actor_critic.pt")

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target = deepcopy(self.actor)

        self.ext_critic.load_state_dict(checkpoint["ext_critic"])
        self.int_critic.load_state_dict(checkpoint["int_critic"])
        self.obs_normalizer.mean = checkpoint["obs_normalizer_mean"]
        self.obs_normalizer.std = checkpoint["obs_normalizer_std"]

        self.t = checkpoint["t"]

    def _learn(self):
        sample = self.rb.sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action'])
        ext_reward = torch.Tensor(sample['ext_reward'])
        int_reward = torch.Tensor(sample['int_reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        ext_Q_current1, ext_Q_current2 = self.ext_critic(obs, action)
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_obs) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            ext_Q1_target_next, ext_Q2_target_next = self.ext_critic_target(next_obs, next_actions)
            ext_Q_target_next = torch.min(ext_Q1_target_next, ext_Q2_target_next)
            ext_Q_target = ext_reward + self.gamma * ext_Q_target_next * (1 - done)

        ext_critic_loss = F.mse_loss(ext_Q_current1, ext_Q_target) + F.mse_loss(ext_Q_current2, ext_Q_target)

        self.ext_critic_optimizer.zero_grad()
        ext_critic_loss.backward()
        self.ext_critic_optimizer.step()

        #####

        int_Q_current1, int_Q_current2 = self.int_critic(obs, action)
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_obs) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            int_Q1_target_next, int_Q2_target_next = self.int_critic_target(next_obs, next_actions)
            int_Q_target_next = torch.min(int_Q1_target_next, int_Q2_target_next)
            int_Q_target = int_reward + self.gamma * int_Q_target_next * (1 - done)

        int_critic_loss = F.mse_loss(int_Q_current1, int_Q_target) + F.mse_loss(int_Q_current2, int_Q_target)

        self.int_critic_optimizer.zero_grad()
        int_critic_loss.backward()
        self.int_critic_optimizer.step()

        #####

        self.predictor_optimizer.zero_grad()

        predictor_loss = F.mse_loss(self.predictor_network(obs), self.target_network(obs))
        predictor_loss.backward()

        self.predictor_optimizer.step()

        #####

        if self.t % self.policy_freq == 0:
            a = self.actor(obs)
            actor_loss = -(self.ext_critic(obs, a)[0].mean() + self.int_critic(obs, a)[0].mean())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.ext_critic.parameters(), self.ext_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.int_critic.parameters(), self.int_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
