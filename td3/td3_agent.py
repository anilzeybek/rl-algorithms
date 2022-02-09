from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import Actor, Critic
from cpprb import ReplayBuffer
import os


class TD3Agent:
    def __init__(self, obs_dim, action_dim, action_bounds, env_name, buffer_size=65536, actor_lr=1e-3, critic_lr=1e-3, batch_size=64, gamma=0.99, tau=0.05, policy_noise=0.05, noise_clip=0.1, policy_freq=4, train_mode=True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.env_name = env_name
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.train_mode = train_mode

        self.actor = Actor(obs_dim, action_dim, action_bounds)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })
        self.total_it = 0

    def act(self, obs):
        with torch.no_grad():
            action = self.actor(torch.Tensor(obs)).numpy()

        if self.train_mode:
            action += max(self.action_bounds['high']) / 5 * np.random.randn(self.action_dim)
            action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])

            random_actions = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, reward, next_obs, done):
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        if done:
            self._learn()
            self.rb.on_episode_end()

    def save(self):
        os.makedirs(f"saved_networks/ddpg/{self.env_name}", exist_ok=True)
        torch.save(self.actor.state_dict(), f"saved_networks/ddpg/{self.env_name}/actor.pt")

    def load(self):
        self.actor.load_state_dict(torch.load(f"saved_networks/ddpg/{self.env_name}/actor.pt"))

    def _learn(self):
        self.total_it += 1

        sample = self.rb.sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        Q_current1, Q_current2 = self.critic(obs, action)
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_obs) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            Q1_target_next, Q2_target_next = self.critic_target(next_obs, next_actions)
            Q_target_next = torch.min(Q1_target_next, Q2_target_next)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

        critic_loss = F.mse_loss(Q_current1, Q_target) + F.mse_loss(Q_current2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic(obs, self.actor(obs))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for t_params, e_params in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_params.data.copy_(self.tau * e_params.data + (1 - self.tau) * t_params.data)

            for t_params, e_params in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_params.data.copy_(self.tau * e_params.data + (1 - self.tau) * t_params.data)
