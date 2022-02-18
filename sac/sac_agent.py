from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import Actor, Critic
from cpprb import ReplayBuffer
import os


class SACAgent:
    def __init__(self, obs_dim, action_dim, action_bounds, env_name, alpha=0.1, start_timesteps=25000, buffer_size=200000, actor_lr=1e-3, critic_lr=1e-3, batch_size=256, gamma=0.99, tau=0.005):
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

        self.actor = Actor(obs_dim, action_dim, self.max_action)

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
        self.t = 0

    def act(self, obs, train_mode=True):
        if not train_mode:
            with torch.no_grad():
                action = self.actor(torch.Tensor(obs), deterministic=True, with_logprob=False)[0].numpy()
        elif train_mode and self.t < self.start_timesteps:
            action = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                       size=self.action_dim)
        else:
            with torch.no_grad():
                action = self.actor(torch.Tensor(obs), with_logprob=False)[0].numpy()

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, reward, next_obs, done):
        self.t += 1
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        if self.t >= self.start_timesteps:
            self._learn()

        if done:
            self.rb.on_episode_end()

    def save(self):
        os.makedirs(f"saved_networks/sac/{self.env_name}", exist_ok=True)
        torch.save(self.actor.state_dict(), f"saved_networks/sac/{self.env_name}/actor.pt")

    def load(self):
        self.actor.load_state_dict(torch.load(f"saved_networks/sac/{self.env_name}/actor.pt"))

    def _learn(self):
        sample = self.rb.sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        Q_current1, Q_current2 = self.critic(obs, action)
        with torch.no_grad():
            next_action, logp_next_action = self.actor(next_obs)

            Q1_target_next, Q2_target_next = self.critic_target(next_obs, next_action)
            Q_target_next = torch.min(Q1_target_next, Q2_target_next)
            Q_target = reward + self.gamma * (Q_target_next - self.alpha * logp_next_action) * (1 - done)

        critic_loss = F.mse_loss(Q_current1, Q_target) + F.mse_loss(Q_current2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ############

        actor_action, logp_actor_action = self.actor(obs)
        Q_actor1, Q_actor2 = self.critic(obs, actor_action)
        Q_actor = torch.min(Q_actor1, Q_actor2)

        actor_loss = (self.alpha * logp_actor_action - Q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
