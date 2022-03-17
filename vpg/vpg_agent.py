import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import Actor, Critic


class VPGBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.97):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._clear()

    def _clear(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []

        self.advantage_buffer = []
        self.ret_buffer = []

    def store(self, obs, action, reward, value):
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)

    def end_of_episode(self):
        self.value_buffer.append(0)

        self.obs_buffer = np.array(self.obs_buffer)
        self.action_buffer = np.array(self.action_buffer)
        self.reward_buffer = np.array(self.reward_buffer)
        self.value_buffer = np.array(self.value_buffer)

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
                    ret=self.ret_buffer, advantage=self.advantage_buffer)
        self._clear()

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class VPGAgent:
    def __init__(self, obs_dim, action_dim, env_name, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, gae_lambda=0.97, train_critic_iters=80):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.train_critic_iters = train_critic_iters

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = VPGBuffer(gamma, gae_lambda)

    def act(self, obs):
        return self.actor(torch.from_numpy(obs))[0]

    def step(self, obs, action, reward, done):
        self.buffer.store(obs, action, reward, self.critic(torch.from_numpy(obs)).item())

        if done:
            self.buffer.end_of_episode()
            self._learn()

    def save(self):
        os.makedirs(f"checkpoints/vpg/{self.env_name}", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()},
                   f"checkpoints/vpg/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/vpg/{self.env_name}/actor_critic.pt")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def _learn(self):
        data = self.buffer.get()
        obs, action, advantage, ret = data['obs'], data['action'], data['advantage'], data['ret']

        self.actor_optimizer.zero_grad()
        log_prob = self.actor(obs, action)[1]
        actor_loss = -(log_prob * advantage).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        for _ in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(self.critic(obs), ret)
            critic_loss.backward()
            self.critic_optimizer.step()
