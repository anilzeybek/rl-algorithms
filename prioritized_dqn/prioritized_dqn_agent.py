from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from cpprb import PrioritizedReplayBuffer
import os


class PrioritizedDQNAgent:
    def __init__(self, obs_dim, action_dim, env_name, buffer_size=65536, lr=1e-3, batch_size=64, gamma=0.99, tau=0.05, eps_start=1.0, eps_end=0.01, eps_decay=0.995, alpha=0.6, beta=0.4, train_mode=True):
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
        self.alpha = alpha
        self.beta = beta
        self.train_mode = train_mode

        self.Q_network = QNetwork(obs_dim, action_dim)
        self.target_network = deepcopy(self.Q_network)

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.lr)

        self.eps = self.eps_start
        self.prb = PrioritizedReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}},
            alpha=self.alpha
        )

    def act(self, obs):
        if self.train_mode and np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        else:
            obs = torch.from_numpy(obs).unsqueeze(0)
            action_values = self.Q_network(obs)
            return torch.argmax(action_values).item()

    def step(self, obs, action, reward, next_obs, done):
        self.prb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self._learn()

        if done:
            self._update_eps()
            self.prb.on_episode_end()

    def save(self):
        os.makedirs(f"saved_networks/prioritized_dqn/{self.env_name}", exist_ok=True)
        torch.save(self.Q_network.state_dict(), f"saved_networks/prioritized_dqn/{self.env_name}/Q_network.pt")

    def load(self):
        self.Q_network.load_state_dict(torch.load(f"saved_networks/prioritized_dqn/{self.env_name}/Q_network.pt"))

    def _update_eps(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def _learn(self):
        sample = self.prb.sample(self.batch_size, beta=self.beta)

        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action']).long()
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        Q_current = self.Q_network(obs).gather(1, action)
        with torch.no_grad():
            a = self.Q_network(next_obs).argmax(1).unsqueeze(1)
            Q_target_next = self.target_network(next_obs).gather(1, a)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

            error = (Q_target - Q_current).abs().numpy()

        self.prb.update_priorities(sample['indexes'], error)

        self.optimizer.zero_grad()
        loss = (torch.Tensor(sample['weights']) * F.mse_loss(Q_current, Q_target)).mean()
        loss.backward()
        self.optimizer.step()

        for t_params, e_params in zip(self.target_network.parameters(), self.Q_network.parameters()):
            t_params.data.copy_(self.tau * e_params.data + (1 - self.tau) * t_params.data)
