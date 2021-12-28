from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer
from copy import deepcopy


BUFFER_SIZE = 25000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.05
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
START_STEPS = 25000
UPDATE_EVERY = 10
UPDATE_AFTER = 1000
ACT_NOISE = 0.1


class DDPGAgent:
    def __init__(self, obs_dim: int, action_dim: int, act_limit_low: float, act_limit_high: float) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limits = (act_limit_low, act_limit_high)
        self.t = 0

        self.actor = PolicyNetwork(obs_dim, action_dim, self.action_limits)
        self.actor_target = deepcopy(self.actor)

        self.critic = QNetwork(obs_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def act(self, state: np.ndarray, train_mode=True) -> np.ndarray:
        with torch.no_grad():
            x = torch.Tensor(state)
            action = self.actor(x).numpy()

        if train_mode:
            action += self.action_limits[1] / 5 * np.random.randn(self.action_dim)
            action = np.clip(action, self.action_limits[0], self.action_limits[1])

            random_actions = np.random.uniform(low=self.action_limits[0], high=self.action_limits[1], size=self.action_dim)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action

    def step(self, state: np.ndarray, action: np.ndarray, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.t += 1
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

        if self.t >= UPDATE_AFTER and self.t % UPDATE_EVERY == 0:
            for _ in range(UPDATE_EVERY):
                batch = self.replay_buffer.sample(BATCH_SIZE)
                self._learn(data=batch)

    def _compute_loss_q(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        state, action, reward, next_state, done = data
        Q_current = self.critic(state, action)

        with torch.no_grad():
            Q_target_next = self.critic_target(next_state, self.actor_target(next_state))
            Q_target = reward + GAMMA * Q_target_next * (1 - done)

        return F.mse_loss(Q_current, Q_target)

    def _compute_loss_pi(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        state = data[0]
        Q = self.critic(state, self.actor(state))

        return -Q.mean()

    def _learn(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        self.critic_optimizer.zero_grad()
        loss_Q = self._compute_loss_q(data)
        loss_Q.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self._compute_loss_pi(data)
        loss_pi.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._soft_update_networks(self.actor, self.actor_target)
        self._soft_update_networks(self.critic, self.critic_target)

    @ staticmethod
    def _soft_update_networks(local_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(TAU * e_params.data + (1 - TAU) * t_params.data)
