import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer
from copy import deepcopy


BUFFER_SIZE = 200000
BATCH_SIZE = 64
GAMMA = 0.99
POLYAK = 0.995
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
START_STEPS = 50000
UPDATE_EVERY = 10   
UPDATE_AFTER = 1000
ACT_NOISE = 0.1


class DDPG_HERAgent:
    def __init__(self, obs_dim, act_dim, act_limits):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limits = act_limits
        self.t = 0

        self.actor_network = PolicyNetwork(obs_dim, act_dim, act_limits)
        self.actor_target = deepcopy(self.actor_network)

        self.critic_network = QNetwork(obs_dim, act_dim)
        self.critic_target = deepcopy(self.critic_network)

        self.trajectory = []
        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LR_CRITIC)

    def act(self, state, noise=ACT_NOISE):
        if self.t > START_STEPS:
            with torch.no_grad():
                a = self.actor_network(torch.as_tensor(state, dtype=torch.float32)).numpy()
                a += noise * np.random.randn(self.act_dim)
                return np.clip(a, -self.act_limits, self.act_limits)
        else:
            return np.random.uniform(low=-self.act_limits, high=self.act_limits, size=(self.act_dim,))

    def step(self, state, action, reward, next_state, done):
        self.t += 1
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

        # Below part is the implementation of HER
        # If you remove it from the code, training will take sooo long
        self.trajectory.append((state, action, reward, next_state, done))
        if done:
            for transition in self.trajectory:
                updated_state = np.concatenate((transition[0][:self.obs_dim//2], transition[3][:self.obs_dim//2]))
                updated_next_state = np.concatenate((transition[3][:self.obs_dim//2], transition[3][:self.obs_dim//2]))

                self.replay_buffer.store_transition(updated_state, transition[1], 1, updated_next_state, True)
            self.trajectory = []

        if self.t >= UPDATE_AFTER and self.t % UPDATE_EVERY == 0:
            for _ in range(UPDATE_EVERY):
                batch = self.replay_buffer.sample(BATCH_SIZE)
                self._learn(data=batch)

    def _compute_loss_q(self, data):
        state, action, reward, next_state, done = data
        Q_current = self.critic_network(state, action)

        with torch.no_grad():
            Q_target_next = self.critic_target(next_state, self.actor_target(next_state))
            Q_target = reward + GAMMA * Q_target_next * (1 - done)

        loss = ((Q_current - Q_target) ** 2).mean()
        return loss

    def _compute_loss_pi(self, data):
        state = data[0]
        Q = self.critic_network(state, self.actor_network(state))

        return -Q.mean()

    def _learn(self, data):
        self.critic_optimizer.zero_grad()
        loss_Q = self._compute_loss_q(data)
        loss_Q.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic_network.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self._compute_loss_pi(data)
        loss_pi.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic_network.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_network.parameters(), self.actor_target.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)

            for p, p_targ in zip(self.critic_network.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)
