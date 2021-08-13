import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer
from copy import deepcopy


BUFFER_SIZE = 1000000
BATCH_SIZE = 64
GAMMA = 0.99
POLYAK = 0.995
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
START_STEPS = 10000
UPDATE_EVERY = 50
UPDATE_AFTER = 1000
ACT_NOISE = 0.1


# TODO: act_limit is for now scalar, but limits may differ from action to action, so fix it
class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_limit):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.t = 0

        self.actor_network = PolicyNetwork(obs_dim, act_dim, act_limit)
        self.actor_target = deepcopy(self.actor_network)

        self.critic_network = QNetwork(obs_dim, act_dim)
        self.critic_target = deepcopy(self.critic_network)

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LR_CRITIC)

    def act(self, state):
        if self.t > START_STEPS:
            with torch.no_grad():
                a = self.actor_network(torch.as_tensor(state, dtype=torch.float32)).numpy()
                a += ACT_NOISE * np.random.randn(self.act_dim)
                return np.clip(a, -self.act_limit, self.act_limit)
        else:
            return np.random.uniform(low=-self.act_limit, high=self.act_limit, size=(self.act_dim,))

    def step(self, state, action, reward, next_state, done):
        self.t += 1
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

        if self.t >= UPDATE_AFTER and self.t % UPDATE_EVERY == 0:
            for _ in range(UPDATE_EVERY):
                batch = self.replay_buffer.sample(BATCH_SIZE)
                self._learn(data=batch)

    def _compute_loss_q(self, data):
        state, action, reward, next_state, done = data
        q = self.critic_network(state, action)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.critic_target(next_state, self.actor_target(next_state))
            backup = reward + GAMMA * (1 - done) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
        return loss_q

    def _compute_loss_pi(self, data):
        state = data[0]
        q_pi = self.critic_network(state, self.actor_network(state))
        return -q_pi.mean()

    def _learn(self, data):
        # First run one gradient descent step for Q.
        self.critic_optimizer.zero_grad()
        loss_q = self._compute_loss_q(data)
        loss_q.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic_network.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
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
