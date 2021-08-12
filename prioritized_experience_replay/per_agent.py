import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from prioritized_replay_buffer import PrioritizedReplayBuffer


BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
SYNC_TARGET_EVERY = 1000
UPDATE_PRIORITIES_EVERY = 20
LR = 1e-3
UPDATE_EVERY = 4
EPS_START = 1.0  # EPS as in the epsilon-greedy
EPS_END = 0.01
EPS_DECAY = 0.995
PRIORITIZATION_FACTOR = 0.5


class PERAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.policy_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

        self.eps = EPS_START
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, PRIORITIZATION_FACTOR)
        self.t_step = 0
        self.update_step = 0
        self.learn_count = 0

        self.max_priority = 1

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0)
            action_values = self.policy_network(state)
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done, self.max_priority)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self._learn(experiences)

        self.update_step = (self.update_step + 1) % UPDATE_PRIORITIES_EVERY
        if self.update_step == 0:
            self.memory.update_probabilities()

        if done:
            self._update_eps()

    def _update_eps(self):
        self.eps = max(EPS_END, EPS_DECAY * self.eps)

    def _learn(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        Q_current = self.policy_network(states).gather(1, actions)

        a = self.policy_network(next_states).argmax(1).unsqueeze(1)
        Q_target_next = self.target_network(next_states).gather(1, a)
        Q_target = rewards + GAMMA * Q_target_next * (1 - dones)

        loss = F.mse_loss(Q_current, Q_target)
        # TODO: insert importance sampling here (change 'loss')

        for idx, e in enumerate(experiences):
            new_priority = (Q_target[idx] - Q_current[idx]).abs().detach().numpy()[0]
            e.update_priority(new_priority)

            self.max_priority = max(self.max_priority, new_priority)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % SYNC_TARGET_EVERY == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
