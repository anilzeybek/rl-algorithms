import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from prioritized_replay_buffer import Memory


BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
SYNC_TARGET_EVERY = 1000
LR = 1e-3
UPDATE_EVERY = 4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
PRIORITIZATION_FACTOR = 0.5


class PERAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.Q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=LR)

        self.eps = EPS_START
        self.memory = Memory(BUFFER_SIZE)
        self.t_step = 0
        self.update_step = 0
        self.learn_count = 0

        self.max_priority = 1

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0)
            action_values = self.Q_network(state)
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        self._append_experience((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences, idxs, is_weights = self.memory.sample(BATCH_SIZE)
            self._learn(experiences, idxs, is_weights)

        if done:
            self._update_eps()

    def _append_experience(self, experience):
        state, action, reward, next_state, done = experience
        with torch.no_grad():
            curr = self.Q_network(torch.from_numpy(state))[action]

            target_next = self.target_network(torch.from_numpy(next_state)).max()
            target = reward + GAMMA * target_next * (1 - done)

            td_error = abs(curr - target)
            self.memory.add(td_error, experience)

    def _update_eps(self):
        self.eps = max(EPS_END, EPS_DECAY * self.eps)

    def _learn(self, experiences, idxs, is_weights):
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()

        Q_current = self.Q_network(states).gather(1, actions)
        with torch.no_grad():
            a = self.Q_network(next_states).argmax(1).unsqueeze(1)
            Q_target_next = self.target_network(next_states).gather(1, a)
            Q_target = rewards + GAMMA * Q_target_next * (1 - dones)

            errors = (Q_target - Q_current).abs().numpy()

        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(Q_current, Q_target)).mean()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % SYNC_TARGET_EVERY == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())
