import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dueling_model import DuelingNetwork
from replay_buffer import ReplayBuffer


BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
SYNC_TARGET_EVERY = 1000
LR = 1e-3
UPDATE_EVERY = 4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


class DuelingAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.dueling_network = DuelingNetwork(state_size, action_size)
        self.target_network = DuelingNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.dueling_network.parameters(), lr=LR)

        self.eps = EPS_START
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0
        self.learn_count = 0

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0)
            action_values = self.dueling_network(state)
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self._learn(experiences)

        if done:
            self._update_eps()

    def _update_eps(self):
        self.eps = max(EPS_END, EPS_DECAY * self.eps)

    def _learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_current = self.dueling_network(states).gather(1, actions)

        with torch.no_grad():
            a = self.dueling_network(next_states).argmax(1).unsqueeze(1)
            Q_target_next = self.target_network(next_states).gather(1, a)
            Q_target = rewards + GAMMA * Q_target_next * (1 - dones)

        loss = F.mse_loss(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % SYNC_TARGET_EVERY == 0:
            self.target_network.load_state_dict(self.dueling_network.state_dict())
