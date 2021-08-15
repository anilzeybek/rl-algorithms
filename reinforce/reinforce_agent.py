import numpy as np
import torch
import torch.nn.functional as F
from model import PolicyNetwork
import torch.optim as optim


LR = 0.0005
GAMMA = 0.99


class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float()
        probabilities = F.softmax(self.policy(state))

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        self.action_memory.append(action_probs.log_prob(action))
        return action.item()

    def step(self, state, action, reward, next_state, done):
        self.reward_memory.append(reward)

        if done:
            self._learn()

    def _learn(self):
        G = torch.zeros_like(torch.tensor(self.reward_memory))
        for i, r in enumerate(reversed(self.reward_memory)):
            G[-(i+1)] = r + GAMMA * G[-i]

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            # We are subtracting because we want gradient ascent, not descent. But by default, pytorch does descent.
            loss -= g * logprob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
