import numpy as np
import torch
import torch.nn.functional as F
from model import PolicyNetwork, VNetwork
import torch.optim as optim


LR_ACTOR = 0.0005
LR_CRITIC = 0.0005
GAMMA = 0.99


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.log_prob = None

        self.actor_network = PolicyNetwork(state_size, action_size)
        self.critic_network = VNetwork(state_size)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LR_CRITIC)

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float()
        probabilities = F.softmax(self.actor_network(state))

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).unsqueeze(0).float()
        next_state = torch.from_numpy(next_state).unsqueeze(0).float()

        v_current = self.critic_network(state)
        with torch.no_grad():
            v_target = reward + GAMMA * self.critic_network(next_state) * (1 - int(done))

        self.critic_optimizer.zero_grad()
        critic_loss = (v_target - v_current)**2
        critic_loss.backward()
        self.critic_optimizer.step()

        advantage = (v_target - v_current).detach()
        self.actor_optimizer.zero_grad()
        actor_loss = -(advantage * self.log_prob)
        actor_loss.backward()
        self.actor_optimizer.step()
