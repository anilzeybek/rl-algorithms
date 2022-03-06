import torch
import torch.nn.functional as F
import torch.optim as optim
from model import PolicyNetwork
import os


class REINFORCEAgent:
    def __init__(self, obs_dim, action_dim, env_name, lr=1e-3, gamma=0.99):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.lr = lr
        self.gamma = gamma

        self.policy = PolicyNetwork(self.obs_dim, self.action_dim)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.reward_memory = []
        self.action_memory = []

    def act(self, obs):
        obs = torch.from_numpy(obs).float()
        probabilities = F.softmax(self.policy(obs), dim=0)

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        self.action_memory.append(action_probs.log_prob(action))
        return action.item()

    def step(self, obs, action, reward, next_obs, done):
        self.reward_memory.append(reward)

        if done:
            self._learn()

    def save(self):
        os.makedirs(f"saved_networks/reinforce/{self.env_name}", exist_ok=True)
        torch.save(self.policy.state_dict(), f"saved_networks/reinforce/{self.env_name}/policy.pt")

    def load(self):
        self.policy.load_state_dict(torch.load(f"saved_networks/reinforce/{self.env_name}/policy.pt"))

    def _learn(self):
        G = torch.zeros_like(torch.Tensor(self.reward_memory))
        for i, r in enumerate(reversed(self.reward_memory)):
            G[-(i+1)] = r + self.gamma * G[-i]

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            # We are subtracting because we want gradient ascent, not descent. But by default, pytorch does descent.
            loss -= g * logprob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
