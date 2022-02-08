import torch
import torch.nn.functional as F
from model import PolicyNetwork, VNetwork
import torch.optim as optim
import os


class A2CAgent:
    def __init__(self, obs_dim, action_dim, env_name,  actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        self.log_prob = None

        self.actor_network = PolicyNetwork(obs_dim, action_dim)
        self.critic_network = VNetwork(obs_dim)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

    def act(self, obs):
        obs = torch.from_numpy(obs).float()
        probabilities = F.softmax(self.actor_network(obs), dim=0)

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def step(self, obs, action, reward, next_obs, done):
        obs = torch.from_numpy(obs).float()
        next_obs = torch.from_numpy(next_obs).float()

        v_current = self.critic_network(obs)
        with torch.no_grad():
            v_target = reward + self.gamma * self.critic_network(next_obs) * (1 - int(done))

        self.critic_optimizer.zero_grad()
        critic_loss = (v_target - v_current)**2
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            advantage = (v_target - v_current)

        self.actor_optimizer.zero_grad()
        actor_loss = -(advantage * self.log_prob)
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self):
        os.makedirs(f"saved_networks/a2c/{self.env_name}", exist_ok=True)
        torch.save(self.actor_network.state_dict(), f"saved_networks/a2c/{self.env_name}/actor_network.pt")

    def load(self):
        self.actor_network.load_state_dict(torch.load(f"saved_networks/a2c/{self.env_name}/actor_network.pt"))
