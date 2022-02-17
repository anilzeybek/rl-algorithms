import torch
import torch.nn.functional as F
from models import Actor, Critic
import torch.optim as optim
import os


class A2CAgent:
    def __init__(self, obs_dim, action_dim, env_name, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        self.log_prob = None

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def act(self, obs):
        obs = torch.from_numpy(obs).float()
        probabilities = F.softmax(self.actor(obs), dim=0)

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def step(self, obs, action, reward, next_obs, done):
        obs = torch.from_numpy(obs).float()
        next_obs = torch.from_numpy(next_obs).float()

        V_current = self.critic(obs)
        with torch.no_grad():
            V_target = reward + self.gamma * self.critic(next_obs) * (1 - done)

        critic_loss = F.mse_loss(V_current, V_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            advantage = (V_target - V_current)

        actor_loss = -(advantage * self.log_prob)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self):
        os.makedirs(f"saved_networks/a2c/{self.env_name}", exist_ok=True)
        torch.save(self.actor.state_dict(), f"saved_networks/a2c/{self.env_name}/actor.pt")

    def load(self):
        self.actor.load_state_dict(torch.load(f"saved_networks/a2c/{self.env_name}/actor.pt"))
