from model import QNetwork, PolicyNetwork
from copy import deepcopy
import torch
import torch.optim as optim


LR_ACTOR = 0.01
LR_CRITIC = 0.01
GAMMA = 0.99
POLYAK = 0.995


class Agent:
    def __init__(self, name, n_agents, state_dim, obs_dim, n_actions):
        self.name = name

        self.critic_network = QNetwork(state_dim, n_agents)
        self.critic_target = deepcopy(self.critic_network)

        self.actor_network = PolicyNetwork(obs_dim, n_actions)
        self.actor_target = deepcopy(self.actor_network)

        for p in self.critic_target.parameters():
            p.requires_grad = False

        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=LR_CRITIC)

    def choose_action(self, obs):
        with torch.no_grad():
            obs = torch.tensor([obs], dtype=torch.float)
            action = self.actor_network(obs)
            return int(action.item())

    def compute_loss_q(self, state, action, reward, next_state, done, all_agents_target_actor):
        Q_current = self.critic_network(state, action)
        with torch.no_grad():
            Q_target_next = self.critic_target(next_state, all_agents_target_actor)
            Q_target = reward + GAMMA * Q_target_next * (1 - done)

        loss = ((Q_current - Q_target) ** 2).mean()
        return loss

    def compute_loss_pi(self, state, all_agents_actor):
        Q = self.critic_network(state, all_agents_actor)
        return -torch.mean(Q)

    def update_network_parameters(self):
        with torch.no_grad():
            for p, p_targ in zip(self.actor_network.parameters(), self.actor_target.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)

            for p, p_targ in zip(self.critic_network.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)
