from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import Actor, Critic
from cpprb import ReplayBuffer
import os


class HER_TD3Agent:
    def __init__(self, obs_dim, action_dim, goal_dim, action_bounds, compute_reward_func, env_name, k_future=4, buffer_size=65536, actor_lr=1e-3, critic_lr=1e-3, batch_size=64, gamma=0.99, tau=0.05, policy_noise=0.05, noise_clip=0.1, policy_freq=4, train_mode=True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.action_bounds = action_bounds
        self.compute_reward_func = compute_reward_func
        self.k_future = k_future
        self.env_name = env_name
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.train_mode = train_mode

        self.actor = Actor(obs_dim, action_dim, goal_dim, action_bounds)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(obs_dim, action_dim, goal_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "goal": {"shape": self.goal_dim},
            "done": {}
        })
        self.total_it = 0
        self.exec_dict = {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "desired_goal": [],
            "next_achieved_goal": []
        }

    def act(self, obs, goal):
        with torch.no_grad():
            x = np.concatenate([obs, goal])
            action = self.actor(torch.Tensor(x)).numpy()

        if self.train_mode:
            action += max(self.action_bounds['high']) / 5 * np.random.randn(self.action_dim)
            action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])

            random_actions = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, reward, next_obs, desired_goal, next_achieved_goal, done):
        self.exec_dict["obs"].append(obs)
        self.exec_dict["action"].append(action)
        self.exec_dict["reward"].append(reward)
        self.exec_dict["next_obs"].append(next_obs)
        self.exec_dict["desired_goal"].append(desired_goal)
        self.exec_dict["next_achieved_goal"].append(next_achieved_goal)

        if done:
            self._store(self.exec_dict)
            for key in self.exec_dict:
                self.exec_dict[key] = []

            for _ in range(20):
                self._learn()

    def save(self):
        os.makedirs(f"saved_networks/her_td3/{self.env_name}", exist_ok=True)
        torch.save(self.actor.state_dict(), f"saved_networks/her_td3/{self.env_name}/actor.pt")

    def load(self):
        self.actor.load_state_dict(torch.load(f"saved_networks/her_td3/{self.env_name}/actor.pt"))

    def _store(self, episode_dict):
        episode_len = len(episode_dict['obs'])
        for t in range(episode_len):
            obs = episode_dict['obs'][t]
            action = episode_dict['action'][t]
            reward = episode_dict['reward'][t]
            next_obs = episode_dict['next_obs'][t]
            goal = episode_dict['desired_goal'][t]
            next_achieved = episode_dict['next_achieved_goal'][t]
            done = reward + 1

            self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, goal=goal, done=done)

            if episode_len > 1:
                for _ in range(self.k_future):
                    future_idx = np.random.randint(low=t, high=episode_len)
                    new_goal = episode_dict['next_achieved_goal'][future_idx]
                    new_reward = self.compute_reward_func(next_achieved, new_goal, None)
                    done = new_reward + 1
                    self.rb.add(obs=obs, action=action, reward=new_reward, next_obs=next_obs, goal=new_goal, done=done)

        self.rb.on_episode_end()

    def _learn(self):
        self.total_it += 1
        sample = self.rb.sample(self.batch_size)

        input = torch.Tensor(np.concatenate([sample['obs'], sample['goal']], axis=1))
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_input = torch.Tensor(np.concatenate([sample['next_obs'], sample['goal']], axis=1))
        done = torch.Tensor(sample['done']).long()

        Q_current1, Q_current2 = self.critic(input, action)
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_input) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            Q1_target_next, Q2_target_next = self.critic_target(next_input, next_actions)
            Q_target_next = torch.min(Q1_target_next, Q2_target_next)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

        critic_loss = F.mse_loss(Q_current1, Q_target) + F.mse_loss(Q_current2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic(input, self.actor(input))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for t_params, e_params in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_params.data.copy_(self.tau * e_params.data + (1 - self.tau) * t_params.data)

            for t_params, e_params in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_params.data.copy_(self.tau * e_params.data + (1 - self.tau) * t_params.data)
