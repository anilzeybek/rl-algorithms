import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer

from models import Actor, Critic

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common.normalizer import Normalizer



class HER_TD3Agent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 goal_dim,
                 action_bounds,
                 compute_reward_func,
                 env_name,
                 expl_noise,
                 start_timesteps,
                 k_future,
                 buffer_size,
                 actor_lr,
                 critic_lr,
                 batch_size,
                 gamma,
                 tau,
                 policy_noise,
                 noise_clip,
                 policy_freq):

        self.max_action = max(action_bounds["high"])

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.action_bounds = action_bounds
        self.compute_reward_func = compute_reward_func
        self.env_name = env_name
        self.expl_noise = expl_noise
        self.start_timesteps = start_timesteps
        self.k_future = k_future
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.policy_freq = policy_freq

        self.actor = Actor(obs_dim+goal_dim, action_dim, self.max_action)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(obs_dim+goal_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        self.normalizer = Normalizer(self.obs_dim+self.goal_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "goal": {"shape": self.goal_dim}
        })
        self.exec_dict = {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_achieved_goal": []
        }
        self.t = 0

    def act(self, obs, goal, train_mode=True):
        with torch.no_grad():
            input = self.normalizer.normalize(np.concatenate([obs, goal]))
            if not train_mode:
                action = self.actor(torch.Tensor(input)).numpy()
            else:
                if self.t < self.start_timesteps:
                    action = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
                else:
                    action = (
                        self.actor(torch.Tensor(input)).numpy()
                        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                    )

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, env_dict, action, reward, next_env_dict, done):
        self.t += 1

        self.exec_dict["obs"].append(env_dict["observation"])
        self.exec_dict["action"].append(action)
        self.exec_dict["reward"].append(reward)
        self.exec_dict["next_obs"].append(next_env_dict["observation"])
        self.exec_dict["achieved_goal"].append(env_dict["achieved_goal"])
        self.exec_dict["desired_goal"].append(env_dict["desired_goal"])
        self.exec_dict["next_achieved_goal"].append(next_env_dict["achieved_goal"])

        if done:
            self._store(self.exec_dict)
            for key in self.exec_dict:
                self.exec_dict[key] = []

        if self.t >= self.start_timesteps:
            self._learn()

    def save(self):
        os.makedirs(f"checkpoints/her_td3/{self.env_name}", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "normalizer_mean": self.normalizer.mean,
                    "normalizer_std": self.normalizer.std,
                    "normalizer_running_sum": self.normalizer.running_sum,
                    "normalizer_running_sumsq": self.normalizer.running_sumsq,
                    "normalizer_running_count": self.normalizer.running_count,
                    "t": self.t
                    }, f"checkpoints/her_td3/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"checkpoints/her_td3/{self.env_name}/actor_critic.pt")

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target = deepcopy(self.actor)

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = deepcopy(self.critic)

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.std = checkpoint["normalizer_std"]
        self.normalizer.running_sum = checkpoint["normalizer_running_sum"]
        self.normalizer.running_sumsq = checkpoint["normalizer_running_sumsq"]
        self.normalizer.running_count = checkpoint["normalizer_running_count"]

        self.t = checkpoint["t"]

    def _store(self, episode_dict):
        inputs_to_normalize = []

        episode_len = len(episode_dict['obs'])
        for t in range(episode_len):
            obs = episode_dict['obs'][t]
            action = episode_dict['action'][t]
            reward = episode_dict['reward'][t]
            next_obs = episode_dict['next_obs'][t]
            goal = episode_dict['desired_goal'][t]
            next_achieved = episode_dict['next_achieved_goal'][t]

            self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, goal=goal)
            inputs_to_normalize.append(np.concatenate([obs, goal]))

            if episode_len > 1:
                for _ in range(self.k_future):
                    future_idx = np.random.randint(low=t, high=episode_len)

                    new_goal = episode_dict['achieved_goal'][future_idx]
                    new_reward = self.compute_reward_func(next_achieved, new_goal, None)

                    self.rb.add(obs=obs, action=action, reward=new_reward, next_obs=next_obs, goal=new_goal)
                    inputs_to_normalize.append(np.concatenate([obs, new_goal]))

        self.normalizer.update(np.array(inputs_to_normalize))
        self.rb.on_episode_end()

    def _learn(self):
        sample = self.rb.sample(self.batch_size)
        input = torch.Tensor(np.concatenate([sample['obs'], sample['goal']], axis=1))
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_input = torch.Tensor(np.concatenate([sample['next_obs'], sample['goal']], axis=1))

        input = self.normalizer.normalize(input).float()
        next_input = self.normalizer.normalize(next_input).float()

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
            Q_target = reward + self.gamma * Q_target_next
            Q_target = torch.clamp(Q_target, -1 / (1 - self.gamma), 0)

        critic_loss = F.mse_loss(Q_current1, Q_target) + F.mse_loss(Q_current2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.t % self.policy_freq == 0:
            a = self.actor(input)
            actor_loss = -self.critic(input, a)[0].mean()
            actor_loss += a.pow(2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
