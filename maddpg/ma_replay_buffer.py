from collections import namedtuple, deque
import numpy as np
import torch


class MAReplayBuffer:
    def __init__(self, buffer_size, n_agents):
        self.n_agents = n_agents

        self.memories = [deque(maxlen=buffer_size) for _ in range(self.n_agents)]
        self.experience = namedtuple("Experience", field_names=["obs", "action", "reward", "next_obs", "done"])

    def store_transition(self, observations, actions, rewards, next_observations, dones):
        for i in range(self.n_agents):
            exp = self.experience(observations[i], actions[i], rewards[i], next_observations[i], dones[i])
            self.memories[i].append(exp)

    def sample(self, batch_size):
        batch = np.random.randint(len(self.memories[0]), size=batch_size)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        for i in range(self.n_agents):
            experiences = [self.memories[i][j] for j in batch]

            observations.append(torch.from_numpy(np.vstack([e.obs for e in experiences if e is not None])).float())
            actions.append(torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float())
            rewards.append(torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float())
            next_observations.append(torch.from_numpy(np.vstack([e.next_obs for e in experiences if e is not None])).float())
            dones.append(torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float())

        return observations, actions, rewards, next_observations, dones
