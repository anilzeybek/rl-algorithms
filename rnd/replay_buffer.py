from collections import namedtuple, deque
from typing import Deque, Tuple
import numpy as np
import random
import torch


class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool

    def __init__(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, buffer_size: int) -> None:
        self.memory: Deque[Experience] = deque(maxlen=buffer_size)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)
