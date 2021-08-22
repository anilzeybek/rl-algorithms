import numpy as np
import gym
from gym import spaces


class BitFlippingEnv(gym.Env):
    def __init__(self, n_bits):
        super(BitFlippingEnv, self).__init__()

        self.action_space = spaces.Discrete(n_bits)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_bits*2,), dtype=np.uint8)

        self.n_bits = n_bits
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)
        self.t = 0

    def reset(self):
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)
        self.t = 0

        return np.concatenate((self.state, self.goal))

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        done = False
        reward = 0

        self.t += 1
        if np.array_equal(self.state, self.goal):
            done = True
            reward = 1
        elif self.t == self.n_bits:  # end of episode without success
            done = True

        return np.concatenate((self.state, self.goal)), reward, done, None
