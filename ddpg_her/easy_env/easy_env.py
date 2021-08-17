import numpy as np


class EasyEnv:
    def __init__(self, length=10):
        self.length = length
        self.max_t = length*10
        self.reset()

    def reset(self):
        self.t = 0
        self.current_loc = np.random.uniform(low=-self.length/2, high=self.length/2)
        self.goal_loc = np.random.uniform(low=-self.length/2, high=self.length/2)
        return (self.current_loc, self.goal_loc)

    def step(self, action):
        self.t += 1

        action = np.clip(action, -1, 1)[0]
        reward = 0
        done = False

        self.current_loc += action
        if self._is_close(self.current_loc, self.goal_loc):
            reward += 1
            done = True
        elif self.t == self.max_t:
            done = True

        return (self.current_loc, self.goal_loc), reward, done, None

    def _is_close(self, loc1, loc2):
        return abs(loc1 - loc2) < 0.1
