import numpy as np


class Normalizer:
    def __init__(self, size, use_mean=True, eps=1e-2):
        self.size = size
        self.use_mean = use_mean
        self.eps = eps

        self.running_sum = np.zeros(self.size, np.float32)
        self.running_sumsq = np.zeros(self.size, np.float32)
        self.running_count = 1

        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, x):
        x = x.reshape(-1, self.size)

        self.running_sum += x.sum(axis=0)
        self.running_sumsq += (np.square(x)).sum(axis=0)
        self.running_count += x.shape[0]

        self.mean = self.running_sum / self.running_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.running_sumsq / self.running_count) - np.square(
            self.running_sum / self.running_count)))

    # normalize the observation
    def normalize(self, x):
        if self.use_mean:
            return (x - self.mean) / self.std

        return x / self.std
