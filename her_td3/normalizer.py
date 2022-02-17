import numpy as np


class Normalizer:
    def __init__(self, size, eps=1e-2):
        self.size = size
        self.eps = eps
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = 1
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # update the total stuff
        self.total_sum += v.sum(axis=0)
        self.total_sumsq += (np.square(v)).sum(axis=0)
        self.total_count += v.shape[0]
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v):
        return (v - self.mean) / self.std
