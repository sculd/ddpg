"""Running mean/std normalizer with clipping (as in baselines HER)."""
import numpy as np


class Normalizer:
    def __init__(self, size, eps=1e-2, clip=5.0):
        self.size = size
        self.eps = eps
        self.clip = clip
        self.sum = np.zeros(size, dtype=np.float64)
        self.sumsq = np.zeros(size, dtype=np.float64)
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float32)
        self.std = np.ones(size, dtype=np.float32)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.size)
        self.sum += x.sum(axis=0)
        self.sumsq += (x ** 2).sum(axis=0)
        self.count += x.shape[0]
        self.mean = (self.sum / self.count).astype(np.float32)
        var = self.sumsq / self.count - (self.sum / self.count) ** 2
        self.std = np.sqrt(np.maximum(self.eps ** 2, var)).astype(np.float32)

    def normalize(self, x):
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)

    def state_dict(self):
        return {'sum': self.sum, 'sumsq': self.sumsq, 'count': self.count,
                'mean': self.mean, 'std': self.std}

    def load_state_dict(self, d):
        self.sum, self.sumsq, self.count = d['sum'], d['sumsq'], d['count']
        self.mean, self.std = d['mean'], d['std']
