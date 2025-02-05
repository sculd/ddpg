import numpy as np

# This noise is adopeted from openai codebase.
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Additionally `_sigma_decay` is added.

_sigma_decay = 0.001


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None, toggle_sigma_decay=True):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.toggle_sigma_decay = toggle_sigma_decay
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        if self.toggle_sigma_decay:
            self.sigma *= (1. - _sigma_decay)
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma}, theta={self.theta})'
