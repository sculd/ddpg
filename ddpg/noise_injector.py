import numpy as np

# This noise is adopeted from openai codebase.
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Additionally `_sigma_decay` is added.


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, sigma_final=0.2, sigma_decay=0.001, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.sigma_initial = sigma
        self.sigma_final = sigma_final
        self.sigma_decay = sigma_decay
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def decay_sigma(self):
        self.sigma += (self.sigma_final - self.sigma) * self.sigma_decay

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.sigma = self.sigma_initial
        print("reset the noise")

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma}, theta={self.theta})'
