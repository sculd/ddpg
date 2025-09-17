import numpy as np

# This noise is adopted from the OpenAI codebase.
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

_SIGMA_DECAY_RATE = 0.001


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, sigma_final=0.2, theta=0.15, dt=1e-2, x0=None, seed=None):
        self.theta = theta
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = sigma
        self.sigma_initial = sigma
        self.sigma_final = sigma_final
        self.dt = dt
        self.x0 = x0
        self._rng = np.random.default_rng(seed)
        self.reset()

    def __call__(self):
        gaussian = self._rng.normal(size=self.mu.shape).astype(self.mu.dtype, copy=False)
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * gaussian
        self.x_prev = x
        return x

    def decay_sigma(self):
        self.sigma += (self.sigma_final - self.sigma) * _SIGMA_DECAY_RATE

    def reset(self, reset_sigma=True):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        if reset_sigma:
            self.sigma = self.sigma_initial

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma}, theta={self.theta})'


class VectorizedOrnsteinUhlenbeckActionNoise:
    """Maintain one OU process per environment index."""

    def __init__(self,
                 num_envs,
                 action_dim,
                 sigma=0.2,
                 sigma_final=0.2,
                 theta=0.15,
                 dt=1e-2,
                 x0=None,
                 seed=None):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self._noises = []
        for idx in range(num_envs):
            noise_seed = None if seed is None else seed + idx
            self._noises.append(
                OrnsteinUhlenbeckActionNoise(
                    mu=np.zeros(action_dim, dtype=np.float32),
                    sigma=sigma,
                    sigma_final=sigma_final,
                    theta=theta,
                    dt=dt,
                    x0=x0,
                    seed=noise_seed,
                )
            )

    def sample(self, env_indices):
        indices = np.atleast_1d(env_indices)
        return np.stack([self._noises[int(idx)]() for idx in indices], axis=0)

    def __call__(self, env_indices):
        return self.sample(env_indices)

    def reset(self, env_indices=None, reset_sigma=True):
        if env_indices is None:
            target_indices = range(self.num_envs)
        else:
            target_indices = np.atleast_1d(env_indices)
        for idx in target_indices:
            self._noises[int(idx)].reset(reset_sigma=reset_sigma)

    def decay_sigma(self):
        for noise in self._noises:
            noise.decay_sigma()

    def set_sigma(self, new_sigma):
        for noise in self._noises:
            noise.sigma = max(noise.sigma_final, float(new_sigma))

    def reset_sigma(self):
        for noise in self._noises:
            noise.sigma = noise.sigma_initial

    @property
    def sigma(self):
        if not self._noises:
            return 0.0
        return float(np.mean([noise.sigma for noise in self._noises]))
