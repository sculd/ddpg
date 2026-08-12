"""Temporally correlated (colored) action noise for exploration.

Implements the exploration scheme of "Pink Noise Is All You Need" (Eberhard et
al., ICLR 2023): replace the i.i.d. standard-normal noise used to sample actions
with a sequence whose power spectral density follows 1/f^beta.

beta = 0 recovers white noise (standard SAC), beta = 1 is pink noise, beta = 2
is red/Brownian noise (OU-like). Each sequence has unit marginal variance, so it
is a drop-in replacement for eps ~ N(0, 1).
"""
import numpy as np


def powerlaw_psd_gaussian(beta, shape, rng):
    """Gaussian noise with power spectrum ~ 1/f^beta (Timmer & Koenig 1995).

    shape: (..., n_samples); the spectrum is shaped along the last axis.
    Returns an array of the given shape with (approximately) unit variance.
    """
    shape = tuple(shape)
    n_samples = shape[-1]

    f = np.fft.rfftfreq(n_samples)
    # Avoid a divergent DC component; treat it like the lowest nonzero frequency.
    s_scale = f.copy()
    s_scale[0] = s_scale[1]
    s_scale = s_scale ** (-beta / 2.0)

    # Theoretical std of the generated time series, for normalization
    w = s_scale[1:].copy()
    w[-1] *= (1 + (n_samples % 2)) / 2.0  # correct one-sided spectrum weight
    sigma = 2 * np.sqrt(np.sum(w ** 2)) / n_samples

    freq_shape = shape[:-1] + (len(f),)
    sr = rng.normal(scale=s_scale, size=freq_shape)
    si = rng.normal(scale=s_scale, size=freq_shape)

    # Real signal constraints on the DC and (for even n) Nyquist components
    if not (n_samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)
    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)

    y = np.fft.irfft(sr + 1j * si, n=n_samples, axis=-1) / sigma
    return y


class ColoredNoiseProcess:
    """Per-episode colored noise sequence, sampled one step at a time.

    Pre-generates a chunk of length seq_len with PSD ~ 1/f^beta for each action
    dimension; reset() (or exhaustion) draws a fresh chunk.
    """

    def __init__(self, beta, action_dim, seq_len, seed=None):
        self.beta = beta
        self.action_dim = action_dim
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self._buf = powerlaw_psd_gaussian(
            self.beta, (self.action_dim, self.seq_len), self._rng).astype(np.float32)
        self._idx = 0

    def sample(self):
        if self._idx >= self.seq_len:
            self.reset()
        eps = self._buf[:, self._idx]
        self._idx += 1
        return eps


class VectorizedColoredNoiseProcess:
    """One independent colored noise process per environment."""

    def __init__(self, num_envs, beta, action_dim, seq_len, seed=None):
        self.num_envs = num_envs
        self._processes = [
            ColoredNoiseProcess(beta, action_dim, seq_len,
                                seed=None if seed is None else seed + i)
            for i in range(num_envs)
        ]

    def sample(self, n=None):
        """Sample the next eps for the first n envs -> (n, action_dim)."""
        if n is None:
            n = self.num_envs
        return np.stack([p.sample() for p in self._processes[:n]], axis=0)

    def reset(self, env_index=None):
        if env_index is None:
            for p in self._processes:
                p.reset()
        else:
            self._processes[int(env_index)].reset()
