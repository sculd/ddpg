import torch

import sac.utils
from sac.agent import SACAgent
from sac.noise import VectorizedColoredNoiseProcess


class SACCNAgent(SACAgent):
    """SAC with colored-noise exploration ("Pink Noise Is All You Need",
    Eberhard et al., ICLR 2023).

    Identical to SAC except for rollout action sampling: instead of drawing
    i.i.d. eps ~ N(0, 1) each step, eps comes from a temporally correlated
    noise sequence with PSD ~ 1/f^noise_beta (noise_beta=1: pink noise). The
    correlated torques produce the coherent, momentum-building behavior that
    sparse-reward tasks like MountainCarContinuous require, while the learned
    policy and all update rules stay pure SAC.
    """

    def __init__(self, num_envs=1, noise_beta=1.0, noise_seq_len=1024,
                 noise_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.noise = VectorizedColoredNoiseProcess(
            num_envs=num_envs,
            beta=noise_beta,
            action_dim=kwargs['action_dim'],
            seq_len=noise_seq_len,
            seed=noise_seed,
        )

    def reset(self):
        self.noise.reset()

    def reset_noise(self, env_index):
        """Called by the training loop when env_index finishes an episode."""
        self.noise.reset(env_index)

    def act(self, obs, sample=False):
        # Deterministic (eval) path is unchanged
        if not sample:
            return super().act(obs, sample=False)

        obs = torch.FloatTensor(obs).to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        with torch.no_grad():
            dist = self.actor(obs)
            eps = torch.as_tensor(self.noise.sample(obs.shape[0]),
                                  device=self.device)
            # pre-tanh Gaussian with colored eps, squashed like SquashedNormal
            action = torch.tanh(dist.loc + dist.scale * eps)
        action = action.clamp(*self.action_range)

        assert action.ndim == 2  # (batch_size, action_dim)

        if squeeze_output:
            return sac.utils.to_np(action[0])
        else:
            return sac.utils.to_np(action)
