"""DDPG + HER (Andrychowicz et al. 2017; hyperparameters follow the OpenAI
baselines HER implementation used in Plappert et al. 2018) with pluggable
exploration noise: white (i.i.d. Gaussian, the paper's choice), pink / red
(colored noise, Eberhard et al. 2023), or OU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from her.normalizer import Normalizer
from sac.noise import ColoredNoiseProcess
from ddpg.noise_injector import OrnsteinUhlenbeckActionNoise


def mlp(in_dim, out_dim, hidden=256, depth=3, out_act=None):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ExplorationNoise:
    """eps ~ unit-variance sequence; kind in {white, pink, red, ou}."""

    def __init__(self, kind, action_dim, T, seed=None):
        self.kind = kind
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        beta = {'white': 0.0, 'pink': 1.0, 'red': 2.0}.get(kind)
        if kind == 'ou':
            # theta=0.15, dt=1 -> stationary std sigma/sqrt(2*theta) ~ 1.83*sigma
            # so scale to unit variance
            self._ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=1.0,
                                                    theta=0.15, dt=1.0, seed=seed)
            self._ou_scale = np.sqrt(2 * 0.15)
        elif beta is not None:
            self._cn = ColoredNoiseProcess(beta, action_dim, T, seed=seed)
        else:
            raise ValueError(kind)

    def reset(self):
        if self.kind == 'ou':
            self._ou.reset()
        else:
            self._cn.reset()

    def sample(self):
        if self.kind == 'ou':
            return self._ou() * self._ou_scale
        return self._cn.sample()


class DDPGHerAgent:
    def __init__(self, obs_dim, goal_dim, action_dim, T,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.98, polyak=0.95,
                 hidden=256, batch_size=256, action_l2=1.0,
                 noise_kind='white', noise_eps=0.2, random_eps=0.3,
                 normalize=True, clip_return=True, seed=None,
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim, self.goal_dim, self.action_dim = obs_dim, goal_dim, action_dim
        self.gamma, self.polyak, self.batch_size = gamma, polyak, batch_size
        self.action_l2, self.noise_eps, self.random_eps = action_l2, noise_eps, random_eps
        self.normalize, self.clip_return = normalize, clip_return
        self._rng = np.random.default_rng(seed)
        in_dim = obs_dim + goal_dim
        self.actor = mlp(in_dim, action_dim, hidden, out_act=nn.Tanh()).to(self.device)
        self.critic = mlp(in_dim + action_dim, 1, hidden).to(self.device)
        self.actor_target = mlp(in_dim, action_dim, hidden, out_act=nn.Tanh()).to(self.device)
        self.critic_target = mlp(in_dim + action_dim, 1, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.o_norm = Normalizer(obs_dim)
        self.g_norm = Normalizer(goal_dim)
        self.noise = ExplorationNoise(noise_kind, action_dim, T, seed=seed)

    # ---- preprocessing -------------------------------------------------
    def _preproc(self, o, g):
        if self.normalize:
            o, g = self.o_norm.normalize(o), self.g_norm.normalize(g)
        x = np.concatenate([o, g], axis=-1).astype(np.float32)
        return torch.as_tensor(x, device=self.device)

    def update_normalizer(self, obs, ag, g_relabelled_sampler=None):
        # baselines updates the normalizer on the (relabelled) episode; using
        # the raw episode obs/goals is a close and simpler approximation.
        self.o_norm.update(obs)
        self.g_norm.update(ag)

    # ---- acting -----------------------------------------------------------
    def reset_episode(self):
        self.noise.reset()

    def act(self, o, g, explore=True):
        with torch.no_grad():
            a = self.actor(self._preproc(o, g)[None])[0].cpu().numpy()
        if not explore:
            return a
        a = np.clip(a + self.noise_eps * self.noise.sample(), -1, 1)
        if self._rng.uniform() < self.random_eps:
            a = self._rng.uniform(-1, 1, size=self.action_dim).astype(np.float32)
        return a.astype(np.float32)

    # ---- learning ---------------------------------------------------------
    def update(self, buffer):
        o, g, a, r, o2 = buffer.sample(self.batch_size)
        x, x2 = self._preproc(o, g), self._preproc(o2, g)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        with torch.no_grad():
            a2 = self.actor_target(x2)
            q2 = self.critic_target(torch.cat([x2, a2], -1))
            target = r + self.gamma * q2
            if self.clip_return:
                target = target.clamp(-1.0 / (1.0 - self.gamma), 0.0)
        q = self.critic(torch.cat([x, a], -1))
        critic_loss = F.mse_loss(q, target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        pi = self.actor(x)
        actor_loss = -self.critic(torch.cat([x, pi], -1)).mean() + self.action_l2 * (pi ** 2).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        return critic_loss.item(), actor_loss.item()

    def soft_update(self):
        with torch.no_grad():
            for net, tgt in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(self.polyak).add_((1 - self.polyak) * p)

    # ---- io ----------------------------------------------------------------
    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                    'o_norm': self.o_norm.state_dict(), 'g_norm': self.g_norm.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(d['actor']); self.critic.load_state_dict(d['critic'])
        self.actor_target.load_state_dict(d['actor']); self.critic_target.load_state_dict(d['critic'])
        self.o_norm.load_state_dict(d['o_norm']); self.g_norm.load_state_dict(d['g_norm'])
