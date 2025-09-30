import math
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn


def make_env(cfg, render_mode):
    """Helper function to create dm_control environment (using gym-dmc)"""
    env_id = cfg.env

    # DeepMind Control Suite case
    if "_" in env_id and not env_id.endswith("-v3"):
        import dmc2gym
        if dmc2gym is None:
            raise ImportError(
                "dmc2gym is required for DeepMind Control Suite environments. "
                "Install it or switch cfg.env to a Gymnasium environment."
            )

        if env_id == "ball_in_cup_catch":
            domain_name, task_name = "ball_in_cup", "catch"
        else:
            parts = env_id.split("_")
            domain_name, task_name = parts[0], "_".join(parts[1:])

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg.seed,
            visualize_reward=True,
        )
        env.seed(cfg.seed)

    # Gymnasium environments (e.g. BipedalWalker-v3)
    else:
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=cfg.seed) 

    # sanity check for SAC [-1,1] action bounds
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def env_with_cfg(cfg, render_mode=None):
    # Check if we should use vectorized environments
    num_envs = getattr(cfg, 'num_envs', 1)

    if num_envs > 1:
        # Create vectorized environment
        env = gym.vector.AsyncVectorEnv([
            lambda: make_env(cfg, render_mode=render_mode)
            for _ in range(num_envs)
        ])
        cfg.agent.obs_dim = env.single_observation_space.shape[0]
        cfg.agent.action_dim = env.single_action_space.shape[0]
        cfg.agent.action_range = [
            float(env.single_action_space.low.min()),
            float(env.single_action_space.high.max())
        ]
    else:
        # Single environment (original behavior)
        env = make_env(cfg, render_mode=render_mode)
        cfg.agent.obs_dim = env.observation_space.shape[0]
        cfg.agent.action_dim = env.action_space.shape[0]
        cfg.agent.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]

    return env, cfg


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.mode)
            model.train(mode=False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(mode=state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.mode)
            model.train(mode=True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(mode=state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

