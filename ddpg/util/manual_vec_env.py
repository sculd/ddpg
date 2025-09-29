from typing import Callable, Iterable, List, Sequence

import gymnasium as gym
import numpy as np


class ManualResetVectorEnv:
    """Vector environment that only resets workers when explicitly asked."""

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]):
        if not env_fns:
            raise ValueError("env_fns must contain at least one environment constructor")
        self.envs: List[gym.Env] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        self._last_obs: List[np.ndarray] = [None] * self.num_envs
        self._terminated = np.zeros(self.num_envs, dtype=bool)
        self._truncated = np.zeros(self.num_envs, dtype=bool)

    def reset(self):
        observations = []
        infos = []
        for idx, env in enumerate(self.envs):
            obs, info = env.reset()
            self._last_obs[idx] = obs
            self._terminated[idx] = False
            self._truncated[idx] = False
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(self, actions: np.ndarray):
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = []
        for idx, (env, action) in enumerate(zip(self.envs, actions)):
            if self._terminated[idx] or self._truncated[idx]:
                # worker is waiting for manual reset; keep last observation and flag termination
                observations.append(self._last_obs[idx])
                rewards.append(0.0)
                terminated.append(self._terminated[idx])
                truncated.append(self._truncated[idx])
                infos.append({})
                continue

            obs, reward, term, trunc, info = env.step(action)
            self._last_obs[idx] = obs
            self._terminated[idx] = term
            self._truncated[idx] = trunc

            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)

        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )

    def reset_done(self, mask: Iterable[bool]):
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (self.num_envs,):
            raise ValueError("mask length must match number of environments")

        observations = []
        infos = []
        for idx, flag in enumerate(mask_array):
            if flag:
                obs, info = self.envs[idx].reset()
                self._last_obs[idx] = obs
                self._terminated[idx] = False
                self._truncated[idx] = False
                observations.append(obs)
                infos.append(info)
            else:
                observations.append(self._last_obs[idx])
                infos.append({})
        return np.stack(observations), infos

    def close(self):
        for env in self.envs:
            env.close()


def make_manual_vector_env(env_id: str, num_envs: int, **env_kwargs) -> ManualResetVectorEnv:
    env_fns = [lambda env_id=env_id, env_kwargs=env_kwargs: gym.make(env_id, **env_kwargs) for _ in range(num_envs)]
    return ManualResetVectorEnv(env_fns)
