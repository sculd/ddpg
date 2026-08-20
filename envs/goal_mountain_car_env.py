"""Goal-conditioned MountainCarContinuous, gymnasium-robotics style.

Wraps the classic env in the dict-observation API (observation / achieved_goal /
desired_goal, compute_reward, info['is_success']) so the HER training scripts run
on it unchanged. achieved_goal = car position (1-D); desired_goal = the flag (0.45);
reward = 0 if |position - g| <= threshold else -1 (the env's action penalty and
success bonus are discarded). Episodes never terminate early, matching the
fixed-T rollouts of the Fetch tasks.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


class GoalMountainCarEnv(gym.Env):
    def __init__(self, threshold=0.05, render_mode=None):
        self._env = gym.make('MountainCarContinuous-v0', render_mode=render_mode).unwrapped
        self.threshold = threshold
        self.action_space = self._env.action_space
        obs_space = self._env.observation_space
        goal_low, goal_high = obs_space.low[:1], obs_space.high[:1]
        self.observation_space = spaces.Dict({
            'observation': obs_space,
            'achieved_goal': spaces.Box(goal_low, goal_high, dtype=np.float64),
            'desired_goal': spaces.Box(goal_low, goal_high, dtype=np.float64),
        })
        self._goal = np.array([0.45])

    def _obs(self, obs):
        return {'observation': obs, 'achieved_goal': obs[:1].copy(), 'desired_goal': self._goal.copy()}

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._obs(obs), info

    def step(self, action):
        obs, _, _, _, info = self._env.step(action)
        achieved = obs[:1]
        reward = float(self.compute_reward(achieved, self._goal, info))
        info['is_success'] = reward == 0.0
        return self._obs(obs), reward, False, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(d > self.threshold).astype(np.float32)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


register(id='GoalMountainCar-v0', entry_point=GoalMountainCarEnv, max_episode_steps=200)
