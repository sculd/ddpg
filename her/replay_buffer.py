"""Episodic replay buffer with HER 'future' relabelling done at sample time.

Storing whole episodes lets us relabel with a goal achieved *later in the same
episode* (the 'future' strategy of Andrychowicz et al. 2017), which is the
strategy the paper found best. replay_k=4 gives a relabel ratio of 0.8.
"""
import numpy as np


class HerReplayBuffer:
    def __init__(self, capacity_episodes, T, obs_dim, goal_dim, action_dim,
                 reward_fn, replay_k=4, use_her=True):
        self.T = T
        self.capacity = capacity_episodes
        self.reward_fn = reward_fn
        self.use_her = use_her
        self.future_p = 1.0 - 1.0 / (1.0 + replay_k) if use_her else 0.0
        self.obs = np.zeros((capacity_episodes, T + 1, obs_dim), dtype=np.float32)
        self.ag = np.zeros((capacity_episodes, T + 1, goal_dim), dtype=np.float32)
        self.g = np.zeros((capacity_episodes, T, goal_dim), dtype=np.float32)
        self.actions = np.zeros((capacity_episodes, T, action_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def store_episode(self, obs, ag, g, actions):
        i = self.ptr
        self.obs[i], self.ag[i], self.g[i], self.actions[i] = obs, ag, g, actions
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ep = np.random.randint(0, self.size, batch_size)
        t = np.random.randint(0, self.T, batch_size)
        obs = self.obs[ep, t]
        obs_next = self.obs[ep, t + 1]
        ag_next = self.ag[ep, t + 1]
        actions = self.actions[ep, t]
        g = self.g[ep, t].copy()

        if self.use_her:
            her_mask = np.random.uniform(size=batch_size) < self.future_p
            future_offset = (np.random.uniform(size=batch_size) * (self.T - t)).astype(int)
            future_t = t + 1 + future_offset
            g[her_mask] = self.ag[ep[her_mask], future_t[her_mask]]

        rewards = self.reward_fn(ag_next, g, {}).astype(np.float32).reshape(-1, 1)
        return obs, g, actions, rewards, obs_next
