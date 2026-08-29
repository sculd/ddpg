"""Episode replay buffer that samples length-(horizon+1) observation windows
which never cross an episode boundary.

Layout: each real step stores (obs_t, a_t, r_t, term_t, ep). end_episode() then
appends one dummy row holding the terminal observation (so obs windows have a
final next-obs). A window of H+1 rows starting at s is valid iff all rows share
one episode id and rows s..s+H-1 are real (non-dummy); the last row may be the
dummy terminal-obs row. The ring-buffer write head is handled by the episode-id
check: any window mixing old and new data has mismatched ids."""
import numpy as np


class SeqReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, horizon):
        self.capacity = capacity
        self.horizon = horizon
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((capacity,), dtype=np.float32)
        self.terminated = np.empty((capacity,), dtype=np.float32)
        self.dummy = np.ones((capacity,), dtype=bool)
        self.ep_id = np.full((capacity,), -1, dtype=np.int64)
        self.idx = 0
        self.full = False
        self._ep = 0

    def _put(self, obs, action, reward, terminated, dummy):
        i = self.idx
        self.obs[i], self.action[i] = obs, action
        self.reward[i], self.terminated[i] = reward, terminated
        self.dummy[i], self.ep_id[i] = dummy, self._ep
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add(self, obs, action, reward, terminated):
        self._put(obs, action, reward, terminated, dummy=False)

    def end_episode(self, final_obs):
        self._put(final_obs, np.zeros_like(self.action[0]), 0.0, 0.0, dummy=True)
        self._ep += 1

    def __len__(self):
        return self.capacity if self.full else self.idx

    def _valid(self, cand):
        H = self.horizon
        ok = (cand >= 0) & (cand < len(self) - H)
        cand = np.where(ok, cand, 0)
        ok &= self.ep_id[cand] == self.ep_id[cand + H]
        for o in range(H):
            ok &= ~self.dummy[cand + o]
        return ok

    def sample(self, batch_size, end_frac=0.25):
        """obs (H+1,B,obs), action (H,B,act), reward (H,B), terminated (H,B).

        Uniform sampling over window *starts* under-represents each episode's
        final transitions ~H-fold (a tail row appears in few valid windows),
        which starves terminal-only rewards. end_frac of the batch is therefore
        drawn from episode-end-aligned windows (last real row = the episode's
        final transition); the official DreamerV3 gets the same coverage by
        letting sequences cross episode boundaries."""
        n, H = len(self), self.horizon
        starts = np.empty(batch_size, dtype=np.int64)
        k = 0
        n_end = int(batch_size * end_frac)
        if n_end > 0:
            ends = np.flatnonzero(self.dummy[:n]) - H     # start s: row s+H is dummy
            if len(ends):
                cand = ends[self._valid(ends)]
                if len(cand):
                    take = min(n_end, batch_size)
                    starts[:take] = np.random.choice(cand, size=take)
                    k = take
        while k < batch_size:
            cand = np.random.randint(0, n - H, size=4 * (batch_size - k))
            cand = cand[self._valid(cand)]
            take = min(len(cand), batch_size - k)
            starts[k:k + take] = cand[:take]
            k += take
        idx = starts[None, :] + np.arange(H + 1)[:, None]   # (H+1, B)
        return (self.obs[idx], self.action[idx[:-1]],
                self.reward[idx[:-1]], self.terminated[idx[:-1]])
