import random

import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        # Handle both single transitions and batched transitions
        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)

        def _add_single(s, a, r, ns, d):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (s, a, r, ns, not d)
            self.position = int((self.position + 1) % self.capacity)

        # Check if this is a batch of transitions
        # For batch: state has shape (batch_size, obs_dim)
        # For single: state has shape (obs_dim,)
        if state.ndim > 1 and len(state.shape) > 1:
            # Batched add
            batch_size = state.shape[0]
            for i in range(batch_size):
                _add_single(state[i], action[i], reward[i], next_state[i], done[i])
        else:
            # Single transition
            _add_single(state, action, reward, next_state, done)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, not_done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, not_done

    def __len__(self):
        return len(self.buffer)
