import numpy as np
import random
import torch
import pickle
import os
from collections import deque, namedtuple
import ddpg.util.device

# Use the same device as defined in util/device.py
device = ddpg.util.device.device

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
ExperienceWithGoal = namedtuple("ExperienceWithGoal", field_names=["state", "goal", "action", "reward", "next_state", "done"])

class ReplayBuffer:

    def __init__(self, env_name, capacity):
        self.env_name = env_name
        self.capacity = capacity

        self.memory = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done, goal=None):
        if goal is None:
            experience = Experience(state, action, reward, next_state, done)
        else:
            experience = ExperienceWithGoal(state, goal, action, reward, next_state, done)
        self.memory.append(experience)

    def reset(self):
        self.memory.clear()

    def total_count(self):
        return len(self.memory)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences])).float().to(device)
        if len(experiences) > 0 and isinstance(experiences[0], ExperienceWithGoal):
            goals = torch.from_numpy(np.vstack([experience.goal for experience in experiences])).float().to(device)
        else:
            goals = None
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences])).float().to(device)        
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences])).float().to(device)        
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences])).float().to(device)  
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences]).astype(np.uint8)).float().to(device)        

        return (states, goals, actions, rewards, next_states, dones)

    def extend(self, other_buffer):
        self.memory.extend(other_buffer.memory)

    def save(self, path=None):
        if path is None:
            path = f"snapshot/{self.env_name}_replay_buffer.pkl"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the memory buffer
        with open(path, 'wb') as f:
            pickle.dump({
                'memory': list(self.memory),
                'capacity': self.capacity,
                'env_name': self.env_name
            }, f)
        print(f"Replay buffer saved ({path})")

    def load(self, path=None):
        if path is None:
            path = f"snapshot/{self.env_name}_replay_buffer.pkl"

        if not os.path.exists(path):
            print(f"Replay buffer file ({path}) does not exist.")
            return

        # Load the memory buffer
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.memory = deque(data['memory'], maxlen=self.capacity)
            # Verify capacity and env_name match
            if data['capacity'] != self.capacity:
                print(f"Warning: Loaded buffer capacity ({data['capacity']}) differs from current capacity ({self.capacity})")
            if data['env_name'] != self.env_name:
                print(f"Warning: Loaded buffer env_name ({data['env_name']}) differs from current env_name ({self.env_name})")

        print(f"Replay buffer loaded ({path}), {len(self.memory)} experiences")
