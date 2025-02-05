import numpy as np
import random
import torch
from collections import deque, namedtuple

# Use cuda if available else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.total_count = 0

        self.memory = deque(maxlen=self.capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def push(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        self.total_count += 1

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences])).float().to(device)        
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences])).float().to(device)        
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences])).float().to(device)  
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences]).astype(np.uint8)).float().to(device)        
        
        return (states, actions, rewards, next_states, dones)
    