import numpy as np
import random
import torch
from collections import deque, namedtuple

# Use cuda if available else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
ExperienceWithGoal = namedtuple("ExperienceWithGoal", field_names=["state", "goal", "action", "reward", "next_state", "done"])

class ReplayBuffer:

    def __init__(self, capacity):
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
