import random
import numpy as np
import torch
import torch.nn.functional as F

from ddpg.network import ActorNetwork, CriticNetwork
from ddpg.replaybuffer import ReplayBuffer

from ddpg.agent import Agent

# Use cuda if available else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
UPDATE_EVERY = 2
N_HER_GOALS = 4

class AgentHer(Agent):
    def __init__(self,  
                 n_inputs, 
                 n_actions, 
                 env_name, 
                 env,
                 lr_actor=0.001, 
                 lr_critic=0.001,
                 tau=0.001, 
                 gamma=0.99, 
                 replay_buffer_size=10**6, 
                 layer1_size=400, 
                 layer2_size=300, 
                 batch_size=16, 
                 noise_sigma=0.5,
                 noise_sigma_final=0.2,
                 noise_sigma_decay=0.001,
                 num_envs=1,
                 noise_seed=None,
                 toggle_sigma_decay=True,
                 ):
        super(AgentHer, self).__init__(
            n_inputs+3, # state + goal
            n_actions, 
            env_name, 
            lr_actor=lr_actor, 
            lr_critic=lr_critic,
            tau=tau, 
            gamma=gamma, 
            replay_buffer_size=replay_buffer_size, 
            layer1_size=layer1_size, 
            layer2_size=layer2_size, 
            batch_size=batch_size,
            noise_sigma=noise_sigma,
            noise_sigma_final=noise_sigma_final,
            noise_sigma_decay=noise_sigma_decay,
            num_envs=num_envs,
            noise_seed=noise_seed,
        )
        self.env = env
        self.memory_her = ReplayBuffer(replay_buffer_size)
        self.reset_episode_goals()

    def reset_episode_goals(self):
        self.episode = []
        self.achieved_goals = []

    def choose_action(self, state, goal, with_noise=True):
        input = torch.concat((torch.from_numpy(state), torch.from_numpy(goal)), dim=0)
        return super().choose_action(input, with_noise=with_noise)

    def learn_online(self):
        # Online learning if we have enough experiences
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0 and self.memory.total_count() > self.batch_size:
            sampled_experiences = self.memory.sample(self.batch_size)
            self.learn(sampled_experiences)

    def _sample_goals(self, i, strategy = 'final'):
        if strategy == 'final':
            return [self.achieved_goals[i]]

        # Sample future goals for HER (k=4 or 8 as in the paper)
        if strategy == 'episode':
            n_her_goals = min(N_HER_GOALS, len(self.achieved_goals))
            if n_her_goals <= 0:
                return []
            return list(random.sample(self.achieved_goals, n_her_goals))
        
        if strategy == 'future':
            n_her_goals = min(N_HER_GOALS, len(self.achieved_goals)-i)
            if n_her_goals <= 0:
                return []
            return list(random.sample(self.achieved_goals[i:], n_her_goals))

        raise ValueError(f"Invalid strategy: {strategy}")

    def add_to_memory(self, state, achieved_goal, desired_goal, action, reward, next_state, done):
        # Store original experience with desired goal
        self.memory.push(state, action, reward, next_state, done, goal=desired_goal)
        self.episode.append((state, achieved_goal, desired_goal, action, reward, next_state, done))
        self.achieved_goals.append(achieved_goal)

    def add_her_batch_to_memory(self):
        # HER: Use achieved goals as additional goals
        memory_her = ReplayBuffer(len(self.episode))
        her_rewards = []
        for i, (state, achieved_goal, desired_goal, action, reward, next_state, done) in enumerate(self.episode):
            gs = self._sample_goals(i, strategy = 'future')
            for g in gs:
                r = self.env.compute_reward(achieved_goal, g, {})
                memory_her.push(state, action, r, next_state, done, goal=g)
                her_rewards.append(r+1)

        self.memory.extend(memory_her)
        return np.sum(her_rewards)

    def learn_episode(self, timestamps):
        # Additional learning at episode end
        timestamps = min(timestamps, self.memory.total_count())
        if self.memory.total_count() < self.batch_size:
            return

        for t in range(timestamps):
            if t % UPDATE_EVERY == 0:
                sampled_experiences = self.memory.sample(self.batch_size)
                self.learn(sampled_experiences)

    def set_testing_mode(self):
        """Configure the agent for testing/evaluation"""
        self.actor.eval()  # Set network to evaluation mode
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        self.noise_sigma = 0.0  # Disable exploration noise
        self.toggle_sigma_decay = False  # Disable noise decay
