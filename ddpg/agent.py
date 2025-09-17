import numpy as np
import torch
import torch.nn.functional as F

from ddpg.network import ActorNetwork, CriticNetwork
from ddpg.noise_injector import OrnsteinUhlenbeckActionNoise
from ddpg.replaybuffer import ReplayBuffer
import ddpg.util.device

UPDATE_EVERY = 1


class Agent:
    def __init__(self,  
                 n_inputs, 
                 n_actions, 
                 env_name, 
                 lr_actor=1e-3, 
                 lr_critic=1e-3,
                 tau=0.001, 
                 gamma=0.99, 
                 replay_buffer_size=10**6, 
                 layer1_size=400, 
                 layer2_size=300, 
                 batch_size=16, 
                 noise_sigma=0.5,
                 noise_sigma_final=0.2,
                 noise_sigma_decay=0.001,
                 ):
        self.gamma = gamma
        self.tau = tau
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions), sigma=noise_sigma, sigma_final=noise_sigma_final, sigma_decay=noise_sigma_decay)
        self.memory = ReplayBuffer(env_name, replay_buffer_size)
        self.batch_size = batch_size

        self.actor = ActorNetwork(lr_actor, n_inputs, layer1_size, layer2_size, n_actions=n_actions, name=f'{env_name}_Actor').to(ddpg.util.device.device)
        self.target_actor = ActorNetwork(lr_actor, n_inputs, layer1_size, layer2_size, n_actions=n_actions, name=f'{env_name}_TargetActor').to(ddpg.util.device.device)
        self.critic = CriticNetwork(lr_critic, n_inputs, layer1_size, layer2_size, n_actions=n_actions, name=f'{env_name}_Critic').to(ddpg.util.device.device)
        self.target_critic = CriticNetwork(lr_critic, n_inputs, layer1_size, layer2_size, n_actions=n_actions, name=f'{env_name}_TargetCritic').to(ddpg.util.device.device)

        self.update_network_parameters(tau=1)
        self.timestep = 0

    def choose_action(self, observation, with_noise=True):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(ddpg.util.device.device)
        mu = self.actor(observation)
        if with_noise:
            mu = mu + torch.tensor(self.noise(), dtype=torch.float).to(ddpg.util.device.device)
            mu = torch.clip(mu, min=-1, max=+1)
        self.actor.train()
        return mu.cpu().detach().numpy()

    def step(self, state, action, reward, next_state, done):
        # Handle vectorized environments - store each transition separately
        if len(state.shape) > 1:  # Vectorized environment
            for i in range(state.shape[0]):
                self.memory.push(state[i], action[i], reward[i], next_state[i], done[i])
                self.timestep += 1
        else:  # Single environment
            self.memory.push(state, action, reward, next_state, done)
            self.timestep += 1

        if self.timestep % UPDATE_EVERY == 0 and self.memory.total_count() > self.batch_size:
            sampled_experiences = self.memory.sample(self.batch_size)
            self.learn(sampled_experiences)
    
    def learn(self, experiences):
        self.learn_critic(experiences)
        self._update_critic_network_parameters(self.tau)
        self.learn_actor(experiences)
        self._update_actor_network_parameters(self.tau)

    def learn_critic(self, experiences):
        states, goals, actions, rewards, next_states, dones = experiences

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        next_actions = self.target_actor.forward(next_states)
        next_critic_q = self.target_critic.forward(next_states, next_actions)
        q_target = rewards + (self.gamma * next_critic_q * (1 - dones))

        q = self.critic.forward(states, actions)

        # update network
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(q_target, q)
        critic_loss.backward()
        self.critic.optimizer.step()

    def learn_actor(self, experiences):
        states, goals, actions, rewards, next_states, dones = experiences

        self.critic.eval()  # freeze it for actor update
        self.actor.train()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        actor_q = self.critic.forward(states, mu)
        # negative sign to maximize q
        actor_loss = torch.mean(-actor_q)
        actor_loss.backward()
        self.actor.optimizer.step()

    def _update_target_network(self, tau, source_network, target_network):
        for source_parameters, target_parameters in zip(source_network.parameters(), target_network.parameters()):
            target_parameters.data.copy_(tau * source_parameters.data + (1.0 - tau) * target_parameters.data)

    def _update_critic_network_parameters(self, tau):
        self._update_target_network(tau, self.critic, self.target_critic)

    def _update_actor_network_parameters(self, tau):
        self._update_target_network(tau, self.actor, self.target_actor)

    def update_network_parameters(self, tau):
        self._update_target_network(tau, self.critic, self.target_critic)
        self._update_target_network(tau, self.actor, self.target_actor)

    def save(self):
        #'''
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.memory.save()
        #'''

    def load(self, load_memory=True):
        #'''
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
        if load_memory:
            self.memory.load()
        #'''