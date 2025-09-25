import numpy as np
import torch
import torch.nn.functional as F
import os

import sac.utils

import hydra

import abc


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, is_train=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, **kwargs):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, is_train=True):
        # is_train False means eval mode
        self.is_train = is_train
        self.actor.train(is_train)
        self.critic.train(is_train)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        #   1. action.ndim == 2: Action must be a 2D tensor (batch_size × action_dim)
        #   2. action.shape[0] == 1: Batch size must be exactly 1
        assert action.ndim == 2 and action.shape[0] == 1
        return sac.utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            sac.utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def save(self, filepath, save_optimizers=False):
        """Save model checkpoints.

        Args:
            filepath: Path to save the checkpoint
            save_optimizers: If True, save optimizer states (larger file, can resume training)
                           If False, only save model weights (smaller file, for inference)
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.data,
        }

        if save_optimizers:
            checkpoint.update({
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict(),
            })

        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath} (optimizers={'included' if save_optimizers else 'excluded'})")

    def load(self, filepath, load_optimizers=False):
        """Load model checkpoints.

        Args:
            filepath: Path to load the checkpoint from
            load_optimizers: If True, load optimizer states (if available)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.log_alpha.data = checkpoint['log_alpha']

        if load_optimizers and 'actor_optimizer_state_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])
            print(f"Loaded checkpoint from {filepath} (with optimizers)")
        else:
            print(f"Loaded checkpoint from {filepath} (model only)")
