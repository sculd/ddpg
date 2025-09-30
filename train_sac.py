#!/usr/bin/env python3
import os
import time

import hydra
import torch
import numpy as np

import sac.utils
from sac.logger import Logger
from sac.replay_buffer import ReplayBuffer


class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.env = env
        self.cfg = cfg
        self.num_envs = getattr(cfg, 'num_envs', 1)
        self.is_vectorized = self.num_envs > 1

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        sac.utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        # Get observation and action space shapes
        if self.is_vectorized:
            obs_shape = self.env.single_observation_space.shape
            action_shape = self.env.single_action_space.shape
        else:
            obs_shape = self.env.observation_space.shape
            action_shape = self.env.action_space.shape

        self.replay_buffer = ReplayBuffer(obs_shape,
                                          action_shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.step = 0

    def run(self):
        if self.is_vectorized:
            self._run_vectorized()
        else:
            self._run_single()

    def _run_single(self):
        """Original single environment training loop"""
        episode, episode_reward, max_episode_reward, done = 0, 0, 0, True
        episode_step = 0
        start_time = time.time()
        num_updates_per_step = getattr(self.cfg, 'num_updates_per_step', 1)

        while self.step < self.cfg.num_train_steps:
            if done or episode_step >= self.cfg.max_episode_steps:
                if self.step < self.cfg.num_seed_steps:
                    print(f"Episode {episode} completed at step {self.step}")
                self.logger.log('train/duration', time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if (episode + 1) % self.cfg.eval_frequency == 0:
                    if max_episode_reward < self.cfg.target_score:
                        self.agent.save(os.path.join(self.work_dir, 'checkpoints/sac.pt'))

                self.logger.log('train/episode_reward', episode_reward, self.step)

                max_episode_reward = max(max_episode_reward, episode_reward)
                if episode_reward > self.cfg.target_score:
                    self.agent.save(os.path.join(self.work_dir, 'checkpoints/sac.pt'))

                obs, _ = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # run training updates (multiple updates per step)
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(num_updates_per_step):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with sac.utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            next_obs, reward, done, _, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_step += 1
            self.step += 1

    def _run_vectorized(self):
        """Vectorized environment training loop"""
        episode = 0
        episode_rewards = np.zeros(self.num_envs)
        episode_steps = np.zeros(self.num_envs, dtype=int)
        max_episode_reward = 0
        start_time = time.time()
        num_updates_per_step = getattr(self.cfg, 'num_updates_per_step', 1)

        # Initialize all environments
        obs, _ = self.env.reset()
        self.agent.reset()

        while self.step < self.cfg.num_train_steps:
            # Run training updates (multiple updates per step)
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(num_updates_per_step):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            # Sample actions for all environments (batched)
            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.single_action_space.sample() for _ in range(self.num_envs)])
            else:
                with sac.utils.eval_mode(self.agent):
                    # Pass all observations at once (batched operation)
                    action = self.agent.act(obs, sample=True)

            # Step all environments
            next_obs, rewards, dones, _, _ = self.env.step(action)
            # Check for episode termination (including max steps)
            terminated = dones | (episode_steps >= self.cfg.max_episode_steps)
            self.replay_buffer.add(obs, action, rewards, next_obs, terminated.astype(np.float32))

            # Update episode stats
            episode_rewards += rewards
            episode_steps += 1

            if np.any(terminated):
                for i in np.where(terminated)[0]:
                    if self.step < self.cfg.num_seed_steps:
                        print(f"Episode {episode} (env {i}) completed at step {self.step}, reward: {episode_rewards[i]:.2f}")

                    self.logger.log('train/episode_reward', episode_rewards[i], self.step)

                    max_episode_reward = max(max_episode_reward, episode_rewards[i])
                    if episode_rewards[i] > self.cfg.target_score:
                        self.agent.save(os.path.join(self.work_dir, 'checkpoints/sac.pt'))

                    episode += 1
                    self.logger.log('train/episode', episode, self.step)

                    if episode % self.cfg.eval_frequency == 0:
                        if max_episode_reward < self.cfg.target_score:
                            self.agent.save(os.path.join(self.work_dir, 'checkpoints/sac.pt'))

                    episode_rewards[i] = 0
                    episode_steps[i] = 0

            obs = next_obs
            self.step += self.num_envs

            # Log duration periodically
            if self.step % self.cfg.log_frequency < self.num_envs:
                self.logger.log('train/duration', time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

@hydra.main(version_base=None, config_path="configs_sac", config_name="train.yaml")
def main(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg)
    workspace = Workspace(env, cfg)
    workspace.run()


if __name__ == '__main__':
    main()
