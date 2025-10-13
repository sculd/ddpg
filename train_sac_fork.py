#!/usr/bin/env python3
import os
import time

import hydra
import numpy as np
import torch

import sac.utils
from sac.logger import Logger
from sac_fork.agent import SAC_FORK
from sac_fork.replay_memory import ReplayMemory

_checkpoint_file_format = 'checkpoints/sac_fork_{env}.pt'


class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        log_dir = os.path.join(self.work_dir, f'tb_sac_{cfg.env}')
        print(f'workspace: {self.work_dir}, log_dir: {log_dir}')

        self.env = env
        self.cfg = cfg
        self.num_envs = getattr(cfg, 'num_envs', 1)
        self.is_vectorized = self.num_envs > 1

        self.logger = Logger(log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        sac.utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Get action_space based on vectorized or single env
        if self.is_vectorized:
            action_space = self.env.single_action_space
        else:
            action_space = self.env.action_space

        self.agent = SAC_FORK(
            obs_dim=cfg.agent.obs_dim,
            action_space=action_space,
            action_range=cfg.agent.action_range,
            device=cfg.agent.device,
            args=cfg.agent.args
        )

        self.replay_buffer = ReplayMemory(int(cfg.replay_buffer_capacity), cfg.seed)

        self.step = 0

    def run(self):
        if self.is_vectorized:
            self._run_vectorized()
        else:
            print(f"run vectorized")

    def _run_vectorized(self):
        """Vectorized environment training loop"""
        episode = 0
        episode_rewards = np.zeros(self.num_envs)
        episode_steps = np.zeros(self.num_envs, dtype=int)
        max_episode_reward = -float('inf')
        start_time = time.time()
        num_updates_per_step = getattr(self.cfg, 'num_updates_per_step', 1)

        # Initialize all environments
        obs, _ = self.env.reset()
        self.agent.reset()

        while self.step < self.cfg.num_train_steps:
            # run training updates (multiple updates per step as num_envs samples are collected per step)
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(num_updates_per_step):
                    self.agent.update(self.replay_buffer, self.cfg.batch_size, self.logger, self.step)

            # Update obs bounds for all observations
            self.agent.obs_upper_bound = max(self.agent.obs_upper_bound, np.amax(obs))
            self.agent.obs_lower_bound = min(self.agent.obs_lower_bound, np.amin(obs))

            # Sample actions for all environments (batched)
            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.single_action_space.sample() for _ in range(self.num_envs)])
            else:
                with sac.utils.eval_mode(self.agent):
                    # Pass all observations at once (batched operation)
                    action = self.agent.act(obs)

            # Step all environments
            next_obs, rewards, dones, _, _ = self.env.step(action)

            # Check for episode termination (including max steps)
            terminated = dones | (episode_steps >= self.cfg.max_episode_steps)

            # Store all transitions at once (batched)
            self.replay_buffer.add(obs, action, rewards, next_obs, terminated.astype(np.float32))

            # Update episode stats
            episode_rewards += rewards
            episode_steps += 1

            if np.any(terminated):
                for i in np.where(terminated)[0]:
                    if self.step < self.cfg.num_seed_steps:
                        print(f"Episode {episode} (env {i}) completed at step {self.step}, reward: {episode_rewards[i]:.2f}")

                    episode += 1
                    self.logger.log('train/episode_reward', episode_rewards[i], self.step)
                    self.logger.log('train/episode', episode, self.step)

                    if episode % self.cfg.eval_frequency == 0:
                        if episode_rewards[i] > self.cfg.target_score:
                            self.agent.save(os.path.join(self.work_dir, _checkpoint_file_format.format(env=self.cfg.env)))

                    if episode_rewards[i] >= max_episode_reward:
                        print(f"Episode {episode}, env_i: {i}, reward: {episode_rewards[i]} winning against {max_episode_reward=}")
                        self.agent.save(os.path.join(self.work_dir, _checkpoint_file_format.format(env=self.cfg.env)))

                    max_episode_reward = max(max_episode_reward, episode_rewards[i])
                    episode_rewards[i] = 0
                    episode_steps[i] = 0

            obs = next_obs
            self.step += self.num_envs

            # Log duration periodically
            if self.step % self.cfg.log_frequency < self.num_envs:
                self.logger.log('train/duration', time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

@hydra.main(version_base=None, config_path="configs_sac_fork", config_name="train_mountain_car.yaml")
def main(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg)
    workspace = Workspace(env, cfg)
    workspace.run()


if __name__ == '__main__':
    main()
