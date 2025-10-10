#!/usr/bin/env python3
import cProfile
import pstats
import os
import time
from contextlib import contextmanager

import hydra
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import sac.utils
from sac.logger import Logger
from sac.replay_buffer import ReplayBuffer

_checkpoint_file_format = 'checkpoints/sac_{env}.pt'


@contextmanager
def maybe_record_function(name, enabled=True):
    """Conditional profiling context manager with zero overhead when disabled."""
    if enabled:
        with record_function(name):
            yield
    else:
        yield


class Workspace(object):
    def __init__(self, env, cfg, torch_profiler=None):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.env = env
        self.cfg = cfg
        self.num_envs = getattr(cfg, 'num_envs', 1)
        self.is_vectorized = self.num_envs > 1
        self.torch_profiler = torch_profiler
        self.profiling_enabled = torch_profiler is not None

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
                with maybe_record_function("agent_update", self.profiling_enabled):
                    for _ in range(num_updates_per_step):
                        self.agent.update(self.replay_buffer, self.logger, self.step)

            # Sample actions for all environments (batched)
            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.single_action_space.sample() for _ in range(self.num_envs)])
            else:
                with sac.utils.eval_mode(self.agent):
                    with maybe_record_function("agent_act", self.profiling_enabled):
                        # Pass all observations at once (batched operation)
                        action = self.agent.act(obs, sample=True)

            # Step all environments
            with maybe_record_function("env_step", self.profiling_enabled):
                next_obs, rewards, dones, _, _ = self.env.step(action)
            # Check for episode termination (including max steps)
            terminated = dones | (episode_steps >= self.cfg.max_episode_steps)
            with maybe_record_function("replay_buffer_add", self.profiling_enabled):
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
                with maybe_record_function("logging", self.profiling_enabled):
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

            # PyTorch profiler step
            if self.torch_profiler is not None:
                self.torch_profiler.step()

def main_with_cfg(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg)

    # Check which profilers are enabled
    enable_cprofile = cfg.get('profile_cprofile', False)
    enable_torch_profile = cfg.get('profile_torch', False)

    # Setup output paths if profiling is enabled
    if enable_cprofile or enable_torch_profile:
        profile_dir = os.path.join(os.getcwd(), 'profiles')
        os.makedirs(profile_dir, exist_ok=True)

    # Setup cProfile
    if enable_cprofile:
        cprofile_output = os.path.join(profile_dir, 'training_profile.prof')
        print(f"\n{'='*60}")
        print(f"cProfile ENABLED")
        print(f"Output: {cprofile_output}")
        print(f"{'='*60}\n")
    else:
        cprofile_output = None

    # Setup PyTorch profiler
    if enable_torch_profile:
        torch_profile_dir = os.path.join(profile_dir, 'torch_profiler')
        print(f"\n{'='*60}")
        print(f"PyTorch Profiler ENABLED")
        print(f"Output: {torch_profile_dir}")
        print(f"{'='*60}\n")

        # Profile first 5 steps, then skip 5, then profile 5 more (to capture warmup and steady state)
        torch_profiler = profile(
            activities=[ProfilerActivity.CPU],  # CPU only to avoid CUPTI warnings
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    else:
        torch_profiler = None
        torch_profile_dir = None

    if not enable_cprofile and not enable_torch_profile:
        print(f"\n{'='*60}")
        print(f"Profiling DISABLED")
        print(f"Enable with: profile_cprofile=true or profile_torch=true")
        print(f"{'='*60}\n")

    # Create workspace with torch profiler
    workspace = Workspace(env, cfg, torch_profiler=torch_profiler)

    # Start cProfile if enabled
    if enable_cprofile:
        cprofiler = cProfile.Profile()
        cprofiler.enable()

    try:
        # Run with or without torch profiler context
        if enable_torch_profile:
            with torch_profiler:
                workspace.run()
        else:
            workspace.run()
    finally:
        # Stop and save cProfile if enabled
        if enable_cprofile:
            cprofiler.disable()
            cprofiler.dump_stats(cprofile_output)

            print(f"\n{'='*60}")
            print(f"cProfile complete!")
            print(f"\n--- Top 20 Time-Consuming Functions ---\n")

            # Print cProfile summary statistics
            stats = pstats.Stats(cprofiler)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(20)

            print(f"\n{'='*60}")
            print(f"To visualize with snakeviz:")
            print(f"  snakeviz {cprofile_output}")
            print(f"{'='*60}\n")

        # Print torch profiler info if enabled
        if enable_torch_profile:
            print(f"\n{'='*60}")
            print(f"PyTorch Profiler complete!")
            print(f"To visualize with TensorBoard:")
            print(f"  tensorboard --logdir={torch_profile_dir}")
            print(f"{'='*60}\n")


@hydra.main(version_base=None, config_path="configs_sac", config_name="train_pendulum.yaml")
def main(cfg):
    main_with_cfg(cfg)

if __name__ == '__main__':
    main()
