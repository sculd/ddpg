#!/usr/bin/env python3
import cProfile
import pstats
import os
import time
from contextlib import contextmanager

import hydra
import imageio
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import sac.utils
from sac.logger import Logger
from sac.replay_buffer import ReplayBuffer

_checkpoint_file_format = 'checkpoints/{agent}_{env}.pt'
_checkpoint_latest_file_format = 'checkpoints/{agent}_{env}_latest.pt'


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
        log_dir = os.path.join(self.work_dir, f'tb_sac_{cfg.env}')
        print(f'workspace: {self.work_dir}, log_dir: {log_dir}')

        self.env = env
        self.cfg = cfg
        self.num_envs = getattr(cfg, 'num_envs', 1)
        self.torch_profiler = torch_profiler
        self.profiling_enabled = torch_profiler is not None

        self.logger = Logger(log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        sac.utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        # Training always runs on a (possibly single-env) vector env
        obs_shape = self.env.single_observation_space.shape
        action_shape = self.env.single_action_space.shape

        self.replay_buffer = ReplayBuffer(obs_shape,
                                          action_shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        # Periodic rendering of a deterministic episode during training
        self.render_frequency = cfg.get('render_frequency', 0)
        if self.render_frequency:
            self.render_env = sac.utils.make_env(cfg, render_mode="rgb_array",
                                                 seed=cfg.seed + 10000)
            self.video_dir = os.path.join(self.work_dir, 'video', 'train')
            os.makedirs(self.video_dir, exist_ok=True)

        self.step = 0

    def render_episode(self, episode):
        """Roll out one deterministic episode with the current policy and save it as mp4."""
        obs, _ = self.render_env.reset()
        frames = [self.render_env.render()]
        episode_reward = 0.0
        done = False
        while not done:
            with sac.utils.eval_mode(self.agent):
                action = self.agent.act(obs, sample=False)
            obs, reward, terminated, truncated, _ = self.render_env.step(action)
            frames.append(self.render_env.render())
            episode_reward += reward
            done = terminated or truncated

        fps = self.render_env.metadata.get('render_fps', 30)
        path = os.path.join(
            self.video_dir,
            f'{self.cfg.agent.name}_{self.cfg.env}_ep{episode:05d}_step{self.step}_r{episode_reward:.0f}.mp4')
        imageio.mimsave(path, frames, fps=fps)
        print(f'Saved training video ({episode_reward=:.2f}) to {path}')

    def run(self):
        """Vectorized environment training loop"""
        episode = 0
        episode_rewards = np.zeros(self.num_envs)
        episode_steps = np.zeros(self.num_envs, dtype=int)
        max_episode_reward = -float('inf')
        start_time = time.time()
        num_updates_per_step = getattr(self.cfg, 'num_updates_per_step', 1)
        checkpoint_file = os.path.join(
            self.work_dir, _checkpoint_file_format.format(agent=self.cfg.agent.name, env=self.cfg.env))
        checkpoint_latest_file = os.path.join(
            self.work_dir, _checkpoint_latest_file_format.format(agent=self.cfg.agent.name, env=self.cfg.env))

        # Initialize all environments
        obs, _ = self.env.reset()
        self.agent.reset()
        # gymnasium NEXT_STEP autoreset: envs flagged here reset on the upcoming
        # step() call, whose returned transition is bookkeeping, not a real step
        autoreset = np.zeros(self.num_envs, dtype=bool)

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
                next_obs, rewards, terminations, truncations, _ = self.env.step(action)
            dones = terminations | truncations

            # store only real transitions; bootstrap through truncations (done = terminated only)
            store = ~autoreset
            with maybe_record_function("replay_buffer_add", self.profiling_enabled):
                if store.any():
                    self.replay_buffer.add(obs[store], action[store], rewards[store],
                                           next_obs[store], terminations[store].astype(np.float32))

            # Update episode stats
            episode_rewards[store] += rewards[store]
            episode_steps[store] += 1

            for i in np.where(dones & store)[0]:
                if self.step < self.cfg.num_seed_steps:
                    print(f"Episode {episode} (env {i}) completed at step {self.step}, reward: {episode_rewards[i]:.2f}")

                episode += 1
                self.logger.log('train/episode_reward', episode_rewards[i], self.step)
                self.logger.log('train/episode', episode, self.step)

                if episode % self.cfg.eval_frequency == 0:
                    if episode_rewards[i] > self.cfg.target_score:
                        self.agent.save(checkpoint_file)

                if episode_rewards[i] >= max_episode_reward:
                    print(f"Episode {episode}, env_i: {i}, reward: {episode_rewards[i]} winning against {max_episode_reward=}")
                    self.agent.save(checkpoint_file)

                max_episode_reward = max(max_episode_reward, episode_rewards[i])
                episode_rewards[i] = 0
                episode_steps[i] = 0

                # give exploration-noise processes a fresh sequence for the new episode
                if hasattr(self.agent, 'reset_noise'):
                    self.agent.reset_noise(i)

                if self.render_frequency and episode % self.render_frequency == 0 \
                        and self.step >= self.cfg.num_seed_steps:
                    self.render_episode(episode)

            autoreset = dones
            obs = next_obs
            self.step += self.num_envs

            # Log duration periodically
            if self.step % self.cfg.log_frequency < self.num_envs:
                with maybe_record_function("logging", self.profiling_enabled):
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))
                    if self.step > self.cfg.num_seed_steps:
                        self.agent.save(checkpoint_latest_file)

            # PyTorch profiler step
            if self.torch_profiler is not None:
                self.torch_profiler.step()

def main_with_cfg(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg, vectorize=True)

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
