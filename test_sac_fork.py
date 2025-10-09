#!/usr/bin/env python3
import os

import hydra
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import sac.utils
from sac_fork.agent import SAC_FORK
from train_sac_fork import _checkpoint_file_format


class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.env = env
        self.cfg = cfg

        sac.utils.set_seed_everywhere(cfg.seed)

        self.agent = SAC_FORK(
            obs_dim=cfg.agent.obs_dim,
            action_space=self.env.action_space,  # Pass the actual action_space object
            action_range=cfg.agent.action_range,
            device=torch.device(cfg.device),
            args=cfg.agent.args
        )

        self.step = 0

    def evaluate(self):
        self.agent.load(os.path.join(self.work_dir, _checkpoint_file_format.format(env=self.cfg.env)))
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            print(f"Evaluating episode {episode}")
            obs, _ = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with sac.utils.eval_mode(self.agent):
                    action = self.agent.act(obs, evaluate=False)
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                episode_step += 1
                if episode_step >= self.cfg.max_episode_steps:
                    break

            print(f"Episode {episode} reward: {episode_reward}")
            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        print(f"Average episode reward: {average_episode_reward}")

@hydra.main(version_base=None, config_path="configs_sac_fork", config_name="test.yaml")
def main(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg, render_mode="rgb_array")

    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="video/",
        name_prefix=f"eval_sacfork_{cfg.env}",
        episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=cfg.num_eval_episodes)

    workspace = Workspace(env, cfg)
    workspace.evaluate()
    env.close()


if __name__ == '__main__':
    main()
