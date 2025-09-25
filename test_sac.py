#!/usr/bin/env python3
import torch
import os

import hydra
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import sac.utils


class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.env = env
        self.cfg = cfg

        sac.utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        self.step = 0

    def evaluate(self):
        self.agent.load(os.path.join(self.work_dir, 'checkpoints/sac.pt'))
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with sac.utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward

            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        print(f"Average episode reward: {average_episode_reward}")
        self.env.close()

@hydra.main(version_base=None, config_path="configs_sac", config_name="train.yaml")
def main(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg, render_mode="rgb_array")

    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="video/",
        name_prefix="eval",
        episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=cfg.num_eval_episodes)

    workspace = Workspace(env, cfg)
    workspace.evaluate()


if __name__ == '__main__':
    main()
