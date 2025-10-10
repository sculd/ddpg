#!/usr/bin/env python3
import os

import hydra
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import sac.utils
from train_sac_fork import _checkpoint_file_format


class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.env = env
        self.cfg = cfg

        sac.utils.set_seed_everywhere(cfg.seed)
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        self.step = 0

    def evaluate(self):
        checkpoint_file = _checkpoint_file_format.format(env=self.cfg.env)
        if not os.path.exists(checkpoint_file):
            print(f"Checkpoint file {checkpoint_file} not found")
        else:
            print(f"Loading checkpoint from {checkpoint_file}")
            self.agent.load(checkpoint_file)

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
                    action = self.agent.act(obs, sample=True)
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                episode_step += 1
                if episode_step >= self.cfg.max_episode_steps:
                    break

            print(f"Episode {episode} reward: {episode_reward}")
            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        print(f"Average episode reward: {average_episode_reward}")

def main_with_cfg(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg, render_mode="rgb_array")

    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="video/",
        name_prefix=f"eval_sac_{cfg.env}",
        episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=cfg.num_eval_episodes)

    workspace = Workspace(env, cfg)
    workspace.evaluate()
    env.close()


@hydra.main(version_base=None, config_path="configs_sac", config_name="test_pendulum.yaml")
def main(cfg):
    main_with_cfg(cfg)

if __name__ == '__main__':
    main()
