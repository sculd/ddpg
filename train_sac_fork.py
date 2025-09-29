#!/usr/bin/env python3
import os
import time

import hydra
import torch

import sac.utils
from sac.logger import Logger
from sac_fork.agent import SAC_FORK
from sac_fork.replay_memory import ReplayMemory

_checkpoint_file = 'checkpoints/sac_fork.pt'

class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.env = env
        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        sac.utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.agent = SAC_FORK(
            obs_dim=cfg.agent.obs_dim,
            action_space=self.env.action_space,  # Pass the actual action_space object
            action_range=cfg.agent.action_range,
            device=cfg.agent.device,
            args=cfg.agent.args
        )
        
        self.replay_buffer = ReplayMemory(cfg.replay_buffer_capacity, cfg.seed)

        self.step = 0

    def run(self):
        episode, episode_reward, max_episode_reward, done = 0, 0, 0, True
        episode_step = 0
        start_time = time.time()
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
                        self.agent.save(os.path.join(self.work_dir, _checkpoint_file))

                self.logger.log('train/episode_reward', episode_reward, self.step)

                max_episode_reward = max(max_episode_reward, episode_reward)
                if episode_reward > self.cfg.target_score:
                    self.agent.save(os.path.join(self.work_dir, _checkpoint_file))

                obs, _ = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with sac.utils.eval_mode(self.agent):
                    action = self.agent.select_action(obs)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update_parameters(self.replay_buffer, self.cfg.batch_size, self.step)

            next_obs, reward, done, _, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done)


            obs = next_obs
            episode_step += 1
            self.step += 1

@hydra.main(version_base=None, config_path="configs_sac_fork", config_name="train.yaml")
def main(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg)
    workspace = Workspace(env, cfg)
    workspace.run()


if __name__ == '__main__':
    main()
