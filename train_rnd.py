import os
import time

import hydra
import numpy as np
import torch

import sac.utils
from sac.logger import Logger
from rnd.agent import DQNRNDAgent
from sac_fork.replay_memory import ReplayMemory

_checkpoint_file_format = 'checkpoints/rnd_{env}.pt'



class Workspace(object):
    def __init__(self, env, cfg):
        self.work_dir = os.getcwd()
        log_dir = os.path.join(self.work_dir, f'tb_rnd_{cfg.env}')
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

        self.agent = DQNRNDAgent(
            action_range=cfg.agent.action_range,
            gamma=cfg.agent.gamma,
            qnet_cfg=cfg.agent.qnet_cfg,
            rnd_cfg=cfg.agent.rnd_cfg,
            device=cfg.device,
        )

        self.replay_buffer = ReplayMemory(int(cfg.replay_buffer_capacity), cfg.seed)

        self.step = 0

    def run(self):
        self._run_unvectorized()


    def run_episode(self):
        obs, _ = self.env.reset()
        sum_r = 0
        sum_tot_r = 0
        loses = []

        while True:
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high - self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
            obs = torch.Tensor(obs).unsqueeze(0)

            Q = self.qnet(obs)
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0, Q.shape[1], (1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q, dim=1)
            new_obs, reward, done, info = self.env.step((action.item()))
            sum_r = sum_r + reward
            reward_i = self.rnd.get_reward(obs).detach().clamp(-1.0, 1.0).item()
            combined_reward = reward + reward_i
            sum_tot_r += combined_reward
            
            self.replay_buffer.append([obs, action, combined_reward, new_obs, done])
            loss = self.update_model()
            loses.append(loss)
            obs = new_obs
            
            self.step_counter = self.step_counter + 1
            if (self.step_counter > self.update_target_step):
                self.qnet_target.load_state_dict(self.qnet.state_dict())
                self.step_counter = 0
                print('updated target model')
            if done:
                break
        
    def _run_unvectorized(self):
        """Single environment training loop"""
        episode = 0
        episode_reward = 0
        episode_step = 0
        max_episode_reward = -float('inf')
        start_time = time.time()
        num_updates_per_step = getattr(self.cfg, 'num_updates_per_step', 1)

        # Initialize environment
        obs, _ = self.env.reset()

        # For DQN with continuous action space, we discretize it
        # Create discrete action mappings for continuous space
        num_discrete_actions = self.cfg.agent.action_dim  # This is set to action_space.shape[0]
        # For MountainCarContinuous, we'll use 3 discrete actions: left, none, right
        discrete_actions = np.linspace(self.cfg.agent.action_range[0],
                                      self.cfg.agent.action_range[1],
                                      3)  # Use 3 discrete actions for mountain car

        while self.step < self.cfg.num_train_steps:
            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(num_updates_per_step):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            # Sample action using epsilon-greedy for DQN
            if self.step < self.cfg.num_seed_steps:
                action_continuous = self.env.action_space.sample()
            else:
                # Update epsilon
                self.agent.steps = self.step
                self.agent.eps = self.agent.epsi_low + (self.agent.epsi_high - self.agent.epsi_low) * (np.exp(-1.0 * self.agent.steps/self.agent.decay))

                # Get Q-values and select discrete action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    Q = self.agent.qnet(obs_tensor)

                if np.random.rand() < self.agent.eps:
                    action_idx = np.random.randint(0, len(discrete_actions))
                else:
                    action_idx = Q.argmax(dim=1).item()
                    action_idx = min(action_idx, len(discrete_actions) - 1)  # Clamp to valid range

                # Convert discrete action to continuous
                action_continuous = np.array([discrete_actions[action_idx]])

            # Step environment with continuous action
            next_obs, reward, done, _, _ = self.env.step(action_continuous)

            # Store the discrete action index for replay buffer
            action_idx = np.argmin(np.abs(discrete_actions - action_continuous[0]))

            # Check for episode termination (including max steps)
            terminated = done or (episode_step >= self.cfg.max_episode_steps)

            # Get intrinsic reward from RND
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                reward_intrinsic = self.agent.rnd.get_reward(obs_tensor).item()
            combined_reward = reward + reward_intrinsic

            # Store transition with discrete action index
            self.replay_buffer.add(obs, action_idx, combined_reward, next_obs, float(not terminated))

            # Update episode stats
            episode_reward += reward
            episode_step += 1

            if terminated:
                if self.step < self.cfg.num_seed_steps:
                    print(f"Episode {episode} completed at step {self.step}, reward: {episode_reward:.2f}")

                episode += 1
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/episode', episode, self.step)

                if self.step >= self.cfg.num_seed_steps:
                    if episode % self.cfg.eval_frequency == 0:
                        if episode_reward > self.cfg.target_score:
                            self.agent.save(os.path.join(self.work_dir, _checkpoint_file_format.format(env=self.cfg.env)))

                    if episode_reward >= max_episode_reward:
                        print(f"Episode {episode}, reward: {episode_reward} winning against {max_episode_reward=}")
                        self.agent.save(os.path.join(self.work_dir, _checkpoint_file_format.format(env=self.cfg.env)))

                max_episode_reward = max(max_episode_reward, episode_reward)
                episode_reward = 0
                episode_step = 0

                # Reset environment for next episode
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            self.step += 1

            # Update target network periodically
            self.agent.step_counter += 1
            if self.agent.step_counter > self.agent.update_target_step:
                self.agent.qnet_target.load_state_dict(self.agent.qnet.state_dict())
                self.agent.step_counter = 0
                if self.step % 10000 == 0:  # Print less frequently
                    print('Updated target model')

            # Log duration periodically
            if self.step % self.cfg.log_frequency == 0:
                self.logger.log('train/duration', time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))




def main_with_cfg(cfg):
    env, cfg = sac.utils.env_with_cfg(cfg)
    workspace = Workspace(env, cfg)
    workspace.run()


@hydra.main(version_base=None, config_path="configs_rnd", config_name="train_mountain_car.yaml")
def main(cfg):
    main_with_cfg(cfg)
    
if __name__ == '__main__':
    main()
