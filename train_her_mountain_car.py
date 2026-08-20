"""DDPG + HER on goal-conditioned MountainCarContinuous (ddpg.goal_mountain_car).

    python train_her_mountain_car.py                 # pink-noise exploration (default)
    python train_her_mountain_car.py --noise white   # i.i.d. Gaussian
    python train_her_mountain_car.py --noise ou      # the OU process used elsewhere
    python train_her_mountain_car.py --no-her        # relabelling off

Same structure as train_her.py; what differs is what the MountainCar experiment
showed to be necessary: 200-step episodes, full-scale exploration noise, and NO
action-L2 penalty (action_l2=1.0 re-creates the "do nothing" optimum this env is
famous for - with it every configuration fails).
"""
import argparse
from collections import deque

import gymnasium as gym
import numpy as np

import envs.goal_mountain_car_env  # registers GoalMountainCar-v0
from ddpg.agent_her import AgentHer as Agent
from sac.noise import ColoredNoiseProcess


class ColoredActionNoise:
    """sac.noise colored sequence behind the OU-noise interface Agent expects."""

    def __init__(self, beta, action_dim, seq_len, sigma=1.0, seed=None):
        self.sigma = sigma
        self._process = ColoredNoiseProcess(beta, action_dim, seq_len, seed=seed)

    def sample(self, env_indices):
        return np.stack([self._process.sample() * self.sigma
                         for _ in np.atleast_1d(env_indices)], axis=0)

    def reset(self, env_indices=None, reset_sigma=True):
        self._process.reset()

    def set_sigma(self, new_sigma):
        self.sigma = float(new_sigma)

    def reset_sigma(self):
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--noise', default='pink', choices=['white', 'pink', 'ou'])
parser.add_argument('--no-her', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=8)
args = parser.parse_args()

env_name = 'GoalMountainCar-v0'
MAX_STEPS = 200
EPISODES_PER_EPOCH = 800
NOISE_SIGMA = 1.0  # full-scale exploration; 0.2 never leaves the valley

env = gym.make(env_name, max_episode_steps=MAX_STEPS)
env.reset(seed=args.seed)
env = env.unwrapped
np.random.seed(args.seed)

noise = None  # ou: keep the agent's default OU process
if args.noise in ('white', 'pink'):
    beta = {'white': 0.0, 'pink': 1.0}[args.noise]
    noise = ColoredActionNoise(beta, env.action_space.shape[0], MAX_STEPS,
                               sigma=NOISE_SIGMA, seed=args.seed)

agent = Agent(n_inputs=env.observation_space['observation'].shape[0],
              n_actions=env.action_space.shape[0],
              env_name=env_name,
              env=env,
              goal_dim=env.observation_space['desired_goal'].shape[0],
              noise_sigma=NOISE_SIGMA,
              toggle_sigma_decay=False,
              lr_actor=0.001,
              lr_critic=0.001,
              tau=0.05,
              batch_size=256,
              action_l2=0.0,  # NOT the Fetch value; see module docstring
              noise=noise)

best_success_rate = 0.0
for epoch in range(args.epochs):
    print(f"\nStarting Epoch {epoch+1}/{args.epochs}")
    success_window = deque(maxlen=EPISODES_PER_EPOCH)
    distance_window = deque(maxlen=EPISODES_PER_EPOCH)

    for episode in range(1, EPISODES_PER_EPOCH + 1):
        done, truncated = False, False
        success = False
        state, _ = env.reset()
        agent.reset_episode_goals()
        agent.noise.reset()  # fresh colored-noise sequence (or OU state) per episode
        obs = state['observation']
        desired_goal = state['desired_goal']

        for t in range(MAX_STEPS):
            if done or truncated:
                break
            act = agent.choose_action(obs, desired_goal)
            next_state, reward, done, truncated, info = env.step(act)
            next_obs, achieved_goal = next_state['observation'], next_state['achieved_goal']

            agent.add_to_memory(obs, achieved_goal, desired_goal, act, reward, next_obs, int(done))
            obs = next_obs

            distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
            if distance_to_goal < 0.05:
                success = True
                break

        if not args.no_her:
            agent.add_her_batch_to_memory()
        agent.learn_episode(t+1)

        success_window.append(float(success))
        distance_window.append(distance_to_goal)

        if episode % 20 == 0:
            print(f'Epoch {epoch+1}, Episode {episode}/{EPISODES_PER_EPOCH}, '
                  f'Success Rate: {np.mean(success_window):.2%}, Avg Distance: {np.mean(distance_window):.4f}', end="\r")

    epoch_success_rate = np.mean(success_window)
    print(f"\nEpoch {epoch+1} completed. Success rate: {epoch_success_rate:.2%}, "
          f"Average distance: {np.mean(distance_window):.4f}")
    if epoch_success_rate > best_success_rate:
        print(f'New best success rate: {epoch_success_rate:.2%}')
        best_success_rate = epoch_success_rate
        agent.save()

print(f"\nTraining completed. Best success rate: {best_success_rate:.2%}")
env.close()
