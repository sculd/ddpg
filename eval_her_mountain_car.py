"""Evaluate the saved GoalMountainCar HER checkpoint; optionally record a gif.

    python eval_her_mountain_car.py [--episodes 20] [--gif images/animation_her_mountain_car.gif]
"""
import argparse

import gymnasium as gym
import numpy as np

import envs.goal_mountain_car_env  # registers GoalMountainCar-v0
from ddpg.agent_her import AgentHer

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--gif', default=None)
args = parser.parse_args()

env = gym.make('GoalMountainCar-v0', max_episode_steps=200,
               render_mode='rgb_array' if args.gif else None).unwrapped
agent = AgentHer(n_inputs=2, n_actions=1, env_name='GoalMountainCar-v0', env=env,
                 goal_dim=1, noise_sigma=0.0, toggle_sigma_decay=False)
agent.load(load_memory=False)
agent.set_testing_mode()

frames, succ, steps = [], [], []
for ep in range(args.episodes):
    s, _ = env.reset(seed=1000 + ep)
    hit = None
    for t in range(200):
        act = agent.choose_action(s['observation'], s['desired_goal'], with_noise=False)
        s, r, done, trunc, info = env.step(act)
        if args.gif and ep == 0:
            frames.append(env.render())
        if r == 0.0:
            hit = t + 1
            break
    succ.append(hit is not None)
    steps.append(hit or 200)
    print(f'ep {ep+1:2d} | success {hit is not None} | steps {hit or 200}')

print(f'\nsuccess {np.mean(succ):.0%} over {args.episodes} episodes, mean steps-to-goal {np.mean(steps):.0f}')
if args.gif and frames:
    import imageio
    imageio.mimsave(args.gif, frames[::2], fps=30, loop=0)
    print('gif saved:', args.gif)
env.close()
