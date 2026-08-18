"""Evaluate a trained DDPG+HER checkpoint on FetchReach.

    python test_her.py --ckpt checkpoints/her_paper_white_her_seed0.pt [--render]
"""
import argparse

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from her.agent import DDPGHerAgent

gym.register_envs(gymnasium_robotics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='FetchReach-v4')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--render', action='store_true')
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()

    env = gym.make(args.env, max_episode_steps=50, render_mode='human' if args.render else None)
    s, _ = env.reset(seed=args.seed)
    T = env.spec.max_episode_steps
    agent = DDPGHerAgent(s['observation'].shape[0], s['desired_goal'].shape[0], env.action_space.shape[0], T)
    agent.load(args.ckpt)

    succ, reached, dists, steps = [], [], [], []
    for ep in range(args.episodes):
        s, _ = env.reset()
        o, g = s['observation'], s['desired_goal']
        first_hit = None
        for t in range(T):
            s, r, term, trunc, info = env.step(agent.act(o, g, explore=False))
            o = s['observation']
            if info['is_success'] and first_hit is None:
                first_hit = t + 1
            if term or trunc:
                break
        succ.append(float(info['is_success']))
        reached.append(float(first_hit is not None))
        dists.append(float(np.linalg.norm(s['achieved_goal'] - g)))
        steps.append(first_hit if first_hit is not None else T)
        print(f'ep {ep+1:3d} | final success {int(succ[-1])} | first hit step {first_hit} | final dist {dists[-1]:.4f}')
    print('=' * 60)
    print(f'final-step success rate : {np.mean(succ):.1%}')
    print(f'reached-at-any-step rate: {np.mean(reached):.1%}')
    print(f'mean final distance     : {np.mean(dists):.4f}')
    print(f'mean steps to first hit : {np.mean(steps):.1f}')
    env.close()


if __name__ == '__main__':
    main()
