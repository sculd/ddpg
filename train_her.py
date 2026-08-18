"""DDPG + HER on FetchReach with selectable exploration noise.

    python train_her.py --noise white            # paper setup (i.i.d. Gaussian)
    python train_her.py --noise pink             # colored (pink) noise exploration
    python train_her.py --noise white --no-her   # plain DDPG, sparse reward
    python train_her.py --preset legacy --noise ou|pink   # old hyper-parameters

Epoch structure follows OpenAI baselines HER (single worker): 50 cycles of
[2 rollout episodes -> 40 gradient steps -> target update], then 10
deterministic evaluation episodes.
"""
import argparse
import csv
import os
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from her.agent import DDPGHerAgent
from her.replay_buffer import HerReplayBuffer

gym.register_envs(gymnasium_robotics)

PRESETS = {
    # baselines/HER defaults (Plappert et al. 2018, Table / her/experiment/config.py)
    'paper': dict(lr_actor=1e-3, lr_critic=1e-3, gamma=0.98, polyak=0.95, batch_size=256,
                  action_l2=1.0, noise_eps=0.2, random_eps=0.3, normalize=True, clip_return=True),
    # roughly the settings of the original train_her.py in this repo
    'legacy': dict(lr_actor=1e-5, lr_critic=1e-4, gamma=0.99, polyak=0.999, batch_size=128,
                   action_l2=0.0, noise_eps=0.2, random_eps=0.0, normalize=False, clip_return=False),
}


def run_episode(env, agent, T, explore):
    obs_l, ag_l, g_l, a_l = [], [], [], []
    state, _ = env.reset()
    o, ag, g = state['observation'], state['achieved_goal'], state['desired_goal']
    agent.reset_episode()
    reached = 0.0
    for t in range(T):
        a = agent.act(o, g, explore=explore)
        state, r, term, trunc, info = env.step(a)
        obs_l.append(o); ag_l.append(ag); g_l.append(g); a_l.append(a)
        o, ag = state['observation'], state['achieved_goal']
        reached = max(reached, float(info['is_success']))
    obs_l.append(o); ag_l.append(ag)
    final_success = float(info['is_success'])
    final_dist = float(np.linalg.norm(ag - g))
    return (np.array(obs_l), np.array(ag_l), np.array(g_l), np.array(a_l)), final_success, reached, final_dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='FetchReach-v4')
    p.add_argument('--noise', default='white', choices=['white', 'pink', 'red', 'ou'])
    p.add_argument('--no-her', action='store_true')
    p.add_argument('--preset', default='paper', choices=list(PRESETS))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--cycles', type=int, default=50)
    p.add_argument('--episodes-per-cycle', type=int, default=2)
    p.add_argument('--batches-per-cycle', type=int, default=40)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default=None)
    p.add_argument('--out-dir', default='exp_her')
    p.add_argument('--stop-at', type=float, default=None, help='stop when eval success >= this')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='override a preset hyper-parameter, e.g. --set polyak=0.95 --set normalize=True')
    args = p.parse_args()
    hparams = dict(PRESETS[args.preset])
    for kv in args.set:
        k, v = kv.split('=')
        assert k in hparams, f'unknown hyper-parameter {k}; choose from {list(hparams)}'
        hparams[k] = type(hparams[k])(eval(v))

    tag = args.tag or f"{args.preset}_{args.noise}_{'noher' if args.no_her else 'her'}"
    if args.set and args.tag is None:
        tag += '_' + '_'.join(kv.replace('=', '') for kv in args.set)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    env = gym.make(args.env, max_episode_steps=50)
    env.reset(seed=args.seed); env.action_space.seed(args.seed)
    T = env.spec.max_episode_steps
    s, _ = env.reset()
    obs_dim, goal_dim, action_dim = s['observation'].shape[0], s['desired_goal'].shape[0], env.action_space.shape[0]

    agent = DDPGHerAgent(obs_dim, goal_dim, action_dim, T, noise_kind=args.noise,
                         seed=args.seed, **hparams)
    buffer = HerReplayBuffer(int(1e6) // T, T, obs_dim, goal_dim, action_dim,
                             reward_fn=env.unwrapped.compute_reward, replay_k=4, use_her=not args.no_her)

    csv_path = os.path.join(args.out_dir, f'{tag}_seed{args.seed}.csv')
    ckpt_path = os.path.join('checkpoints', f'her_{tag}_seed{args.seed}.pt')
    print(f'[{tag} seed={args.seed}] logging to {csv_path}')
    t0 = time.time()
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'env_steps', 'train_success', 'train_reached', 'eval_success',
                    'eval_reached', 'eval_dist', 'critic_loss', 'actor_loss', 'wall_s'])
        env_steps = 0
        for epoch in range(args.epochs):
            tr_s, tr_r, losses = [], [], []
            for _ in range(args.cycles):
                for _ in range(args.episodes_per_cycle):
                    ep, fs, reached, _ = run_episode(env, agent, T, explore=True)
                    buffer.store_episode(*ep)
                    agent.update_normalizer(ep[0], ep[1])
                    env_steps += T
                    tr_s.append(fs); tr_r.append(reached)
                for _ in range(args.batches_per_cycle):
                    losses.append(agent.update(buffer))
                agent.soft_update()
            ev_s, ev_r, ev_d = [], [], []
            for _ in range(args.eval_episodes):
                _, fs, reached, dist = run_episode(env, agent, T, explore=False)
                ev_s.append(fs); ev_r.append(reached); ev_d.append(dist)
            cl, al = np.mean(losses, axis=0)
            row = [epoch, env_steps, np.mean(tr_s), np.mean(tr_r), np.mean(ev_s), np.mean(ev_r),
                   np.mean(ev_d), cl, al, time.time() - t0]
            w.writerow([f'{x:.4f}' if isinstance(x, float) else x for x in row]); f.flush()
            print(f'[{tag} s{args.seed}] epoch {epoch:3d} steps {env_steps:7d} | train succ {np.mean(tr_s):.2f} '
                  f'reached {np.mean(tr_r):.2f} | eval succ {np.mean(ev_s):.2f} dist {np.mean(ev_d):.3f} '
                  f'| Lc {cl:.4f} La {al:.3f} | {time.time()-t0:.0f}s', flush=True)
            agent.save(ckpt_path)
            if args.stop_at is not None and np.mean(ev_s) >= args.stop_at:
                break
    env.close()


if __name__ == '__main__':
    main()
