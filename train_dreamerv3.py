"""Train DreamerV3 (state-only) on a gymnasium env.

    python train_dreamerv3.py --env Pendulum-v1 --steps 30000
    python train_dreamerv3.py --env MountainCarContinuous-v0 --steps 300000
    python train_dreamerv3.py --env FetchReach-v4 --steps 50000     # goal dicts are flattened
    python train_dreamerv3.py --env GoalMountainCar-v0 --steps 300000

Logs epoch rows to exp_dreamerv3/<tag>.csv; checkpoint to checkpoints/dreamerv3_<tag>.pt.
"""
import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch

import envs.goal_mountain_car_env  # registers GoalMountainCar-v0
from dreamerv3.agent import DreamerV3

try:
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
except ImportError:
    pass


def flat_obs(state):
    if isinstance(state, dict):
        return np.concatenate([state['observation'], state['desired_goal']]).astype(np.float32)
    return np.asarray(state, dtype=np.float32)


def evaluate(env, agent, episodes):
    act_low, act_high = env.action_space.low, env.action_space.high
    rets, succ = [], []
    for _ in range(episodes):
        s, _ = env.reset()
        agent.reset_episode()
        ret, reached, done, trunc = 0.0, 0.0, False, False
        while not (done or trunc):
            a_unit = agent.act(flat_obs(s), eval_mode=True)
            a = (act_low + (a_unit + 1) / 2 * (act_high - act_low)).astype(np.float32)
            s, r, done, trunc, info = env.step(a)
            ret += r
            reached = max(reached, float(info.get('is_success', 0.0)))
        rets.append(ret)
        succ.append(reached)
    return float(np.mean(rets)), float(np.mean(succ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='Pendulum-v1')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seed-steps', type=int, default=None)
    p.add_argument('--update-every', type=int, default=1)
    p.add_argument('--seq-len', type=int, default=32)
    p.add_argument('--noise-beta', type=float, default=0.0,
                   help='colored collection-noise exponent: 0 white, 1 pink, 2 red')
    p.add_argument('--collect-min-std', type=float, default=0.0,
                   help='collection-only floor on the exploration std (0 = off)')
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--max-episode-steps', type=int, default=None)
    p.add_argument('--tag', default=None)
    args = p.parse_args()

    kwargs = {}
    if args.max_episode_steps:
        kwargs['max_episode_steps'] = args.max_episode_steps
    env = gym.make(args.env, **kwargs)
    eval_env = gym.make(args.env, **kwargs)
    T = env.spec.max_episode_steps or 1000
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 10000)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    s, _ = env.reset()
    obs_dim = flat_obs(s).shape[0]
    act_dim = env.action_space.shape[0]
    act_low, act_high = env.action_space.low, env.action_space.high
    agent = DreamerV3(obs_dim, act_dim, seq_len=args.seq_len,
                      noise_beta=args.noise_beta, noise_seq_len=T,
                      collect_min_std=args.collect_min_std)
    seed_steps = args.seed_steps or max(1000, 5 * T)

    tag = args.tag or f'{args.env}_seed{args.seed}'
    os.makedirs('exp_dreamerv3', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    csv_path = f'exp_dreamerv3/{tag}.csv'
    print(f'[{tag}] obs {obs_dim} act {act_dim} T {T} -> {csv_path}', flush=True)

    t0 = time.time()
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env_steps', 'episode', 'train_return', 'train_success',
                    'eval_return', 'eval_success', 'model_loss', 'kl',
                    'actor_loss', 'critic_loss', 'imag_return', 'reward_max', 'wall_s'])
        s, _ = env.reset()
        agent.reset_episode()
        ep_ret, ep_reached, episode, losses = 0.0, 0.0, 0, {}
        next_eval = args.eval_every
        for step in range(1, args.steps + 1):
            o = flat_obs(s)
            if step <= seed_steps:
                a = env.action_space.sample()
                a_unit = 2 * (a - act_low) / (act_high - act_low) - 1
            else:
                a_unit = agent.act(o, eval_mode=False)
                a = act_low + (a_unit + 1) / 2 * (act_high - act_low)
            s2, r, done, trunc, info = env.step(a.astype(np.float32))
            agent.buffer.add(o, a_unit.astype(np.float32), r, float(done))
            ep_ret += r
            ep_reached = max(ep_reached, float(info.get('is_success', 0.0)))
            if step > seed_steps and step % args.update_every == 0:
                losses = agent.update()
            if done or trunc:
                agent.buffer.end_episode(flat_obs(s2))
                s, _ = env.reset()
                agent.reset_episode()
                episode += 1
                last_ret, last_reached = ep_ret, ep_reached
                ep_ret, ep_reached = 0.0, 0.0
                if step >= next_eval:
                    next_eval += args.eval_every
                    ev_ret, ev_succ = evaluate(eval_env, agent, args.eval_episodes)
                    agent.reset_episode()  # eval left stale recurrent state
                    row = [step, episode, last_ret, last_reached, ev_ret, ev_succ,
                           losses.get('model', 0), losses.get('kl', 0),
                           losses.get('actor', 0), losses.get('critic', 0),
                           losses.get('ret', 0), losses.get('rmax', 0), time.time() - t0]
                    w.writerow([f'{x:.4f}' if isinstance(x, float) else x for x in row])
                    f.flush()
                    print(f'[{tag}] step {step:7d} ep {episode:4d} | train ret {last_ret:8.1f} '
                          f'| eval ret {ev_ret:8.1f} succ {ev_succ:.2f} | {time.time()-t0:.0f}s', flush=True)
                    agent.save(f'checkpoints/dreamerv3_{tag}.pt')
            else:
                s = s2
    env.close(); eval_env.close()


if __name__ == '__main__':
    main()
