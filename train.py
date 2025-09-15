import datetime
import gymnasium as gym
import numpy as np
import argparse
from collections import deque
from ddpg.agent import Agent
from ddpg.util.config import load_config, merge_with_args
import wandb
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPG agent')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file name or path (e.g., bipedal_walker, lunar_lander)')

    # Allow overriding specific parameters from command line
    parser.add_argument('--env', type=str, default=None,
                        help='Override environment name')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override maximum number of training episodes')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--load-checkpoint', action='store_true',
                        help='Load checkpoint before training')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration file
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.env is not None:
        config['env'] = args.env
    if args.episodes is not None:
        config['episodes'] = args.episodes
    if args.seed is not None:
        config['seed'] = args.seed
    if args.load_checkpoint:
        config['load_checkpoint'] = args.load_checkpoint
    if args.no_wandb:
        config['wandb'] = False

    # Print loaded configuration
    print(f"Loaded configuration from: {args.config}")
    print("Configuration:")
    for key, value in sorted(config.items()):
        if key != 'env_params':
            print(f"  {key}: {value}")

    # Initialize Weights & Biases if enabled
    if config.get('wandb', True):
        project_name = config.get('wandb_project') or f"{config['env']}-Continuous"
        wandb.init(project=project_name, config=config)

    # Get environment parameters
    env_params = config.get('env_params', {})

    # Create environment
    env = gym.make(config['env'], render_mode=None, **env_params)

    # Set random seed
    seed = config.get('seed')
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())
    np.random.seed(seed)

    # Create agent with configurable parameters
    agent = Agent(n_inputs=env.observation_space.shape[0],
                  n_actions=env.action_space.shape[0],
                  env_name=config['env'],
                  lr_actor=config['lr_actor'],
                  lr_critic=config['lr_critic'],
                  tau=config['tau'],
                  gamma=config['gamma'],
                  replay_buffer_size=config['buffer_size'],
                  batch_size=config['batch_size'],
                  noise_sigma=config['noise_sigma'],
                  noise_sigma_final=config['noise_sigma_final'],
                  noise_sigma_decay=config['noise_sigma_decay'])

    if config.get('load_checkpoint', False):
        agent.load()

    max_episodes = config['episodes']
    max_steps = config['max_steps']
    total_steps = 0
    save_interval = config['save_interval']
    scores_window = deque(maxlen=save_interval)
    max_avg_score = 0

    for episode in range(1, max_episodes + 1):
        done, truncated = False, False
        score = 0
        obs, _ = env.reset()

        for t in range(max_steps):
            total_steps += 1
            if done:
                break
            act = agent.choose_action(obs)
            next_state, reward, done, truncated, info = env.step(act)
            agent.step(obs, act, reward, next_state, int(done))
            score += reward

            obs = next_state

        scores_window.append(score)
        avg_score = np.mean(scores_window)
        if avg_score > max_avg_score:
            max_avg_score = avg_score

        if config.get('wandb', True):
            wandb.log({"score": score, "mean_score": avg_score, "sigma": agent.noise.sigma})
        agent.noise.decay_sigma()

        if episode % 10 == 0:
            print(f'{episode=}, {total_steps=} (avg steps per episode: {total_steps/episode:.1f}), {score=:.1f}, mean_score: {avg_score:.1f}', end="\r")

        if avg_score >= config['target_score']:
            print(f'Environment solved in {episode} episodes, average score {avg_score}', end="\r")
            agent.save()
            break

        if episode % save_interval == 0:
            if max_avg_score > 100:
                if avg_score > 100:
                    print(f"{episode=}")
                    agent.save()
                else:
                    print(f"{avg_score} is not high enough thus skip saving.")
            else:
                print(f"{episode=}")
                agent.save()

    agent.save()
    env.close()


if __name__ == "__main__":
    main()
