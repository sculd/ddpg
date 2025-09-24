import datetime
import gymnasium as gym
import numpy as np
import argparse
from collections import deque
from ddpg.util.config import load_config, merge_with_args
from ddpg.util.agent_factory import create_agent
import wandb
import yaml

from ddpg.util.manual_vec_env import make_manual_vector_env


def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPG agent')
    parser.add_argument('--config', type=str, required=False,
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


def main(config_name=None):
    args = parse_args()

    config_name = config_name or args.config
    # Load configuration file
    config = load_config(config_name)

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
    single_env = gym.make(config['env'], render_mode=None, **env_params)
    n_inputs = single_env.observation_space.shape[0]
    n_actions = single_env.action_space.shape[0]
    single_env.close()

    num_envs = config.get('num_envs', 4)
    env = make_manual_vector_env(config['env'], num_envs=num_envs, **env_params)

    # Set random seed
    seed = config.get('seed')
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())
    np.random.seed(seed)

    # Create agent with configurable parameters
    agent_params = {
        'n_inputs': n_inputs,
        'n_actions': n_actions,
        'env_name': config['env'],
        'lr_actor': config['lr_actor'],
        'lr_critic': config['lr_critic'],
        'tau': config['tau'],
        'gamma': config['gamma'],
        'replay_buffer_size': config['buffer_size'],
        'batch_size': config['batch_size'],
        'noise_sigma_initial': config['noise_sigma_initial'],
        'noise_sigma_final': config['noise_sigma_final'],
        'num_envs': num_envs,
        'noise_seed': seed,
    }

    agent = create_agent(
        agent_type=config.get('agent_type', 'agent'),
        env=env,
        **agent_params
    )

    if config.get('load_checkpoint', False):
        agent.load()

    max_episodes = config['episodes']
    max_steps = config['max_steps']
    total_steps = 0
    save_interval = config['save_interval']
    scores_window = deque(maxlen=save_interval)
    max_avg_score = 0
    noise_decay_score_threshold = config.get('noise_decay_score_threshold')
    noise_score_decay_rate = config.get('noise_score_decay_rate')
    noise_sigma_initial = config.get('noise_sigma_initial')
    noise_sigma_final = config.get('noise_sigma_final')

    for episode in range(1, max_episodes + 1):
        done = np.zeros(env.num_envs, dtype=bool)
        truncated = np.zeros(env.num_envs, dtype=bool)
        score = np.zeros(env.num_envs)
        obs, _ = env.reset()
        agent.noise.reset(reset_sigma=False)
        agent.set_noise_sigma(noise_sigma_initial)

        completed_scores = []
        active_envs = np.ones(env.num_envs, dtype=bool)

        for t in range(max_steps):
            if not np.any(active_envs):
                break
            total_steps += 1

            prev_active = active_envs.copy()
            actions = np.zeros((env.num_envs, n_actions), dtype=np.float32)
            if np.any(prev_active):
                active_indices = np.where(prev_active)[0]
                chosen_actions = agent.choose_action(
                    obs[prev_active],
                    env_indices=active_indices,
                )
                actions[prev_active] = chosen_actions

            next_state, reward, done, truncated, info = env.step(actions)
            terminated = done | truncated

            if np.any(prev_active):
                agent.step(
                    obs[prev_active],
                    actions[prev_active],
                    reward[prev_active],
                    next_state[prev_active],
                    terminated[prev_active],
                )
                score[prev_active] += reward[prev_active]

            newly_done = terminated & prev_active
            if np.any(newly_done):
                completed_scores.extend(score[newly_done].tolist())
                active_envs[newly_done] = False

            still_active = prev_active & ~terminated
            if np.any(still_active):
                obs[still_active] = next_state[still_active]

        if np.any(active_envs):
            completed_scores.extend(score[active_envs].tolist())

        mean_score = np.mean(completed_scores) if completed_scores else 0.0
        scores_window.append(mean_score)
        avg_score = np.mean(scores_window)
        if avg_score > max_avg_score:
            max_avg_score = avg_score

        if config.get('wandb', True):
            wandb.log({"score": mean_score, "mean_score": avg_score, "sigma": agent.noise.sigma})

        if noise_decay_score_threshold is not None and avg_score >= noise_decay_score_threshold:
            current_sigma = agent.noise.sigma
            updated_sigma = current_sigma * (1 - noise_score_decay_rate)
            target_noise_sigma = max(noise_sigma_final, updated_sigma)
            agent.set_noise_sigma(target_noise_sigma)

        if episode % 10 == 0:
            print(f'{episode=}, {total_steps=} (avg steps per episode: {total_steps/episode:.1f}), score={np.mean(score):.1f}, mean_score: {avg_score:.1f}', end="\r")

        if avg_score >= config['target_score']:
            print(f'Environment solved in {episode} episodes, average score {avg_score}', end="\r")
            agent.save()
            break

        if episode % save_interval == 0:
            if max_avg_score > config['target_score'] / 2:
                if avg_score > config['target_score'] / 2:
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
    main("bipedal_walker")
