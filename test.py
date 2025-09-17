import datetime
import gymnasium as gym
import numpy as np
import argparse
from ddpg.util.config import load_config
from ddpg.util.agent_factory import create_agent


def parse_args():
    parser = argparse.ArgumentParser(description='Test DDPG agent')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file name or path (e.g., bipedal_walker, lunar_lander)')

    # Allow overriding specific parameters from command line
    parser.add_argument('--episodes', type=int, default=3,
                        help='Override number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--record', action='store_true',
                        help='Enable video recording')
    parser.add_argument('--with-noise', action='store_true',
                        help='Use noise during evaluation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration file
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.episodes is not None:
        config['test_episodes'] = args.episodes
    if args.no_render:
        config['render'] = False
    if args.record:
        config['record'] = True
    if args.with_noise:
        config['with_noise'] = True
    if args.seed is not None:
        config['seed'] = args.seed

    # Print loaded configuration
    print(f"Loaded configuration from: {args.config}")
    print("Test Configuration:")
    test_keys = ['env', 'test_episodes', 'max_steps', 'render', 'record',
                 'with_noise', 'noise_sigma', 'score_threshold']
    for key in test_keys:
        if key in config:
            print(f"  {key}: {config[key]}")

    # Get environment parameters
    env_params = config.get('env_params', {})

    # Create environment
    render_mode = "human" if config.get('render', True) else None
    env = gym.make(config['env'], render_mode=render_mode, **env_params)

    # Optionally wrap with video recorder
    if config.get('record', False):
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, video_folder=config.get('video_folder', './images'),
                         episode_trigger=lambda t: t % config.get('video_freq', 10) == 0,
                         disable_logger=True)

    # Set random seed
    seed = config.get('seed')
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())
    np.random.seed(seed)

    # Create and load agent
    agent_params = {
        'n_inputs': env.observation_space.shape[0],
        'n_actions': env.action_space.shape[0],
        'env_name': config['env'],
        'noise_sigma': config.get('noise_sigma', 0.1),
        'noise_sigma_final': config.get('noise_sigma', 0.1)
    }

    agent = create_agent(
        agent_type=config.get('agent_type', 'agent'),
        env=env if config.get('agent_type') == 'agent_her' else None,
        **agent_params
    )
    agent.load(load_memory=config.get('load_memory', False))

    max_episodes = config['test_episodes']
    max_steps = config['max_steps']

    for episode in range(1, max_episodes + 1):
        done, truncated = False, False
        score = 0
        obs, _ = env.reset()

        for t in range(max_steps):
            if done or truncated:
                break
            act = agent.choose_action(obs, with_noise=config.get('with_noise', False))
            next_state, reward, done, truncated, info = env.step(act)
            score += reward
            if score > config.get('score_threshold', 200):
                break

            obs = next_state
            if config.get('with_noise', False):
                agent.noise.decay_sigma()

        if episode % 1 == 0:
            print(f'{episode=}, {score=:.1f}', end="\n")

    env.close()


if __name__ == "__main__":
    main()


