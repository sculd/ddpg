import datetime
import gymnasium as gym
import numpy as np
from ddpg.agent import Agent

#env_name = "LunarLander-v3"
#env_param = {"continuous": True}
#env_name = "Pusher-v5"
env_name = "BipedalWalker-v3"
env_param = {}

env = gym.make(env_name, render_mode="human", **env_param)
#env = gym.make(env_name, **env_param)

from gymnasium.wrappers import RecordVideo
#env = RecordVideo(env, video_folder="./images", episode_trigger=lambda t: t % 10 == 0, disable_logger=True)

np.random.seed(int(datetime.datetime.now().timestamp()))

agent = Agent(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], env_name=env_name,
              noise_sigma=0.1, noise_sigma_final=0.1)
agent.load(load_memory=False)

max_episodes = 100
max_steps = 500

for episode in range(1, max_episodes + 1):
    done, truncated = False, False
    score = 0
    obs, _ = env.reset()

    for t in range(max_steps):
        if done or truncated:
            break
        act = agent.choose_action(obs, with_noise=False)
        next_state, reward, done, truncated, info = env.step(act)
        score += reward
        if score > 200:
            break

        obs = next_state
        agent.noise.decay_sigma()

    if episode % 1 == 0:
        print(f'{episode=}, {score=:.1f}', end="\n")

env.close()


