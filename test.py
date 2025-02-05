import datetime
import gymnasium as gym
import numpy as np
from ddpg.agent import Agent


env_name = "LunarLander-v3"
env_param = {"continuous": True}
#env_name = "BipedalWalker-v3"
#env_param = {}

env = gym.make(env_name, render_mode="human", **env_param)
np.random.seed(int(datetime.datetime.now().timestamp()))

agent = Agent(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], env_name=env_name)
agent.load_models()

max_episodes = 100
max_steps = 1000

for episode in range(1, max_episodes + 1):
    done = False
    score = 0
    obs, _ = env.reset()

    for t in range(max_steps):
        if done:
            break
        act = agent.choose_action(obs, with_noise=False)
        next_state, reward, done, truncated, info = env.step(act)
        agent.step(obs, act, reward, next_state, int(done))
        score += reward

        obs = next_state
    
    if episode % 1 == 0:
        print(f'{episode=}, {score=}', end="\r")

env.close()


