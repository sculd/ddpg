import datetime
import gymnasium as gym
import numpy as np
from collections import deque
from ddpg.agent import Agent
import wandb


env_name = "LunarLander-v3"
env_param = {"continuous": True}
#env_name = "BipedalWalker-v3"
#env_param = {}

wandb.init(
    # set the wandb project where this run will be logged
    project=f"{env_name}-Continuous",
)

env = gym.make(env_name, render_mode=None, **env_param)
np.random.seed(int(datetime.datetime.now().timestamp()))

agent = Agent(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], env_name=env_name,
              noise_sigma=0.2,
              toggle_sigma_decay=False)
agent.load_models()

max_episodes = 1000
max_steps = 1000
save_interval = 50
scores_window = deque(maxlen=save_interval)
max_avg_score = 0

for episode in range(1, max_episodes + 1):
    done = False
    score = 0
    obs, _ = env.reset()

    for t in range(max_steps):
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

    wandb.log({"score": score, "mean_score": avg_score})
    if episode % 10 == 0:
        print(f'{episode=}, {score=}, mean_score: {avg_score}', end="\r")

    if score >= 1000:
        print(f'Environment solved in {episode} episodes, average score {avg_score}', end="\r")
        agent.save_models()
        break

    if episode % save_interval == 0:
        if max_avg_score > 100:
            if avg_score > 100:
                print(f"{episode=}")
                agent.save_models()
            else:
                print(f"{avg_score} is not high enough thus skip saving.")
        else:
            print(f"{episode=}")
            agent.save_models()

agent.update_network_parameters(tau=0.1)
agent.save_models()
env.close()
