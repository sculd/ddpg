import datetime
import gymnasium as gym
import numpy as np
from collections import deque
from ddpg.agent import Agent
import wandb

'''
lr_actor=1e-3
lr_critic=1e-3
achieves avg score > 240 for lunar lander around 3k episodes.
'''
#env_name = "LunarLander-v3"
#env_param = {"continuous": True}
#env_name = "Pusher-v5"
env_name = "BipedalWalker-v3"
env_param = {}

wandb.init(
    # set the wandb project where this run will be logged
    project=f"{env_name}-Continuous",
)

env = gym.make(env_name, render_mode=None, **env_param)
np.random.seed(int(datetime.datetime.now().timestamp()))

tag = ""
agent = Agent(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], env_name=env_name,
              noise_sigma=0.2, noise_sigma_final=0.2, noise_sigma_decay=1./1000)
#agent.load()

max_episodes = 4000
max_steps = 500
total_steps = 0
save_interval = 50
reset_sigma_interval = 1000
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

    wandb.log({"score": score, "mean_score": avg_score, "sigma": agent.noise.sigma})
    agent.noise.decay_sigma()
     
    if episode % 10 == 0:
        print(f'{episode=}, {total_steps=} (avg steps per episode: {total_steps/episode:.1f}), {score=:.1f}, mean_score: {avg_score:.1f}', end="\r")

    if avg_score >= 240:
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
