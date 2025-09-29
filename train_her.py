import datetime
from collections import deque

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from ddpg.agner_her import AgentHer as Agent

gym.register_envs(gymnasium_robotics)
env_name = 'FetchReach-v3'
env = gym.make(env_name, max_episode_steps=100, render_mode=None) # "human"
env = env.unwrapped

LOAD_MODELS = False  # Set to True only when testing
SAVE_BEST_ONLY = True  # Only save when we get better performance
EPISODES_PER_EPOCH = 800
N_EPOCHS = 200  # This will give 40,000 episodes total
MAX_STEPS = 50

agent = Agent(n_inputs=env.observation_space['observation'].shape[0], n_actions=env.action_space.shape[0], env_name=env_name,
              env=env,
              noise_sigma=0.2,
              toggle_sigma_decay=True,
              lr_actor=0.00001,
              lr_critic=0.0001,
              batch_size=128)
if LOAD_MODELS:
    print("Loading pre-trained models...")
    agent.load_models()

best_success_rate = 0.0
best_avg_distance = float('inf')
epoch_success_rates = []  # Track success rate for each epoch
epoch_distances = []  # Track average distance per epoch

for epoch in range(N_EPOCHS):
    print(f"\nStarting Epoch {epoch+1}/{N_EPOCHS}")
    success_window = deque(maxlen=EPISODES_PER_EPOCH)  # Track all episodes in epoch
    scores_window = deque(maxlen=EPISODES_PER_EPOCH)
    her_score_window = deque(maxlen=EPISODES_PER_EPOCH)
    distance_window = deque(maxlen=EPISODES_PER_EPOCH)  # Track distances for the epoch

    for episode in range(1, EPISODES_PER_EPOCH + 1):
        done, truncated = False, False
        score = 0
        success = False
        state, _ = env.reset()
        agent.reset_episode_goals()
        obs = state['observation']
        desired_goal = state['desired_goal']

        for t in range(MAX_STEPS):
            if done or truncated:
                break
            act = agent.choose_action(obs, desired_goal)
            next_state, reward, done, truncated, info = env.step(act)
            next_obs, achieved_goal = next_state['observation'], next_state['achieved_goal']
            
            agent.add_to_memory(obs, achieved_goal, desired_goal, act, reward, next_obs, int(done))
            score += reward + 1.0
            obs = next_obs
            
            distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
            if distance_to_goal < 0.05:
                success = True
                if episode % 50 == 0:  # Reduce printing frequency
                    print(f'\nSuccess in episode {episode} at step {t+1}! Distance: {distance_to_goal:.4f}')
                break

        her_score = agent.add_her_batch_to_memory()
        # Learn from the episode
        agent.learn_episode(t+1)

        scores_window.append(score)
        her_score_window.append(her_score)
        success_window.append(float(success))
        distance_window.append(distance_to_goal)  # Track final distance of episode

        if episode % 20 == 0:
            current_success_rate = np.mean(success_window)
            current_avg_distance = np.mean(distance_window)
            print(f'Epoch {epoch+1}, Episode {episode}/{EPISODES_PER_EPOCH}, Success Rate: {current_success_rate:.2%}, Avg Distance: {current_avg_distance:.4f}', end="\r")

    # End of epoch processing
    epoch_success_rate = np.mean(success_window)
    epoch_avg_distance = np.mean(distance_window)
    epoch_success_rates.append(epoch_success_rate)
    epoch_distances.append(epoch_avg_distance)
    print(f"\nEpoch {epoch+1} completed. Success rate: {epoch_success_rate:.2%}, Average distance: {epoch_avg_distance:.4f}")
    
    # Save if either metric improves
    improved = False
    if epoch_success_rate > best_success_rate:
        print(f'New best success rate: {epoch_success_rate:.2%} (previous: {best_success_rate:.2%})')
        best_success_rate = epoch_success_rate
        improved = True
    if epoch_avg_distance < best_avg_distance:
        print(f'New best average distance: {epoch_avg_distance:.4f} (previous: {best_avg_distance:.4f})')
        best_avg_distance = epoch_avg_distance
        improved = True
    
    if improved and SAVE_BEST_ONLY:
        agent.save_models()
    
    agent.memory.reset()

print(f"\nTraining completed. Best success rate achieved: {best_success_rate:.2%}")
print("\nEpoch history:")
for i, (rate, dist) in enumerate(zip(epoch_success_rates, epoch_distances)):
    print(f"Epoch {i+1}: Success rate = {rate:.2%}, Average distance = {dist:.4f}")
env.close()

