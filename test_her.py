import datetime
import numpy as np
from collections import deque
from ddpg.agner_her import AgentHer as Agent
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
env_name = 'FetchReach-v3'
env = gym.make(env_name, max_episode_steps=50, render_mode="human")  # Reduced max steps to match training
env = env.unwrapped

# Initialize agent in testing mode
agent = Agent(n_inputs=env.observation_space['observation'].shape[0], 
              n_actions=env.action_space.shape[0], 
              env_name=env_name,
              env=env,
              noise_sigma=0.0,  # No noise during testing
              toggle_sigma_decay=False)
agent.load_models()
agent.set_testing_mode()  # Properly set networks to eval mode

# Testing parameters
n_test_episodes = 100
max_steps = 50  # Match training max steps
success_threshold = 0.05  # Distance threshold for success

# Metrics tracking
success_window = deque(maxlen=100)
distance_window = deque(maxlen=100)
steps_window = deque(maxlen=100)

print("\nStarting evaluation...")
print("=" * 50)

for episode in range(1, n_test_episodes + 1):
    done, truncated = False, False
    success = False
    state, _ = env.reset()
    obs = state['observation']
    desired_goal = state['desired_goal']
    
    for t in range(max_steps):
        if done or truncated:
            break
            
        # Get action without noise
        act = agent.choose_action(obs, desired_goal, with_noise=False)
        next_state, reward, done, truncated, info = env.step(act)

        # Track metrics
        achieved_goal = next_state['achieved_goal']
        distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
        
        if distance_to_goal < success_threshold:
            success = True
            break

        obs = next_state['observation']

    # Store episode metrics
    success_window.append(float(success))
    distance_window.append(distance_to_goal)
    steps_window.append(t + 1)
    
    # Print progress with more detailed metrics
    success_rate = np.mean(success_window) * 100
    avg_distance = np.mean(distance_window)
    avg_steps = np.mean(steps_window)
    
    print(f"Episode {episode}/{n_test_episodes} | ", end="")
    print(f"Success: {'Yes' if success else 'No'} | ", end="")
    print(f"Steps: {t+1} | ", end="")
    print(f"Final Distance: {distance_to_goal:.4f}")
    
    if episode % 10 == 0:
        print("-" * 50)
        print(f"Last 100 episodes statistics:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Distance: {avg_distance:.4f}")
        print(f"Average Steps to Goal: {avg_steps:.1f}")
        print("-" * 50)

print("\nEvaluation completed!")
print("=" * 50)
print(f"Final Success Rate: {success_rate:.1f}%")
print(f"Final Average Distance: {avg_distance:.4f}")
print(f"Final Average Steps to Goal: {avg_steps:.1f}")
print("=" * 50)

env.close()
