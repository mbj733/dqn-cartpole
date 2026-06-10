"""
DQN (Deep Q-Network) for CartPole-v1
Based on: Mnih et al., "Playing Atari with Deep Reinforcement Learning", 2013
"""

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


# ============================================================
# DQN Network
# ============================================================
class DQN(nn.Module):
    """Simple 3-layer MLP for Q-value approximation."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Hyperparameters
# ============================================================
ENV_NAME = "CartPole-v1"
LR = 1e-3                # Learning rate
GAMMA = 0.99             # Discount factor
EPSILON_START = 1.0      # Initial exploration rate
EPSILON_MIN = 0.01       # Minimum exploration rate
EPSILON_DECAY = 0.995    # Decay per step
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE = 10       # Update target network every N episodes
EPISODES = 800

# ============================================================
# Environment & Networks
# ============================================================
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]  # 4: position, velocity, angle, angular velocity
act_dim = env.action_space.n               # 2: left or right

q_net = DQN(obs_dim, act_dim)
target_net = DQN(obs_dim, act_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=BUFFER_SIZE)
epsilon = EPSILON_START

# ============================================================
# Training Loop
# ============================================================
episode_rewards = []

for episode in range(EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # ---- Epsilon-greedy action selection ----
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                action = q_net(obs_t).argmax().item()

        # ---- Step environment ----
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # ---- Store transition ----
        replay_buffer.append((obs, action, reward, next_obs, done))
        obs = next_obs
        total_reward += reward

        # ---- Train on a batch ----
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            obs_b, act_b, rew_b, next_b, done_b = zip(*batch)

            obs_b = torch.FloatTensor(np.array(obs_b))
            act_b = torch.LongTensor(act_b).unsqueeze(1)
            rew_b = torch.FloatTensor(rew_b).unsqueeze(1)
            next_b = torch.FloatTensor(np.array(next_b))
            done_b = torch.FloatTensor(done_b).unsqueeze(1)

            # Q-learning target: r + gamma * max_a' Q_target(s', a')
            with torch.no_grad():
                target = rew_b + GAMMA * target_net(next_b).max(1, keepdim=True)[0] * (1 - done_b)

            current_q = q_net(obs_b).gather(1, act_b)
            loss = nn.MSELoss()(current_q, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---- End of episode ----
    episode_rewards.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    # Logging
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode + 1:4d} | Avg Reward (last 50): {avg_reward:6.1f} | Epsilon: {epsilon:.3f}")

env.close()

# ============================================================
# Plot & Save
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.6, linewidth=0.8, label="Episode Reward")
# Smoothed curve
if len(episode_rewards) >= 50:
    smoothed = np.convolve(episode_rewards, np.ones(50) / 50, mode="valid")
    plt.plot(range(49, len(episode_rewards)), smoothed, linewidth=2, label="50-ep Moving Average", color="red")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN on CartPole-v1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("training_curve.png", dpi=150)
print("Training curve saved as training_curve.png")
