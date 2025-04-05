import gym
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hyperparameters
alpha = 0.01
gamma = 0.99
epsilon = 0.1
episodes = 500

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_features = env.observation_space.shape[0]

# Feature vector for (state, action)
def featurize_state_action(state, action):
    state = np.asarray(state)
    features = np.zeros(n_features * n_actions)
    start = action * n_features
    features[start:start + n_features] = state
    return np.append(features, 1.0)  # Add bias

# ε-greedy action selection
def select_action(state, w):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    q_values = [np.dot(w, featurize_state_action(state, a)) for a in range(n_actions)]
    return np.argmax(q_values)

# Training function with reward tracking
def train():
    w = np.zeros(n_features * n_actions + 1)
    reward_per_episode = []

    for ep in range(episodes):
        state = env.reset()[0]
        action = select_action(state, w)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_action = select_action(next_state, w)
            x = featurize_state_action(state, action)
            x_next = featurize_state_action(next_state, next_action)

            q = np.dot(w, x)
            q_next = np.dot(w, x_next) if not done else 0
            td_error = reward + gamma * q_next - q
            w += alpha * td_error * x

            state = next_state
            action = next_action

        reward_per_episode.append(total_reward)

    return w, reward_per_episode

# Print function for weights
def print_weights(w, n_features, n_actions):
    print("\n=== Learned Weights ===")
    for a in range(n_actions):
        start = a * n_features
        end = start + n_features
        print(f"Action {a} weights: {w[start:end]}")
    print(f"Bias term: {w[-1]}")

# Run training and plot results
weights, rewards = train()
print_weights(weights, n_features, n_actions)

# Plotting the reward per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Progress of Linear Q-Approximation on CartPole")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
