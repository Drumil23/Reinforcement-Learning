import numpy as np

# Simple environment with states from 1 to 5 (states 0 and 6 are terminal)
N_STATES = 7
TERMINAL_LEFT = 0
TERMINAL_RIGHT = 6
ALPHA = 0.01  # Learning rate
GAMMA = 1.0   # Discount factor
EPISODES = 100

# Feature representation for each state (one-hot encoding for simplicity)
def one_hot(state, size=N_STATES):
    vec = np.zeros(size)
    vec[state] = 1
    return vec

# TD(0) with linear function approximation
def td_zero_linear_approximation():
    w = np.zeros(N_STATES)  # Weight vector for linear function approximation

    for episode in range(EPISODES):
        state = 3  # Start from middle state
        while state != TERMINAL_LEFT and state != TERMINAL_RIGHT:
            x = one_hot(state)

            # Take a random action (left or right)
            action = np.random.choice([-1, 1])
            next_state = state + action

            reward = 0
            if next_state == TERMINAL_RIGHT:
                reward = 1

            x_next = one_hot(next_state)

            # TD target and TD error
            v_hat = np.dot(w, x)
            v_hat_next = np.dot(w, x_next)
            delta = reward + GAMMA * v_hat_next - v_hat

            # Update weights
            w += ALPHA * delta * x

            state = next_state

    return w

# Train and show weights
weights = td_zero_linear_approximation()
print("Learned weights (approximate value function):")
for s in range(1, 6):
    print(f"V({s}) ≈ {weights[s]:.3f}")
