import numpy as np
import matplotlib.pyplot as plt
class Environment:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.special_states = {(0, 1): ((4, 1), 10), (0, 3): ((2, 3), 5)}

    def get_reward(self, state, next_state):
        if state in self.special_states:
            return self.special_states[state]
        elif next_state[0] < 0 or next_state[0] >= self.size or next_state[1] < 0 or next_state[1] >= self.size:
            return state, -1
        else:
            return next_state, 0


class Agent:
    def __init__(self, environment, discount_factor=0.9):
        self.environment = environment
        self.discount_factor = discount_factor
        self.values = np.zeros_like(environment.grid)

    def bellman_optimality_update(self, state):
        possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        max_value = float('-inf')

        for action in possible_actions:
            next_state = (state[0] + action[0], state[1] + action[1])
            next_state, reward = self.environment.get_reward(state, next_state)

            if 0 <= next_state[0] < self.environment.size and 0 <= next_state[1] < self.environment.size:
                next_value = reward + self.discount_factor * self.values[next_state]
                max_value = max(max_value, next_value)

        return max_value

    def value_iteration_optimality(self, num_iterations=1000):
        for _ in range(num_iterations):
            new_values = np.zeros_like(self.values)

            for i in range(self.environment.size):
                for j in range(self.environment.size):
                    state = (i, j)
                    new_values[state] = self.bellman_optimality_update(state)

            self.values = new_values

    def plot_values(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.values, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title('Final Values of the States')
        plt.show()

# Create the environment and agent
env = Environment()
agent = Agent(env)

# Perform value iteration with Bellman optimality equation
agent.value_iteration_optimality()
agent.plot_values()

# Print the final values
np.set_printoptions(precision=1, suppress=True)
print(agent.values)
