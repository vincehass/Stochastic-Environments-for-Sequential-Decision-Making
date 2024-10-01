import numpy as np
from tfb.lib.acquisition_fn import calculate_acquisition  # Adjust the import based on your actual function
from tfb.lib.generator import GFlowNetGenerator  # Assuming this is the base class

class StochasticStateGFlowNetGenerator(GFlowNetGenerator):
    def __init__(self, num_quantiles=100, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.num_quantiles = num_quantiles
        self.quantiles = np.linspace(-1, 1, num_quantiles)  # Define quantile values
        self.quantile_counts = np.zeros(num_quantiles)  # Count of rewards in each quantile
        self.total_count = 0  # Total count of rewards for normalization

    def update_quantiles(self, reward):
        # Update the quantile counts based on the received reward
        bin_index = np.digitize(reward, self.quantiles) - 1
        if 0 <= bin_index < self.num_quantiles:
            self.quantile_counts[bin_index] += 1  # Increment count for the quantile
            self.total_count += 1  # Increment total count

    def sample_reward(self):
        # Sample a reward based on the quantile distribution
        probabilities = self.quantile_counts / self.total_count if self.total_count > 0 else np.ones(self.num_quantiles) / self.num_quantiles
        return np.random.choice(self.quantiles, p=probabilities)

    def calculate_trajectory_balance(self, trajectories):
        # Implement your logic to calculate the trajectory balance
        balance = np.mean([np.sum(trajectory) for trajectory in trajectories])  # Example calculation
        return balance

    def calculate_reward(self, state, action, trajectories):
        # Calculate the reward based on the trajectory balance
        balance = self.calculate_trajectory_balance(trajectories)
        reward = calculate_acquisition(state, action) + balance  # Combine acquisition and balance
        return reward  # Return the combined value as the reward

    def handle_stochastic_states(self, state, action):
        # Simulate the stochastic nature of the environment
        # For example, we can define a transition probability matrix
        transition_probabilities = {
            (0, 0): [(0.8, 0), (0.2, 1)],  # Action 0 in state 0 leads to state 0 with 80% and state 1 with 20%
            (0, 1): [(0.5, 0), (0.5, 1)],  # Action 1 in state 0 leads to state 0 or 1 with equal probability
            (1, 0): [(0.6, 0), (0.4, 1)],  # Action 0 in state 1
            (1, 1): [(0.1, 0), (0.9, 1)],  # Action 1 in state 1
        }

        # Get the possible transitions for the current state and action
        transitions = transition_probabilities.get((state[0], action), [])
        next_states = []
        for prob, next_state in transitions:
            if np.random.rand() < prob:  # Sample based on the transition probability
                next_states.append(next_state)

        return next_states  # Return the possible next states

    def trajectory_balance_loss(self, state, action, trajectories):
        # Example of how to integrate trajectory balance into the loss function
        reward = self.calculate_reward(state, action, trajectories)

        # Update the quantiles with the new reward
        self.update_quantiles(reward)

        # Sample a stochastic reward from the quantile distribution
        stochastic_reward = self.sample_reward()

        # Handle stochastic states
        next_states = self.handle_stochastic_states(state, action)

        # Return the reward, the sampled stochastic reward, and the next possible states
        return reward, stochastic_reward, next_states

# Example usage
if __name__ == "__main__":
    generator = StochasticStateGFlowNetGenerator(num_quantiles=100)
    state = np.array([0])  # Example state
    action = np.random.choice([0, 1])  # Example action
    trajectories = [np.random.rand(5) for _ in range(10)]  # Example trajectories
    reward, stochastic_reward, next_states = generator.trajectory_balance_loss(state, action, trajectories)
    print(f"Reward: {reward}, Stochastic Reward: {stochastic_reward}, Next States: {next_states}")




    language:tfb/lib/generator/gfn.py
import numpy as np
import tensorflow as tf
from tfb.lib.acquisition_fn import calculate_acquisition  # Adjust the import based on your actual function
from tfb.lib.generator import GFlowNetGenerator  # Assuming this is the base class

class StochasticDBGFlowNetGenerator(GFlowNetGenerator):
    def __init__(self, num_quantiles=100, alpha=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.num_quantiles = num_quantiles
        self.alpha = alpha  # CVaR parameter
        self.quantiles = np.linspace(-1, 1, num_quantiles)  # Define quantile values
        self.quantile_counts = np.zeros(num_quantiles)  # Count of rewards in each quantile
        self.total_count = 0  # Total count of rewards for normalization

    def update_quantiles(self, reward):
        # Update the quantile counts based on the received reward
        bin_index = np.digitize(reward, self.quantiles) - 1
        if 0 <= bin_index < self.num_quantiles:
            self.quantile_counts[bin_index] += 1  # Increment count for the quantile
            self.total_count += 1  # Increment total count

    def sample_reward(self):
        # Sample a reward based on the quantile distribution
        probabilities = self.quantile_counts / self.total_count if self.total_count > 0 else np.ones(self.num_quantiles) / self.num_quantiles
        return np.random.choice(self.quantiles, p=probabilities)

    def calculate_trajectory_balance(self, trajectories):
        # Calculate the trajectory balance
        balance = np.mean([np.sum(trajectory) for trajectory in trajectories])  # Example calculation
        return balance

    def calculate_reward(self, state, action, trajectories):
        # Calculate the reward based on the trajectory balance and acquisition function
        balance = self.calculate_trajectory_balance(trajectories)
        reward = calculate_acquisition(state, action) + balance  # Combine acquisition and balance
        return reward  # Return the combined value as the reward

    def handle_stochastic_states(self, state, action):
        # Define a transition probability matrix for stochastic states
        transition_probabilities = {
            (0, 0): [(0.8, 0), (0.2, 1)],
            (0, 1): [(0.5, 0), (0.5, 1)],
            (1, 0): [(0.6, 0), (0.4, 1)],
            (1, 1): [(0.1, 0), (0.9, 1)],
        }

        # Get the possible transitions for the current state and action
        transitions = transition_probabilities.get((state[0], action), [])
        next_states = []
        for prob, next_state in transitions:
            if np.random.rand() < prob:  # Sample based on the transition probability
                next_states.append(next_state)

        return next_states  # Return the possible next states

    def get_dynamics_loss(self, state, action, next_state):
        # Implement your logic to calculate the dynamics loss
        # This is a placeholder for the actual dynamics loss calculation
        return tf.reduce_mean(tf.square(next_state - state))  # Example loss

    def get_loss(self, trajectories, state, action):
        # Calculate the overall loss based on trajectories, state, and action
        dynamics_loss = self.get_dynamics_loss(state, action, trajectories)
        # Add other loss components as needed
        return dynamics_loss  # Return the total loss

    def cvar_loss(self, rewards):
        # Calculate the Conditional Value-at-Risk (CVaR) loss
        sorted_rewards = np.sort(rewards)
        index = int(self.alpha * len(sorted_rewards))
        cvar = np.mean(sorted_rewards[:index])  # CVaR calculation
        return cvar

    def train_step(self, trajectories, state, action):
        """Perform a training step for the StochasticDBGFlowNetGenerator."""
        # Calculate the reward based on the current state, action, and trajectories
        reward = self.calculate_reward(state, action, trajectories)

        # Update the quantiles with the new reward
        self.update_quantiles(reward)

        # Sample a stochastic reward from the quantile distribution
        stochastic_reward = self.sample_reward()

        # Handle stochastic states
        next_states = self.handle_stochastic_states(state, action)

        # Calculate the CVaR loss
        cvar = self.cvar_loss([reward, stochastic_reward])

        # Calculate the overall loss
        loss = self.get_loss(trajectories, state, action)

        # Return the reward, the sampled stochastic reward, next possible states, and the loss
        return reward, stochastic_reward, next_states, loss, cvar

# Example usage
if __name__ == "__main__":
    generator = StochasticDBGFlowNetGenerator(num_quantiles=100)
    state = np.array([0])  # Example state
    action = np.random.choice([0, 1])  # Example action
    trajectories = [np.random.rand(5) for _ in range(10)]  # Example trajectories
    reward, stochastic_reward, next_states, loss, cvar = generator.train_step(trajectories, state, action)
    print(f"Reward: {reward}, Stochastic Reward: {stochastic_reward}, Next States: {next_states}, Loss: {loss}, CVaR: {cvar}")