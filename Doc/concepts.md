In this repository, for the molecule task, novelty, rewards, and the number of modes are key concepts used in training the generative model. Here's a breakdown of how each of these metrics is computed:

### 1. **Novelty**:

- Novelty measures how different a newly generated sample (e.g., a molecule) is compared to previously generated samples.
- In the code, novelty is typically computed by checking how often a molecule has been generated before. A new structure that hasn't been seen often is considered more "novel."
- This can be done by maintaining a count of previously generated samples and comparing the new one against this dataset.

### 2. **Rewards**:

- Rewards in the molecule task are calculated based on specific properties of the generated molecules. For example, you may want to reward molecules that have certain desirable chemical properties such as drug-likeness, synthesizability, or biological activity.
- The reward function can be parameterized using a **reward exponent** to modulate how the rewards are distributed among generated samples.
- Rewards are typically defined as \( R(x) \), where \( x \) represents the generated molecule, and they can be stochastic if different properties are sampled.

### 3. **Number of Modes**:

- The "number of modes" refers to the different high-reward regions or solutions in the space of generated molecules.
- In the GFlowNet framework, the model learns to sample from diverse modes by generating molecules that cover a wide range of the solution space, rather than focusing on a single optimal solution.
- To estimate the number of modes, you can cluster the generated molecules into different groups based on their chemical structure or properties and count the number of distinct clusters.

### **Pseudo-code for Novelty, Rewards, and Modes Calculation**:

```python
# Pseudo-code for calculating novelty, rewards, and modes in GFlowNets

def calculate_novelty(molecule, previous_samples):
    # Novelty is inversely proportional to the frequency of occurrence
    if molecule in previous_samples:
        return 1 / (previous_samples[molecule] + 1)
    else:
        return 1.0  # Full novelty if never seen before

def calculate_reward(molecule, reward_exponent=4):
    # Example: reward based on drug-likeness score (or other properties)
    score = compute_drug_likeness(molecule)
    return score ** reward_exponent  # Apply reward exponent

def calculate_modes(generated_samples, clustering_method):
    # Cluster generated molecules to estimate number of modes
    clusters = clustering_method(generated_samples)
    return len(set(clusters))  # Number of distinct modes (clusters)

# Main function to generate molecules and evaluate metrics
def gflow_net_run():
    generated_samples = []
    previous_samples = {}

    for i in range(total_iterations):
        molecule = generate_molecule()

        # Update sample count for novelty calculation
        previous_samples[molecule] = previous_samples.get(molecule, 0) + 1

        # Calculate metrics
        novelty = calculate_novelty(molecule, previous_samples)
        reward = calculate_reward(molecule)

        generated_samples.append(molecule)

    # Estimate number of modes
    num_modes = calculate_modes(generated_samples, clustering_method)
    return novelty, reward, num_modes
```

The code runs a GFlowNet process where molecules are generated, and metrics for novelty, rewards, and modes are computed at each step. The reward exponent, clustering method, and novelty calculation are central to guiding the GFlowNet to explore diverse and high-quality samples.

## Flow of the `run_tfbind.py` script

Let's break down the flow of the `run_tfbind.py` script, focusing on the key functions related to training, testing, and sample generation. This will help clarify when each part of the process occurs.

### Overview of the Flow

1. **Main Function (`main`)**: This is the entry point of the script where the experiment is initiated.
2. **Training Function (`train`)**: This function handles the training of the generator model.
3. **Sample Generation (`sample_batch`)**: This function generates samples based on the current state of the model.
4. **Rollout Worker (`RolloutWorker`)**: This class is responsible for executing the rollout process, which involves generating trajectories and collecting data for training.

### Detailed Breakdown

#### 1. Main Function

The `main` function is where the script starts executing. It typically looks like this:

```python
def main():
    args = parse_args()  # Parse command-line arguments
    oracle = get_oracle(args)  # Initialize the oracle
    dataset = get_dataset(args)  # Load the dataset
    train(args, oracle, dataset)  # Start the training process
```

- **`parse_args()`**: Parses command-line arguments.
- **`get_oracle(args)`**: Initializes the oracle, which is used for evaluating the quality of generated sequences.
- **`get_dataset(args)`**: Loads the dataset for training and evaluation.
- **`train(args, oracle, dataset)`**: Calls the training function, which is the core of the training process.

#### 2. Training Function

The `train` function is responsible for the main training loop. It looks like this:

```python
def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)  # Initialize the tokenizer
    proxy = construct_proxy(args, tokenizer, dataset=dataset)  # Create a proxy model
    generator = get_generator(args, tokenizer)  # Initialize the generator model

    rollout_worker, _ = train_generator(args, generator, oracle, proxy, tokenizer, dataset)  # Start training
```

- **`get_tokenizer(args)`**: Initializes the tokenizer used for processing sequences.
- **`construct_proxy(args, tokenizer, dataset)`**: Constructs a proxy model that helps in evaluating the generated sequences.
- **`get_generator(args, tokenizer)`**: Initializes the generator model that will be trained.
- **`train_generator(...)`**: This function handles the actual training process, including generating samples and updating the model.

#### 3. Sample Generation

Within the `train_generator` function, samples are generated using the `sample_batch` function. Here’s a simplified version of how it works:

```python
def train_generator(args, generator, oracle, proxy, tokenizer, dataset):
    for it in tqdm(range(args.gen_num_iterations + 1)):
        # Execute a training episode batch
        batch = sample_batch(args, rollout_worker, generator, dataset, oracle)
        # Process the batch for training
        dataset.add(batch)  # Add the generated samples to the dataset
```

- **`sample_batch(...)`**: This function generates samples based on the current state of the generator and the dataset. It collects sequences and their corresponding scores.

#### 4. Rollout Worker

The `RolloutWorker` class is crucial for generating trajectories during training. Here’s a simplified view of its `execute_train_episode_batch` method:

```python
class RolloutWorker:
    def execute_train_episode_batch(self, generator, it, dataset, use_rand_policy=False):
        visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones = self.rollout(generator, self.episodes_per_step, use_rand_policy=use_rand_policy)

        # Process the collected trajectories
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            # Additional processing...

        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
            }
        }
```

- **`self.rollout(...)`**: This method generates trajectories by interacting with the environment using the generator model.
- **`self.workers.pop_all()`**: This retrieves the completed trajectories for processing.

### Summary of Steps

1. **Initialization**: The script initializes necessary components (oracle, dataset, tokenizer, generator).
2. **Training Loop**: The `train` function enters a loop for a specified number of iterations.
3. **Sample Generation**: Within each iteration, `sample_batch` is called to generate new samples based on the current model state.
4. **Rollout Execution**: The `RolloutWorker` executes rollouts to generate trajectories, which are then processed and added to the dataset.
5. **Model Update**: The generator model is updated based on the collected data.

### Example Code Chunks

Here are some example code chunks to illustrate the flow:

**Main Function:**

```python
def main():
    args = parse_args()
    oracle = get_oracle(args)
    dataset = get_dataset(args)
    train(args, oracle, dataset)
```

**Training Function:**

```python
def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)
    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    generator = get_generator(args, tokenizer)
    rollout_worker, _ = train_generator(args, generator, oracle, proxy, tokenizer, dataset)
```

**Sample Generation:**

```python
def sample_batch(args, rollout_worker, generator, current_dataset, oracle):
    # Generate samples and scores
    samples = ([], [])
    while len(samples[0]) < args.num_sampled_per_round:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, dataset=current_dataset, use_rand_policy=False)
        # Collect states and scores
        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals)
    return (np.array(samples[0]), np.array(samples[1]))
```

### Conclusion

This breakdown should help clarify the flow of the `run_tfbind.py` script, particularly regarding training, testing, and sample generation. Each function plays a specific role in the overall process, and understanding their interactions is key to grasping how the training is conducted. If you have any further questions or need more details on specific parts, feel free to ask!

## Calculate the number of modes

To adapt the `calculate_num_modes` function based on the code from the provided GitHub repository, we will need to analyze the relevant sections of the code in `metrics.py` from the Distributional-GFlowNets project. The goal is to ensure that the mode calculation is robust and can handle multiple modes effectively.

### Steps to Adapt the Code

1. **Review the Code**: We will look for how modes are calculated in the `metrics.py` file.
2. **Identify Key Logic**: We will identify the logic used for calculating modes and any relevant distance metrics.
3. **Integrate Changes**: We will integrate the identified logic into our `calculate_num_modes` function.

### Example Code Adaptation

Assuming the relevant code from `metrics.py` uses a more sophisticated method for calculating modes, here’s how you might adapt your function:

#### Original `calculate_num_modes` Function

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_num_modes(sequences, distance_threshold=1):
    # Ensure sequences is a 2D numpy array
    sequences = np.array(sequences)
    if sequences.ndim == 1:
        sequences = sequences.reshape(-1, 1)

    # If sequences are strings, convert to integer representation
    if sequences.dtype.kind in ['U', 'S']:  # Unicode or byte string
        unique_chars = np.unique(sequences.ravel())
        char_to_int = {char: i for i, char in enumerate(unique_chars)}
        int_sequences = np.array([[char_to_int[char] for char in seq] for seq in sequences])
    else:
        int_sequences = sequences

    # Ensure int_sequences is 2D
    if int_sequences.ndim == 1:
        int_sequences = int_sequences.reshape(-1, 1)

    # Calculate pairwise Hamming distances
    distances = squareform(pdist(int_sequences, metric='hamming'))
    print("Pairwise distances:\n", distances)

    # Initialize modes
    modes = []
    for i in range(len(sequences)):
        # Check if the current sequence is a mode
        is_mode = True
        for j in modes:
            if distances[i, j] <= distance_threshold / len(sequences[0]):
                is_mode = False
                break
        if is_mode:
            modes.append(i)

    return len(modes)
```

#### Adapted Function Based on `metrics.py`

Here’s an adapted version of the function that incorporates a more flexible approach to mode calculation:

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_num_modes(sequences, distance_threshold=1):
    # Ensure sequences is a 2D numpy array
    sequences = np.array(sequences)
    if sequences.ndim == 1:
        sequences = sequences.reshape(-1, 1)

    # If sequences are strings, convert to integer representation
    if sequences.dtype.kind in ['U', 'S']:  # Unicode or byte string
        unique_chars = np.unique(sequences.ravel())
        char_to_int = {char: i for i, char in enumerate(unique_chars)}
        int_sequences = np.array([[char_to_int[char] for char in seq] for seq in sequences])
    else:
        int_sequences = sequences

    # Ensure int_sequences is 2D
    if int_sequences.ndim == 1:
        int_sequences = int_sequences.reshape(-1, 1)

    # Calculate pairwise Hamming distances
    distances = squareform(pdist(int_sequences, metric='hamming'))
    print("Pairwise distances:\n", distances)

    # Initialize modes
    modes = []
    for i in range(len(sequences)):
        # Check if the current sequence is a mode
        is_mode = True
        for j in modes:
            # Adjust the condition to allow for more flexibility
            if distances[i, j] < distance_threshold / len(sequences[0]):
                is_mode = False
                break
        if is_mode:
            modes.append(i)

    # Return the unique modes based on their indices
    unique_modes = np.unique(modes)
    return len(unique_modes)

# Test the function
if __name__ == "__main__":
    # Define a set of sequences for testing
    sequences = [
        "abc",
        "abd",
        "xyz",
        "xzy",
        "abc",
        "abz"
    ]

    # Set a distance threshold
    distance_threshold = 1  # You can experiment with this value

    # Calculate the number of modes
    num_modes = calculate_num_modes(sequences, distance_threshold)
    print(f"Number of modes: {num_modes}")
```

### Key Changes Made

1. **Flexible Mode Identification**: The condition for identifying modes has been adjusted to allow for more flexibility in determining whether a sequence can be considered a mode.
2. **Unique Modes Calculation**: The function now explicitly calculates unique modes using `np.unique(modes)` to ensure that only distinct modes are counted.

### Conclusion

This adapted function should provide a more robust way to calculate the number of modes in your sequences. You can further refine the distance threshold and the logic based on the specific requirements of your application. If you have access to the specific logic used in the `metrics.py` file, you can integrate that directly for even better results. If you need further assistance or specific details from the GitHub repository, please let me know!

The number of modes identified in a dataset can sometimes remain constant or even decrease due to several factors related to the data, the distance metric used, and the logic implemented in the mode calculation function. Here are some common reasons:

### 1. **Distance Metric Sensitivity**

- **Choice of Distance Metric**: The distance metric (e.g., Hamming, Euclidean) can significantly affect mode identification. If the metric is too sensitive or not sensitive enough, it may not capture the true differences between sequences.
- **Normalization Issues**: If the distance is not normalized correctly, it may lead to misleading results. For example, if the distance threshold is not appropriately set relative to the length of the sequences, it may cause all sequences to be considered too similar.

### 2. **Distance Threshold**

- **Too Restrictive Threshold**: If the distance threshold is set too low, it may prevent the identification of distinct modes. Sequences that are slightly different may be considered too close to each other, leading to fewer identified modes.
- **Increased Threshold**: Conversely, if the threshold is increased too much, it may cause sequences that are actually similar to be counted as separate modes, leading to an inflated count.

### 3. **Data Characteristics**

- **Homogeneity of Data**: If the dataset is homogeneous (i.e., the sequences are very similar), the number of modes will naturally be low. This can happen if the data is generated from a narrow distribution or if there is little diversity in the samples.
- **Redundant Sequences**: If the dataset contains many identical or nearly identical sequences, the mode calculation will reflect this redundancy, resulting in fewer unique modes.

### 4. **Implementation Logic**

- **Mode Identification Logic**: The logic used to determine whether a sequence is a mode can be overly restrictive. If the conditions for adding a sequence to the modes list are too stringent, it may lead to fewer modes being identified.
- **Unique Modes Calculation**: If the logic for calculating unique modes does not account for all relevant sequences, it may lead to a decrease in the reported number of modes.

### 5. **Changes in Data Over Time**

- **Dynamic Datasets**: In scenarios where the dataset changes over time (e.g., in iterative training processes), the number of modes may fluctuate based on the samples generated in each iteration. If the new samples are too similar to existing modes, the count may decrease.
- **Sampling Bias**: If the sampling method changes or if certain sequences are favored over others, it may lead to a decrease in the diversity of the generated samples, affecting the mode count.

### 6. **Statistical Variability**

- **Randomness in Generation**: If the sequences are generated randomly, statistical variability can lead to fluctuations in the number of modes identified. Some runs may yield more diverse samples than others purely by chance.

### Conclusion

To address issues with the number of modes not changing or decreasing, consider the following actions:

- **Review the Distance Metric**: Ensure that the chosen distance metric is appropriate for the data type and the problem at hand.
- **Adjust the Distance Threshold**: Experiment with different threshold values to find a balance that captures the desired level of diversity.
- **Analyze Data Characteristics**: Investigate the dataset for homogeneity or redundancy and consider augmenting the data if necessary.
- **Refine Mode Identification Logic**: Review and potentially simplify the logic used to identify modes to ensure it is not overly restrictive.

By understanding these factors, you can better diagnose and address issues related to mode identification in your datasets.

## Novelty and Diversity

In the context of the `run_tfbind.py` script, the novelty metric is typically calculated to assess how unique or different newly generated sequences are compared to a reference set of sequences. This is particularly important in generative models, where the goal is often to produce diverse outputs.

### How Novelty is Calculated

The novelty metric is usually calculated using the following steps:

1. **Define Reference Sequences**: A set of reference sequences is established, which can be the previously generated sequences or a dataset of known sequences.

2. **Count Novel Sequences**: For each new sequence generated, the algorithm checks whether it is present in the reference set. If it is not present, it is considered "novel."

3. **Calculate Novelty Score**: The novelty score is typically calculated as the ratio of novel sequences to the total number of new sequences generated. This can be expressed mathematically as:
   \[
   \text{Novelty} = \frac{\text{Number of Novel Sequences}}{\text{Total Number of Generated Sequences}}
   \]

### Example Code Snippet

Here’s a simplified example of how novelty might be calculated in the script:

```python
def calculate_novelty(new_sequences, reference_sequences):
    reference_set = set(tuple(seq) for seq in reference_sequences)
    novel_count = sum(1 for seq in new_sequences if tuple(seq) not in reference_set)
    return novel_count / len(new_sequences) if new_sequences else 0
```

### Reasons for Constant Novelty

There are several reasons why the novelty metric might remain constant over time:

1. **Homogeneity of Generated Sequences**:

   - If the generative model produces similar or identical sequences across iterations, the novelty score will remain low or constant. This can happen if the model has not learned to explore the output space effectively.

2. **Limited Reference Set**:

   - If the reference set is small or not diverse enough, even slightly different sequences may be counted as non-novel. This can lead to a situation where the novelty score does not change significantly.

3. **Convergence of the Model**:

   - If the generative model has converged to a specific mode or set of modes, it may produce similar outputs consistently, resulting in a constant novelty score.

4. **Insufficient Exploration**:

   - If the sampling strategy used during generation does not encourage exploration of the output space (e.g., using a greedy approach), the model may repeatedly generate the same or similar sequences, leading to low novelty.

5. **Static Dataset**:

   - If the reference dataset does not change over time, and the generated sequences do not introduce new variations, the novelty score will remain constant.

6. **Thresholding Issues**:
   - If the criteria for what constitutes a novel sequence are too strict, it may prevent the identification of new sequences, leading to a constant novelty score.

### Conclusion

To address issues with constant novelty, consider the following actions:

- **Enhance Model Diversity**: Implement techniques to encourage exploration in the generative model, such as adjusting sampling strategies or introducing randomness.
- **Expand Reference Set**: Use a more diverse or larger reference set to better capture the uniqueness of generated sequences.
- **Monitor Model Training**: Ensure that the model is adequately trained and not overfitting to a specific set of sequences.

By understanding how novelty is calculated and the factors that influence its constancy, you can better diagnose and improve the diversity of outputs from your generative model. If you have specific code snippets or sections from `run_tfbind.py` that you would like to discuss further, please let me know!

## Strategies to increase diversity

To modify the code in `run_tfbind.py` to avoid the novelty metric being constant, we can implement several strategies that encourage diversity in the generated sequences. Here are some approaches you can take:

1. **Increase Exploration**: Adjust the sampling strategy to encourage more exploration of the output space. This can be done by introducing randomness or using a more diverse sampling method.

2. **Dynamic Reference Set**: Update the reference set dynamically to include newly generated sequences, which can help in identifying novel sequences over time.

3. **Adjust the Novelty Calculation**: Modify the novelty calculation to account for a broader range of sequences or to use a different metric that captures diversity more effectively.

### Example Code Modifications

Here’s how you might implement these strategies in the novelty calculation function:

#### 1. Increase Exploration

You can modify the sampling strategy in the `sample_batch` function to introduce randomness. For example, you can use a stochastic policy instead of a greedy one:

```python
def sample_batch(args, rollout_worker, generator, current_dataset, oracle):
    samples = ([], [])
    while len(samples[0]) < args.num_sampled_per_round:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, dataset=current_dataset, use_rand_policy=True)  # Enable random policy
        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals)
    return (np.array(samples[0]), np.array(samples[1]))
```

#### 2. Dynamic Reference Set

You can modify the novelty calculation to include newly generated sequences in the reference set:

```python
def calculate_novelty(new_sequences, reference_sequences):
    reference_set = set(tuple(seq) for seq in reference_sequences)
    novel_count = sum(1 for seq in new_sequences if tuple(seq) not in reference_set)

    # Update the reference set with new sequences
    reference_sequences.extend(new_sequences)  # Add new sequences to the reference set
    return novel_count / len(new_sequences) if new_sequences else 0
```

#### 3. Adjust the Novelty Calculation

You can modify the novelty calculation to use a different metric or to consider the diversity of the generated sequences:

```python
def calculate_novelty(new_sequences, reference_sequences):
    reference_set = set(tuple(seq) for seq in reference_sequences)
    novel_count = sum(1 for seq in new_sequences if tuple(seq) not in reference_set)

    # Calculate diversity based on pairwise distances
    if len(new_sequences) > 1:
        distances = squareform(pdist(new_sequences, metric='hamming'))
        diversity_score = np.mean(distances)  # Average distance as a diversity measure
    else:
        diversity_score = 0

    # Combine novelty and diversity
    novelty_score = novel_count / len(new_sequences) if new_sequences else 0
    return novelty_score + diversity_score  # Adjust the return value as needed
```

### Summary of Changes

1. **Enable Random Sampling**: Modify the sampling strategy to encourage exploration.
2. **Dynamic Reference Set**: Update the reference set with newly generated sequences to ensure that the novelty calculation reflects the most current state of the model.
3. **Diversity Calculation**: Incorporate a diversity measure into the novelty calculation to provide a more comprehensive view of the generated sequences.

### Conclusion

By implementing these changes, you should be able to avoid the novelty metric being constant and encourage more diverse outputs from your generative model. Adjust the parameters and logic as needed based on your specific application and dataset.

### Other Strategies

1. **Increase the Number of Modes**:

   - **Data Augmentation**: Collect more diverse data to increase the number of modes.
   - **Synthetic Data Generation**: Use generative models to create synthetic data that is more varied.

2. **Adjust Distance Threshold**:

   - **Experiment with Threshold**: Fine-tune the distance threshold to better capture the diversity in your data.

3. **Enhance Model Diversity**:

   - **Sampling Strategies**: Implement techniques to encourage exploration in the generative model, such as adjusting sampling strategies or introducing randomness.
   - **Regularization Techniques**: Use regularization methods to prevent the model from becoming too confident in its predictions, which can help in exploring more diverse outputs.

4. **Feedback Mechanisms**:

   - **Reward Shaping**: Incorporate feedback mechanisms that reward diversity in the generated outputs. This can be done by adjusting the reward function to favor more diverse sequences.
   - **Diversity Penalties**: Introduce penalties for generating similar outputs, encouraging the model to explore more diverse regions of the output space.

5. **Iterative Training**:

   - **Dynamic Datasets**: Use dynamic datasets that change over time, allowing the model to encounter more diverse samples in each iteration.
   - **Feedback Loops**: Implement feedback loops that adjust the model's behavior based on the generated outputs, potentially leading to more diverse results.

6. **Ensemble Methods**:

   - **Multiple Generative Models**: Use an ensemble of generative models to produce diverse outputs. Each model can focus on different aspects of the data distribution.
   - **Generative Adversarial Networks (GANs)**: GANs can be trained to generate diverse outputs by having two networks compete with each other.

7. **Regularization Techniques**:

   - **KL Divergence Regularization**: Incorporate KL divergence regularization in the loss function to encourage diversity.
   - **Entropy Constraints**: Use entropy constraints to ensure that the model generates outputs with a certain level of diversity.

8. **Data Augmentation**:

   - **Synthetic Data Generation**: Use generative models to create synthetic data that is more varied.

9. **Feedback Mechanisms**:

   - **Reward Shaping**: Incorporate feedback mechanisms that reward diversity in the generated outputs. This can be done by adjusting the reward function to favor more diverse sequences.
   - **Diversity Penalties**: Introduce penalties for generating similar outputs, encouraging the model to explore more diverse regions of the output space.

10. **Iterative Training**:

- **Dynamic Datasets**: Use dynamic datasets that change over time, allowing the model to encounter more diverse samples in each iteration.
- **Feedback Loops**: Implement feedback loops that adjust the model's behavior based on the generated outputs, potentially leading to more diverse results.
