## Explanation of the algorithm in `tfb/run_tfbind.py`:

1. Data Preprocessing:

The data is handled by the `TFBind8Dataset` class. It typically contains:

- `self.train`: Training sequences (shape: [num_train_samples, sequence_length])
- `self.valid`: Validation sequences (shape: [num_valid_samples, sequence_length])
- `self.train_scores`: Scores for training sequences (shape: [num_train_samples])
- `self.valid_scores`: Scores for validation sequences (shape: [num_valid_samples])

2. Model MLP (Multi-Layer Perceptron):

The MLP is likely defined in a separate file, but it's used for various purposes in the GFlowNet. It typically has:

- Input layer: Size depends on the state representation
- Hidden layers: Multiple layers with ReLU activation
- Output layer: Size depends on the task (e.g., predicting scores or action probabilities)

3. GFlowNet Processing:

GFlowNet is a generative flow network that learns to sample sequences. It consists of:

- Forward policy: Generates sequences step by step
- Backward policy: Reconstructs the generation process backwards
- State representation: Encodes the current partial sequence
- Action space: Possible actions at each step (e.g., adding a nucleotide)

4. State Generation:

States are generated sequentially:

- Start with an empty sequence
- At each step, choose an action based on the forward policy
- Update the state representation
- Repeat until the sequence is complete

The state representation might have a shape like [batch_size, current_sequence_length, embedding_dim]

5. Learning:

The learning process involves:
a) Sampling trajectories:

- Generate sequences using the forward policy
- Shape: [batch_size, sequence_length]

b) Computing rewards:

- Use the oracle to evaluate generated sequences
- Shape: [batch_size]

c) Computing losses:

- Flow matching loss: Ensures consistency between forward and backward policies
- Detailed balance loss: Enforces the GFlowNet equilibrium condition
- Shape: Scalar values

d) Updating model parameters:

- Use gradient descent to minimize the losses

6. Prediction:

Prediction typically involves:

- Encoding a sequence into the state representation
- Passing it through the MLP
- Output could be a score prediction or action probabilities
- Shape of prediction: [batch_size, output_dim]

7. Sequence Generation:

To generate new sequences:

- Start with an empty state
- Use the forward policy to sample actions step by step
- Append chosen actions to build the sequence
- Repeat until the sequence is complete
- Shape of generated sequences: [num_generated, sequence_length]

Key Functions:

1. `sample_batch(args, rollout_worker, generator, dataset, oracle)`:

   - Generates a batch of sequences using the current policy
   - Evaluates the sequences using the oracle
   - Returns sequences and their scores

2. `train(args, oracle, dataset)`:

   - Main training loop
   - Samples batches, computes losses, updates model parameters
   - Logs metrics and saves checkpoints

3. `log_overall_metrics(args, dataset, collected)`:

   - Computes various metrics on the current dataset
   - Includes top-k scores, novelty, and distances

4. `main(args)`:
   - Sets up the experiment
   - Initializes the dataset, oracle, and models
   - Calls the training function

The algorithm iteratively generates sequences, evaluates them, and updates the model to improve the generation process. It aims to produce high-scoring, novel sequences that satisfy the given constraints.

This is a high-level overview, and the specific implementation details may vary. The exact tensor shapes would depend on the hyperparameters like batch size, sequence length, and model architecture specifics.

Let's expand on how the learning process is done in the context of the GFlowNet algorithm, focusing on the validation sequences (`self.valid`) with shape [num_valid_samples, sequence_length].

The learning process in GFlowNet involves several key steps:

1. Sampling trajectories:

   - The forward policy generates sequences step by step.
   - For each step, it selects an action (e.g., adding a nucleotide) based on the current state.
   - This process continues until complete sequences are generated.
   - The resulting trajectories have shape [batch_size, sequence_length].

2. Evaluating sequences:

   - The generated sequences are evaluated using the oracle (typically a pre-trained model or experimental data).
   - This produces scores for each sequence, with shape [batch_size].

3. Computing losses:

   - Flow matching loss: This ensures consistency between forward and backward policies.
     - It compares the probabilities of generating a sequence forward vs. reconstructing it backward.
     - Shape: Scalar value averaged over the batch.
   - Detailed balance loss: This enforces the GFlowNet equilibrium condition.
     - It ensures that the flow into each state equals the flow out of that state.
     - Shape: Scalar value averaged over the batch.

4. Updating model parameters:

   - The computed losses are used to update the parameters of the forward and backward policies.
   - This is typically done using gradient descent or a variant like Adam optimizer.

5. Validation:

   - The `self.valid` sequences are used to evaluate the model's performance on unseen data.
   - The current model generates sequences or predicts scores for the validation set.
   - These predictions are compared with the true scores of the validation sequences.
   - Metrics like mean squared error or correlation coefficient might be computed.
   - The validation process helps monitor overfitting and guide hyperparameter tuning.

6. Iterative improvement:
   - Steps 1-5 are repeated for multiple epochs or until convergence.
   - The model gradually learns to generate sequences that score highly according to the oracle.

Throughout this process, the validation sequences (`self.valid`) play a crucial role in assessing the model's generalization ability. They help ensure that the learned policy can generate high-quality sequences beyond just memorizing the training data.

The exact implementation details may vary, but this general framework describes how the GFlowNet learning process incorporates validation data to improve and evaluate sequence generation.

In the GFlowNet algorithm for TFBind, the reward calculation is a crucial part of the learning process. The rewards are typically derived from the oracle's evaluation of the generated sequences. Let's dive into the details of how these rewards are calculated:

1. Sequence Generation:

   - The GFlowNet generates a batch of sequences, let's call this batch S.
   - Shape of S: [batch_size, sequence_length]

2. Oracle Evaluation:

   - The oracle (which could be a pre-trained model or experimental data) evaluates each sequence in S.
   - Let's call the oracle function O(s) for a sequence s.
   - The oracle returns a score for each sequence.

3. Raw Reward Calculation:

   - For each sequence s in S, the raw reward R(s) is typically the score given by the oracle:
     R(s) = O(s)
   - Shape of R: [batch_size]

4. Reward Transformation:

   - Often, the raw rewards are transformed to better guide the learning process.
   - A common transformation is exponentiation:
     R'(s) = exp(β \* R(s))
   - Where β is a temperature parameter that controls the sharpness of the reward distribution.
   - This transformation helps emphasize high-scoring sequences.

5. Reward Normalization:

   - The transformed rewards are often normalized within the batch:
     R''(s) = R'(s) / Σ R'(s') for all s' in S
   - This ensures that the rewards sum to 1 within each batch.
   - Shape of R'': [batch_size]

6. Flow Matching:

   - In GFlowNet, the goal is to learn a policy that generates sequences with probabilities proportional to their rewards.
   - The flow F(s) for a sequence s should satisfy:
     F(s) ∝ R''(s)

7. Loss Calculation:

   - The learning process aims to minimize the discrepancy between the actual flows generated by the model and the target flows implied by the rewards.
   - Flow Matching Loss:
     L_FM = Σ (log F(s) - log R''(s))² for all s in S
   - Detailed Balance Loss:
     L_DB = Σ (log F(s→s') - log F(s'→s) - log R''(s') + log R''(s))² for all valid transitions s→s'

8. Total Loss:

   - The total loss is typically a weighted sum of these components:
     L*total = α * L*FM + (1-α) * L_DB
   - Where α is a hyperparameter balancing the two loss terms.

9. Optimization:
   - The model parameters are updated to minimize L_total using gradient descent or a variant.

This reward calculation and associated loss formulation guide the GFlowNet to learn a policy that generates high-scoring sequences according to the oracle, while maintaining the proper flow balance required by the GFlowNet framework.

The exact implementation might vary, especially in how the oracle scores are transformed into rewards and how the losses are computed and combined. The specific choices can significantly impact the learning dynamics and the quality of the generated sequences.

How to add a new metric to measure the number of modes in the generated sequences:

Adding a new metric to measure the number of modes in the generated sequences is a great idea. This metric can help assess the diversity of the generated sequences. Here's how you can add this metric to the `run_tfbind.py` file:

1. First, let's define a function to calculate the number of modes. We'll consider sequences as belonging to the same mode if they are within a certain edit distance of each other.

```python:tfb/run_tfbind.py
from scipy.spatial.distance import pdist, squareform
import numpy as np

def calculate_num_modes(sequences, distance_threshold=2):
    # Convert sequences to string representation
    seq_strings = [''.join(map(str, seq)) for seq in sequences]

    # Calculate pairwise edit distances
    distances = squareform(pdist(seq_strings, metric='hamming'))

    # Initialize modes
    modes = []
    for i, seq in enumerate(seq_strings):
        if not any(distances[i, j] <= distance_threshold for j in modes):
            modes.append(i)

    return len(modes)
```

2. Now, let's modify the `log_overall_metrics` function to include this new metric:

```python:tfb/run_tfbind.py
def log_overall_metrics(args, dataset, collected=False):
    k = 100
    top100 = dataset.top_k(k)
    top100_collected = dataset.top_k_collected(k) if collected else top100

    # Ensure dataset.train and dataset.valid are 2D
    train = dataset.train if dataset.train.ndim == 2 else dataset.train.reshape(-1, 1)
    valid = dataset.valid if dataset.valid.ndim == 2 else dataset.valid.reshape(-1, 1)

    # Ensure train and valid have the same number of columns
    max_cols = max(train.shape[1], valid.shape[1])
    if train.shape[1] < max_cols:
        train = np.pad(train, ((0, 0), (0, max_cols - train.shape[1])), mode='constant')
    if valid.shape[1] < max_cols:
        valid = np.pad(valid, ((0, 0), (0, max_cols - valid.shape[1])), mode='constant')

    # Concatenate the data
    all_sequences = np.concatenate((train, valid))

    # Calculate metrics
    max_100_collected_scores = np.max(top100_collected[1])
    novelty = calculate_novelty(top100_collected[0], all_sequences)
    top_100_collected_dists = calculate_dists(top100_collected[0])
    top_100_collected_scores = np.mean(top100_collected[1])
    top_100_dists = calculate_dists(top100[0])
    top_100_scores = np.mean(top100[1])

    # Calculate number of modes
    num_modes_collected = calculate_num_modes(top100_collected[0])
    num_modes_all = calculate_num_modes(top100[0])

    return {
        'max-100-collected-scores': max_100_collected_scores,
        'novelty': novelty,
        'top-100-collected-dists': top_100_collected_dists,
        'top-100-collected-scores': top_100_collected_scores,
        'top-100-dists': top_100_dists,
        'top-100-scores': top_100_scores,
        'num-modes-collected': num_modes_collected,
        'num-modes-all': num_modes_all
    }
```

3. Finally, update the `train` function to log these new metrics:

```python:tfb/run_tfbind.py
def train(args, oracle, dataset):
    # ... existing setup code ...

    for step in range(args.gen_num_iterations):
        # ... existing training code ...

        # Log metrics at each step
        curr_round_infos = log_overall_metrics(args, dataset, collected=True)

        wandb.log({
            "step": step,
            "max_100_collected_scores": curr_round_infos['max-100-collected-scores'],
            "novelty": curr_round_infos['novelty'],
            "top_100_collected_dists": curr_round_infos['top-100-collected-dists'],
            "top_100_collected_scores": curr_round_infos['top-100-collected-scores'],
            "top_100_dists": curr_round_infos['top-100-dists'],
            "top_100_scores": curr_round_infos['top-100-scores'],
            "num_modes_collected": curr_round_infos['num-modes-collected'],
            "num_modes_all": curr_round_infos['num-modes-all']
        })

        # ... rest of the training loop ...

    # ... rest of the function ...
```

These changes will add two new metrics:

- `num_modes_collected`: The number of modes in the top 100 collected sequences.
- `num_modes_all`: The number of modes in the top 100 sequences overall.

The `calculate_num_modes` function uses a simple clustering approach based on edit distance to estimate the number of distinct modes in the sequences. You can adjust the `distance_threshold` parameter to change how similar sequences need to be to be considered part of the same mode.

Remember to import the necessary libraries at the top of your file:

```python:tfb/run_tfbind.py
from scipy.spatial.distance import pdist, squareform
import numpy as np
```

After making these changes, the number of modes will be calculated and logged along with the other metrics at each step of the training process. This will allow you to track how the diversity of the generated sequences changes over time.

## Number of modes in the generated sequences for Stochastic Rewards:

The approach to calculating modes in the GFlowNet implementation with Stochastic Rewards is different from the one described in the original GFlowNet paper. Let's compare the two:

1. Original GFlowNet paper approach:

The original paper, as described in the blog post [1], uses a biochemical measure of diversity to estimate the number of distinct modes found. While the exact details aren't provided, it's mentioned that this approach is used to show that GFlowNet finds much more varied candidates compared to other methods.

2. Implementation of number of modes in the Stochastic Rewards:

The implementation in the GitHub repository [2] uses a different approach to calculate modes. Looking at the `metrics.py` file, we can see that modes are calculated using a clustering-based method:

```python
def get_num_modes(smiles_list, threshold=0.4):
    fps = get_fingerprints(smiles_list)
    dists = 1 - (np.dot(fps, fps.T) / (np.sum(fps, axis=1)[:, None] * np.sum(fps, axis=1)[None, :]))
    num_modes = 1
    for i in range(1, len(smiles_list)):
        if np.min(dists[i, :i]) > threshold:
            num_modes += 1
    return num_modes
```

This method:

- Converts molecules to fingerprints
- Calculates pairwise distances between fingerprints
- Counts a new mode if the minimum distance to all previous molecules is above a threshold

The key differences are:

1. Fingerprint-based approach: The GitHub implementation uses molecular fingerprints to represent molecules, which is a common cheminformatics technique.

2. Distance-based clustering: It uses a distance threshold to determine if a molecule belongs to a new mode, effectively performing a form of clustering.

3. Configurable threshold: The threshold for considering a new mode is adjustable, allowing for different levels of granularity in mode detection.

4. Deterministic calculation: This method provides a deterministic way to count modes, which may be easier to reproduce and compare across different runs or methods.

The approach in the GitHub repository is more explicit and potentially more generalizable to different types of molecular datasets. It provides a concrete, implementable method for counting modes, whereas the original paper's approach is not fully detailed in the available information.

Both approaches aim to quantify the diversity of generated molecules, but they use different techniques to do so. The GitHub implementation provides a more transparent and reproducible method for researchers to use and build upon.

[1]: https://folinoid.com/w/gflownet/
[2]: https://github.com/zdhNarsil/Distributional-GFlowNets/blob/main/mols/metrics.py
[3]: https://github.com/zdhNarsil/Distributional-GFlowNets/blob/main/mols/metrics.py

## Novelty:

Novelty is a measure of how many of the sequences in the generated set have not been seen before. It is calculated as the percentage of sequences in the generated set that are not present in the training set.

Novelty = (Number of unique sequences in generated set) / (Total number of sequences in generated set) \* 100

In the context of the GFlowNet algorithm for TFBind, novelty is calculated as follows:

1. Generate a set of sequences using the GFlowNet.

2. Calculate the edit distance between each generated sequence and the training set.

3. Count the number of generated sequences that have an edit distance of 0 (i.e., they are identical to a sequence in the training set).

4. Calculate the novelty as the percentage of sequences in the generated set that have an edit distance of 0.

5. Return the novelty value.

The novelty metric helps assess the diversity of the generated sequences. A higher novelty value indicates that the generated sequences are more diverse and less likely to be seen before.

## More Detailed Explanation:

In the current `tfb/run_tfbind.py` file, novelty is calculated using a simple set-based approach. Here's how it's implemented:

```python
def calculate_novelty(new_sequences, reference_sequences):
    reference_set = set(tuple(seq) for seq in reference_sequences)
    novel_count = sum(1 for seq in new_sequences if tuple(seq) not in reference_set)
    return novel_count / len(new_sequences)
```

This method:

1. Converts the reference sequences into a set of tuples for efficient lookup.
2. Counts how many sequences in the new set are not present in the reference set.
3. Returns the fraction of novel sequences.

In contrast, the novelty calculation in the Distributional GFlowNets repository [1] is more sophisticated:

```python
def novelty(gen_smiles, train_smiles):
    return 1 - (len(set(gen_smiles) & set(train_smiles)) / len(set(gen_smiles)))
```

Key differences:

1. The Distributional GFlowNets version uses SMILES representations of molecules, which are string-based encodings of molecular structures.
2. It calculates novelty as 1 minus the fraction of generated molecules that are also in the training set.
3. It uses set operations to efficiently compute the intersection of generated and training molecules.

The main conceptual difference is that the Distributional GFlowNets version focuses on the uniqueness of the entire generated set compared to the training set, while the current `tfb/run_tfbind.py` version calculates the fraction of individual sequences that are novel.

Both methods aim to quantify how different the generated samples are from the reference/training set, but they approach it from slightly different angles. The Distributional GFlowNets version might be more suitable for molecular generation tasks where the overall novelty of the generated set is of interest, while the current `tfb/run_tfbind.py` version provides a more granular view of novelty on a per-sequence basis.

[1] https://github.com/zdhNarsil/Distributional-GFlowNets/blob/main/mols/metrics.py

## Top-k Scores:

Top-k scores are a measure of the quality of the generated sequences. It is calculated as the average score of the top k sequences in the generated set.

Top-k Scores = (Sum of scores of top k sequences) / k

In the context of the GFlowNet algorithm for TFBind, top-k scores are calculated as follows:

1. Generate a set of sequences using the GFlowNet.

2. Evaluate the quality of each generated sequence using the oracle.

3. Select the top k sequences based on their scores.

4. Calculate the average score of these top k sequences.

5. Return the top-k scores value.

The top-k scores metric helps assess the quality of the generated sequences. A higher top-k scores value indicates that the generated sequences are of higher quality and score well according to the oracle.

## Top-k Distances:

Top-k distances are a measure of the diversity of the generated sequences. It is calculated as the average distance between the top k sequences in the generated set.

Top-k Distances = (Sum of distances between top k sequences) / k

In the context of the GFlowNet algorithm for TFBind, top-k distances are calculated as follows:

1. Generate a set of sequences using the GFlowNet.

2. Calculate the edit distance between each pair of generated sequences.

3. Select the top k sequences based on their scores.

4. Calculate the average distance between these top k sequences.

5. Return the top-k distances value.

The top-k distances metric helps assess the diversity of the generated sequences. A lower top-k distances value indicates that the generated sequences are more diverse and less likely to be similar to each other.
