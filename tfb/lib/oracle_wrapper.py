import numpy as np
import random
from typing import List, Tuple

class TFBind8Wrapper:
    def __init__(self, args):
        self.sequence_to_score = {}
        self.sequences = []
        self.scores = []
        self.vocab = ['A', 'C', 'G', 'T']
        self.sequence_length = 8

        # Load data from file
        with open('tfb/lib/data/tfbind.txt', 'r') as f:
            for line in f:
                sequence, score = line.strip().split('\t')
                score = float(score)
                self.sequence_to_score[sequence] = score
                self.sequences.append(sequence)
                self.scores.append(score)

        self.sequences = np.array(self.sequences)
        self.scores = np.array(self.scores)

    def __call__(self, x: List[str], batch_size: int = 256) -> np.ndarray:
        scores = []
        for sequence in x:
            score = self.sequence_to_score.get(sequence, 0)  # Default to 0 if sequence not found
            scores.append(score)
        return np.array(scores)

    def get_initial_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.sequences), num_samples, replace=False)
        return self.sequences[indices], self.scores[indices]

    def sample(self, num_samples: int = 1000) -> List[str]:
        return [''.join(random.choices(self.vocab, k=self.sequence_length)) for _ in range(num_samples)]

    def get_fitness(self, x: List[str]) -> np.ndarray:
        return self(x)

def get_oracle(args):
    return TFBind8Wrapper(args)

# Example usage
if __name__ == "__main__":
    class Args:
        pass

    args = Args()
    oracle = get_oracle(args)

    # Test get_initial_data
    initial_x, initial_y = oracle.get_initial_data(10)
    print("Initial data:")
    for seq, score in zip(initial_x, initial_y):
        print(f"{seq}: {score}")

    # Test sample and get_fitness
    sampled_sequences = oracle.sample(5)
    fitness_scores = oracle.get_fitness(sampled_sequences)
    print("\nSampled sequences and their fitness:")
    for seq, score in zip(sampled_sequences, fitness_scores):
        print(f"{seq}: {score}")