import random
import numpy as np
import json

class SimplifiedTFBind8:
    def __init__(self, sequence_length=8):
        self.sequence_length = sequence_length
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.motifs = {
            'AT': 0.5,
            'CG': 0.7,
            'GC': 0.6,
            'TA': 0.4
        }

    def generate_sequence(self):
        return ''.join(random.choice(self.nucleotides) for _ in range(self.sequence_length))

    def calculate_score(self, sequence):
        score = 0
        reasoning = []

        # Check for motifs
        for motif, value in self.motifs.items():
            count = sequence.count(motif)
            if count > 0:
                motif_score = count * value
                score += motif_score
                reasoning.append(f"Found {count} {motif} motif(s): +{motif_score:.2f}")

        # Check for specific positions
        if sequence[0] == 'A' and sequence[-1] == 'T':
            score += 0.3
            reasoning.append("Starts with A and ends with T: +0.30")

        # Check for repeats
        repeats = sum(1 for i in range(1, len(sequence)) if sequence[i] == sequence[i-1])
        repeat_score = repeats * 0.1
        score += repeat_score
        if repeats > 0:
            reasoning.append(f"Found {repeats} adjacent repeat(s): +{repeat_score:.2f}")

        # Normalize score to be between 0 and 1
        normalized_score = min(max(score / 3, 0), 1)

        return normalized_score, reasoning

    def generate_dataset(self, num_samples=1000):
        dataset = []
        for _ in range(num_samples):
            sequence = self.generate_sequence()
            score, reasoning = self.calculate_score(sequence)
            dataset.append({
                'sequence': sequence,
                'score': score,
                'reasoning': reasoning
            })
        return dataset

    def save_dataset(self, filename, num_samples=1000):
        dataset = self.generate_dataset(num_samples)
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")

# Usage example
if __name__ == "__main__":
    tfbind8 = SimplifiedTFBind8()
    tfbind8.save_dataset("tfbind8_simplified_dataset.json", num_samples=1000)

    # Print a few examples
    with open("tfbind8_simplified_dataset.json", 'r') as f:
        dataset = json.load(f)

    print("Sample entries from the dataset:")
    for item in dataset[:5]:
        print(f"Sequence: {item['sequence']}")
        print(f"Score: {item['score']:.4f}")
        print("Reasoning:")
        for reason in item['reasoning']:
            print(f"- {reason}")
        print()