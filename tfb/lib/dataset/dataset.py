import numpy as np
from collections import defaultdict

class Dataset:
    def __init__(self, args, oracle):
        self.args = args
        self.oracle = oracle
        self.rng = np.random.default_rng(args.seed)
        self.data = defaultdict(list)
        self.all_sequences = []
        self.load_data()

    def load_data(self):
        with open('tfb/lib/data/tfbind.txt', 'r') as f:
            for line in f:
                sequence, score = line.strip().split('\t')
                self.data['train'].append((sequence, float(score)))
                self.all_sequences.append(sequence)

    def get_all_sequences(self):
        return self.all_sequences

    def sample(self, n):
        indices = np.random.choice(len(self.data['train']), n, replace=False)
        return [self.data['train'][i] for i in indices]

    def add(self, batch):
        sequences, scores = batch
        for seq, score in zip(sequences, scores):
            self.data['train'].append((seq, score))
            self.all_sequences.append(seq)

    def top_k(self, k):
        sorted_data = sorted(self.data['train'], key=lambda x: x[1], reverse=True)
        return list(zip(*sorted_data[:k]))

    def top_k_collected(self, k):
        return self.top_k(k)

    def create_all_stochastic_datasets(self, stick):
        # This method is not needed for the simplified version, but we'll keep it empty for compatibility
        pass

    def sample_with_stochastic_data(self, n):
        return self.sample(n)

def get_dataset(args, oracle):
    return Dataset(args, oracle)