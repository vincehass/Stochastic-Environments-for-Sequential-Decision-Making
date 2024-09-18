import numpy as np

class TFBind8Wrapper:
    def __init__(self, args):
        self.sequence_to_score = {}
        with open('tfb/lib/data/tfbind.txt', 'r') as f:
            for line in f:
                sequence, score = line.strip().split('\t')
                self.sequence_to_score[sequence] = float(score)

    def __call__(self, x, batch_size=256):
        scores = []
        for sequence in x:
            score = self.sequence_to_score.get(sequence, 0)  # Default to 0 if sequence not found
            scores.append(score)
        return np.array(scores)

def get_oracle(args):
    return TFBind8Wrapper(args)