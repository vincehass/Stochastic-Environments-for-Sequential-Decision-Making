import numpy as np
from tfbind8_simplified import SimplifiedTFBind8

class TFBind8Wrapper:
    def __init__(self, args):
        self.task = SimplifiedTFBind8()
        self.dataset = self.task.generate_dataset(10000)  # Generate a large dataset
        self.sequence_to_score = {item['sequence']: item['score'] for item in self.dataset}

    def __call__(self, x, batch_size=256):
        scores = []
        for sequence in x:
            score = self.sequence_to_score.get(sequence, 0)  # Default to 0 if sequence not found
            scores.append(score)
        return np.array(scores)