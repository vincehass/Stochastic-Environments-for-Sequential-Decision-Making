import json
import numpy as np

class TFBind8Wrapper:
    def __init__(self, args):
        with open("tfbind8_simplified_dataset.json", 'r') as f:
            self.dataset = json.load(f)
        self.sequence_to_score = {item['sequence']: item['score'] for item in self.dataset}

    def __call__(self, x, batch_size=256):
        scores = []
        for sequence in x:
            score = self.sequence_to_score.get(sequence, 0)  # Default to 0 if sequence not found
            scores.append(score)
        return np.array(scores)