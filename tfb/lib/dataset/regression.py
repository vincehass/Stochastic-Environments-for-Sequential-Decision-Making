import numpy as np
from sklearn.model_selection import train_test_split

from lib.dataset.dataset import Dataset

class TFBind8Dataset(Dataset):
    def __init__(self, args, oracle):
        super().__init__(args, oracle)
        self._load_dataset()
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self):
        # Assume oracle provides the data
        x, y = self.oracle.get_initial_data()
        y = y.reshape(-1)
        self.train, self.valid, self.train_scores, self.valid_scores = train_test_split(x, y, test_size=0.1, random_state=int(self.rng.integers(2**32 - 1)))

    def create_all_stochastic_datasets(self, stick):
        self.train_thought = self.create_stochastic_data(stick, self.train)
        self.valid_thought = self.create_stochastic_data(stick, self.valid)
        print('\033[32mfinished creating stochastic datasets for train, valid\033[0m')
        
    def create_stochastic_data(self, stick, det_data):
        stochastic_data = []
        for curr_seq in det_data:
            curr_len = len(curr_seq)
            curr_rand_probs = self.rng.random(curr_len)
            curr_rand_actions = self.rng.randint(0, 4, curr_len)

            curr_actions = [curr_rand_actions[i] if curr_rand_probs[i] < stick else curr_seq[i] for i in range(curr_len)]
            stochastic_data.append(curr_actions)
        return stochastic_data

    def sample(self, n):
        indices = self.rng.choice(len(self.train), n)
        return ([self.train[i] for i in indices], [self.train_scores[i] for i in indices])
    
    def sample_with_stochastic_data(self, n):
        indices = self.rng.choice(len(self.train), n)
        return ([[self.train[i], self.train_thought[i]] for i in indices], [self.train_scores[i] for i in indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        train_seq, val_seq = [], []
        for x, score in zip(samples, scores):
            if self.rng.random() < 0.1:
                val_seq.append(x)
                val.append(score)
            else:
                train_seq.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train)).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val)).reshape(-1)
        self.train = np.concatenate((self.train, train_seq))
        self.valid = np.concatenate((self.valid, val_seq))
    
    def _tostr(self, seqs):
        return ["".join(map(str, x)) for x in seqs]

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.array(data[1])[indices]
        topk_prots = np.array(data[0])[indices]
        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        data = (np.concatenate((self.train, self.valid)), np.concatenate((self.train_scores, self.valid_scores)))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]))
        data = (seqs, scores)
        return self._top_k(data, k)


