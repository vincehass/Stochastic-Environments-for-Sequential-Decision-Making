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
        train_seq, train_y = batch

        # Convert train_seq and train_y to numpy arrays if they're not already
        train_seq = np.array(train_seq)
        train_y = np.array(train_y)

        # Print shapes for debugging
        print(f"Debug: self.train shape: {self.train.shape}")
        print(f"Debug: train_seq shape: {train_seq.shape}")
        print(f"Debug: self.train_scores shape: {self.train_scores.shape}")
        print(f"Debug: train_y shape: {train_y.shape}")

        # If self.train is empty, initialize it with the same shape as train_seq
        if self.train.size == 0:
            self.train = np.empty((0, train_seq.shape[1] if train_seq.ndim > 1 else 1), dtype=train_seq.dtype)

        # Ensure train_seq is 2-dimensional
        if train_seq.ndim == 1:
            train_seq = train_seq.reshape(-1, 1)

        # Ensure self.train is 2-dimensional
        if self.train.ndim == 1:
            self.train = self.train.reshape(-1, 1)

        # If self.train has only one column and train_seq has multiple columns, reshape self.train
        if self.train.shape[1] == 1 and train_seq.shape[1] > 1:
            self.train = np.repeat(self.train, train_seq.shape[1], axis=1)

        # Ensure the second dimension matches
        if self.train.shape[1] != train_seq.shape[1]:
            raise ValueError(f"Mismatch in sequence length. self.train has length {self.train.shape[1]}, but new data has length {train_seq.shape[1]}")

        self.train = np.concatenate((self.train, train_seq))
        self.train_scores = np.concatenate((self.train_scores, train_y))

        # Print final shapes for debugging
        print(f"Debug: Final self.train shape: {self.train.shape}")
        print(f"Debug: Final self.train_scores shape: {self.train_scores.shape}")

    def _tostr(self, seqs):
        if seqs.ndim == 2:
            return ["".join(map(str, seq)) for seq in seqs]
        else:
            return "".join(map(str, seqs))

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.array(data[1])[indices]
        topk_prots = np.array(data[0])[indices]
        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        # Ensure self.train and self.valid are 2D
        train = self.train if self.train.ndim == 2 else self.train.reshape(-1, 1)
        valid = self.valid if self.valid.ndim == 2 else self.valid.reshape(-1, 1)

        # Ensure train_scores and valid_scores are 1D
        train_scores = self.train_scores.flatten()
        valid_scores = self.valid_scores.flatten()

        # Print shapes for debugging
        print(f"Debug: train shape: {train.shape}")
        print(f"Debug: valid shape: {valid.shape}")
        print(f"Debug: train_scores shape: {train_scores.shape}")
        print(f"Debug: valid_scores shape: {valid_scores.shape}")

        # Ensure train and valid have the same number of columns
        max_cols = max(train.shape[1], valid.shape[1])
        if train.shape[1] < max_cols:
            train = np.pad(train, ((0, 0), (0, max_cols - train.shape[1])), mode='constant')
        if valid.shape[1] < max_cols:
            valid = np.pad(valid, ((0, 0), (0, max_cols - valid.shape[1])), mode='constant')

        # Concatenate the data
        all_sequences = np.concatenate((train, valid))
        all_scores = np.concatenate((train_scores, valid_scores))

        # Sort and get top k
        indices = np.argsort(all_scores)[::-1][:k]
        topk_scores = all_scores[indices]
        topk_sequences = all_sequences[indices]

        return self._tostr(topk_sequences), topk_scores

    def top_k_collected(self, k):
        # Ensure self.train and self.valid are 2D
        train = self.train[self.train_added:] if self.train[self.train_added:].ndim == 2 else self.train[self.train_added:].reshape(-1, 1)
        valid = self.valid[self.val_added:] if self.valid[self.val_added:].ndim == 2 else self.valid[self.val_added:].reshape(-1, 1)

        # Ensure train_scores and valid_scores are 1D
        train_scores = self.train_scores[self.train_added:].flatten()
        valid_scores = self.valid_scores[self.val_added:].flatten()

        # Print shapes for debugging
        print(f"Debug: train shape: {train.shape}")
        print(f"Debug: valid shape: {valid.shape}")
        print(f"Debug: train_scores shape: {train_scores.shape}")
        print(f"Debug: valid_scores shape: {valid_scores.shape}")

        # Ensure train and valid have the same number of columns
        max_cols = max(train.shape[1], valid.shape[1])
        if train.shape[1] < max_cols:
            train = np.pad(train, ((0, 0), (0, max_cols - train.shape[1])), mode='constant')
        if valid.shape[1] < max_cols:
            valid = np.pad(valid, ((0, 0), (0, max_cols - valid.shape[1])), mode='constant')

        # Concatenate the data
        seqs = np.concatenate((train, valid))
        scores = np.concatenate((train_scores, valid_scores))

        # Sort and get top k
        indices = np.argsort(scores)[::-1][:k]
        topk_scores = scores[indices]
        topk_seqs = seqs[indices]

        return self._tostr(topk_seqs), topk_scores


