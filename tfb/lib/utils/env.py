import torch
import numpy as np

class Vocab:
    def __init__(self, alphabet) -> None:
        self.stoi = {c: i for i, c in enumerate(alphabet)}
        self.itos = {i: c for i, c in enumerate(alphabet)}

class TokenizerWrapper:
    def __init__(self, vocab, dummy_process):
        self.vocab = vocab
        self.dummy_process = dummy_process
        self.eos_token = '%'
    
    def process(self, x):
        if not x:
            return torch.tensor([])
        
        if isinstance(x[0], list) and all(isinstance(s, int) for s in x[0]):
            return torch.tensor(x, dtype=torch.long)
        
        processed = self.process_fn(x)
        return torch.tensor(processed, dtype=torch.long)
    
    def process_fn(self, x):
        return [[self.vocab.stoi[c] for c in seq] for seq in x]
    
    @property
    def itos(self):
        return self.vocab.itos

    @property
    def stoi(self):
        return self.vocab.stoi

    def decode(self, state):
        return ''.join(self.itos[s] for s in state)

def distance(s1, s2):
    assert len(s1) == len(s2)
    return sum([int(s1[i] != s2[i]) for i in range(len(s1))])

def M_distance(s, M):
    return min([distance(s, ms) for ms in M])

def log_reward(s, M):
    return -M_distance(s, M)

def reward(s, M):
    return np.exp(log_reward(s, M))

class StochasticGFNEnvironment:
    def __init__(self, tokenizer, max_len, oracle, stick):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.oracle = oracle
        self.stick = stick
    
    def reset(self):
        return []
    
    def step(self, state, action):
        if len(state) >= self.max_len or action == len(self.tokenizer.vocab.stoi):
            return state, self.calculate_reward(state), True
        
        # Apply stochastic behavior based on stick parameter
        if np.random.random() < self.stick:
            # Stick to the current state
            return state, 0, False
        
        new_state = state + [action]
        return new_state, 0, False
    
    def calculate_reward(self, state):
        decoded_state = self.tokenizer.decode(state)
        return self.oracle(decoded_state)
    
    @property
    def action_space(self):
        return len(self.tokenizer.vocab.stoi) + 1  # Include EOS action
    
    @property
    def observation_space(self):
        return self.max_len

def get_tokenizer(args):
    if args.task == "tfbind":
        alphabet = ['A', 'C', 'T', 'G']
    
    vocab = Vocab(alphabet)
    tokenizer = TokenizerWrapper(vocab, dummy_process=(args.task != "amp"))
    return tokenizer

def get_environment(args, tokenizer, oracle):
    return StochasticGFNEnvironment(tokenizer, args.max_len, oracle, args.stick)

