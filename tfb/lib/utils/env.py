import torch
import numpy as np

class Vocab:
    def __init__(self, alphabet) -> None:
        self.stoi = {c: i for i, c in enumerate(alphabet)}
        self.itos = {i: c for i, c in enumerate(alphabet)}

class TokenizerWrapper:
    def __init__(self, vocab):
        if isinstance(vocab, Vocab):
            # If vocab is our custom Vocab object
            self.token_to_id = vocab.stoi
            self.id_to_token = vocab.itos
        elif hasattr(vocab, 'get_itos'):
            # If vocab is a torchtext Vocab object
            self.token_to_id = {token: i for i, token in enumerate(vocab.get_itos())}
            self.id_to_token = {i: token for i, token in enumerate(vocab.get_itos())}
        elif isinstance(vocab, dict):
            # If vocab is already a dictionary
            self.token_to_id = vocab
            self.id_to_token = {i: token for token, i in vocab.items()}
        else:
            # If vocab is a list or another iterable
            self.token_to_id = {token: i for i, token in enumerate(vocab)}
            self.id_to_token = {i: token for i, token in enumerate(vocab)}
        
        # Add '<UNK>' token if not present
        self.token_to_id.setdefault('<UNK>', len(self.token_to_id))
        self.id_to_token[self.token_to_id['<UNK>']] = '<UNK>'
        
        self.eos_token = '%'
        self.vocab = vocab  # Store the original vocab object
    
    def process(self, sequence):
        #print(f"Debug: process input type: {type(sequence)}, content: {sequence}")
        
        if isinstance(sequence, str):
            # If input is a single string
            return torch.tensor([self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in sequence], dtype=torch.long)
        elif isinstance(sequence, (list, tuple)):
            if len(sequence) == 0:
                return torch.tensor([], dtype=torch.long)
            if isinstance(sequence[0], (str, int)):
                # If input is a list/tuple of strings or integers
                return torch.tensor([self.token_to_id.get(token, self.token_to_id['<UNK>']) if isinstance(token, str) else token for token in sequence], dtype=torch.long)
            elif isinstance(sequence[0], (list, tuple)):
                # If input is a list/tuple of lists/tuples (batch of sequences)
                return torch.tensor([[self.token_to_id.get(token, self.token_to_id['<UNK>']) if isinstance(token, str) else token for token in seq] for seq in sequence], dtype=torch.long)
        
        raise ValueError(f"Unexpected input type: {type(sequence)}. Expected str, List[str], List[int], List[List[str]], List[List[int]], or their tuple equivalents.")
    
    def process_fn(self, x):
        if isinstance(x[0], str):
            return [[self.token_to_id.get(c, self.token_to_id['<UNK>']) for c in seq] for seq in x]
        elif isinstance(x[0], list):
            return x
        else:
            raise ValueError(f"Unexpected input type: {type(x[0])}")

    @property
    def itos(self):
        return self.id_to_token

    @property
    def stoi(self):
        return self.token_to_id

    def decode(self, state):
        if isinstance(state, torch.Tensor):
            state = state.tolist()
        return ''.join(self.itos.get(s, '<UNK>') for s in state)

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
    tokenizer = TokenizerWrapper(vocab)
    return tokenizer

def get_environment(args, tokenizer, oracle):
    return StochasticGFNEnvironment(tokenizer, args.max_len, oracle, args.stick)
