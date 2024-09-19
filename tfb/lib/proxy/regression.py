import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from lib.model.mlp import MLP

class DropoutRegressor(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.num_tokens = int(args.vocab_size)  # Convert to int
        self.max_len = int(args.max_len)  # Convert to int
        self.tokenizer = tokenizer
        
        self.proxy_arch = 'mlp'  # Hard-coded value
        self.init_model()
        
        self.device = args.device

    def init_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.num_tokens * self.max_len, self.args.proxy_num_hid),
            nn.ReLU(),
            nn.Dropout(self.args.proxy_dropout),
            nn.Linear(self.args.proxy_num_hid, self.args.proxy_num_hid),
            nn.ReLU(),
            nn.Dropout(self.args.proxy_dropout),
            nn.Linear(self.args.proxy_num_hid, 1)
        )
        self.model.to(self.args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.args.proxy_learning_rate)

    def fit(self, data, reset=False):
        if reset:
            self.init_model()

        for _ in tqdm(range(self.args.proxy_num_iterations)):
            x, y = data.sample(self.args.proxy_num_per_minibatch)
            # Ensure x is a list of strings or a list of lists of integers
            if isinstance(x[0], str):
                x = [list(seq) for seq in x]
            x = self.tokenizer.process(x).to(self.device)
            x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].float()
            x = x.reshape(x.shape[0], -1)
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            
            output = self.model(x).squeeze(1)
            loss = F.mse_loss(output, y)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return {}

    def forward(self, curr_x):
        x = self.tokenizer.process(curr_x).to(self.device)
        x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].float()
        x = x.reshape(x.shape[0], -1)
        return self.model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))