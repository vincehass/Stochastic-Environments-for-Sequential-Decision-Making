import torch
import torch.nn.functional as F
import torch.nn as nn

from lib.generator.base import GeneratorBase
from lib.model.mlp import MLP
from lib.acquisition_fn import get_acq_fn
import numpy as np
from itertools import chain

import itertools
from torch.distributions import Categorical
from tqdm import tqdm

import h5py
import time

from sklearn import manifold



class StochasticKLGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.device = args.device
        
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

        self.forward_dynamics = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=self.num_tokens, 
            num_hid=args.dynamics_num_hid,
            num_layers=args.dynamics_num_layers,
            max_len=self.max_len + 1,
            dropout=0,
            partition_init=args.gen_partition_init, 
            causal=args.gen_do_explicit_Z 
        )
        self.forward_dynamics.to(args.device)

        self.dynamics_opt = torch.optim.Adam(self.forward_dynamics.model_params(), args.dynamics_lr, weight_decay=args.dynamics_L2, betas=(0.9, 0.999))
        self.dynamics_clip = args.dynamics_clip

        self.ce_loss = nn.CrossEntropyLoss()

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()

        return loss.item()

    def get_loss(self, batch):
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        # Forward dynamics model output
        forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
        forward_model_outs = forward_model_outs[:-1, :, :]

        # Calculate forward dynamics loss
        forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))

        # KL Divergence Loss Calculation
        model_outs = self.model(x, None, return_all=True, lens=lens) 
        pol_logits = model_outs[:, :, :self.num_tokens] 
        log_flows = model_outs[:, :, -1] 

        
        
        # Ensure the shapes of pol_logits and log_flows are consistent
        pol_logits = self.logsoftmax2(pol_logits)[:-1]
        
        # Normalize log_flows to sum to 1
        log_flows = torch.exp(log_flows) / torch.sum(torch.exp(log_flows), dim=-1, keepdim=True)
        # Verify if pol_logits and log_flows are real probabilities
        pol_logits_sum = torch.sum(torch.exp(pol_logits), dim=-1)
        log_flows_sum = torch.sum(log_flows, dim=-1)

        # Check if the sums are close to 1, indicating they are probabilities
        pol_logits_prob_check = torch.allclose(pol_logits_sum, torch.ones_like(pol_logits_sum))
        log_flows_prob_check = torch.allclose(log_flows_sum, torch.ones_like(log_flows_sum))

        if not pol_logits_prob_check:
            print("pol_logits do not sum to 1, indicating they are not probabilities.")
        if not log_flows_prob_check:
            print("log_flows do not sum to 1, indicating they are not probabilities.")
        s = s.swapaxes(0, 1) 
        thought_s = thought_s.swapaxes(0, 1)
        n = (s.shape[0] - 1) * s.shape[1]

        pol_logits = pol_logits.reshape((n, self.num_tokens)) 
        pol_logits = pol_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_logits = pol_logits.reshape(s[1:].shape) 

        log_flows = log_flows[1:].transpose(1, 0)
        
        # Debugging shapes of tensors
        print(f"pol_logits shape: {pol_logits.shape}")
        print(f"log_flows shape: {log_flows.shape}")
        print(f"real_actions shape: {real_actions.shape}")
        

        # KL Divergence Loss
        kl_loss = 0
        epsilon = 1e-10  # Small constant to avoid log(0)
        for t in range(len(pol_logits)):
            print(f"pol_logits[t] shape: {pol_logits[t].shape}")
            print(f"log_flows[:, t] shape: {log_flows[:, t].shape}")
            print(f"pol_logits[t] value : {pol_logits[t]}")
            print(f"log_flows[:, t] value: {log_flows[:, t]}")
            # Add epsilon to avoid NaN
            kl_loss += ((pol_logits[t] + epsilon) - log_flows[:, t].unsqueeze(1)).sum()  
            print(f"kl_loss at t={t}: {kl_loss}")

        # Check if kl_loss is NaN
        if torch.isnan(kl_loss):
            print("kl_loss is NaN, returning early.")
            return float('nan'), {}  # Return NaN to avoid unpacking error

        # Entropy Ratio Estimation
        H_high = -torch.mean(torch.exp(pol_logits) * pol_logits)  # High entropy states
        H_low = -torch.mean(torch.exp(pol_logits) * pol_logits)  # Low entropy states
        gamma = 0.5  # Example value, can be parameterized
        R_entropy = H_high / (gamma * H_high + (1 - gamma) * H_low)

        # Adjust the forward policy
        pol_logits = pol_logits * R_entropy
        pol_logits = pol_logits - torch.logsumexp(pol_logits, dim=-1, keepdim=True)  # Normalize

        # Total Loss
        total_loss = kl_loss + forward_dynamics_loss

        return total_loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out    



class StochasticKL3GFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1


        # Initialize policy and forward dynamics models
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.forward_dynamics = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=self.num_tokens, 
            num_hid=args.dynamics_num_hid,
            num_layers=args.dynamics_num_layers,
            max_len=self.max_len + 1,
            dropout=0,
            partition_init=args.gen_partition_init, 
            causal=args.gen_do_explicit_Z 
        )
        self.forward_dynamics.to(args.device)

        # Initialize optimizers
        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.dynamics_opt = torch.optim.Adam(self.forward_dynamics.model_params(), args.dynamics_lr, weight_decay=args.dynamics_L2, betas=(0.9, 0.999))
        self.dynamics_clip = args.dynamics_clip
        
        self.dynamics_off_pol = args.dynamics_off_pol
        if self.dynamics_off_pol:
            self.dynamics_buffer = ReplayBuffer(self.max_len)
            self.dynamics_sample_size = args.dynamics_sample_size
            self.dynamics_off_pol_rounds = args.dynamics_off_pol_rounds

        # Entropy ratio estimation settings
        self.gamma = torch.tensor([0.25, 0.5, 0.75], device=args.device)  # Example quantiles
        self.device = args.device
        self.ce_loss = nn.CrossEntropyLoss()
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def entropy_ratio(self, H_high, H_low):
        """Compute entropy ratio and return as tensor."""
        r_gamma = H_high / (self.gamma * H_high + (1 - self.gamma) * H_low)
        return r_gamma

    def kl_divergence_loss(self, forward_policy, backward_policy, r_gamma):
        """KL Divergence Loss using a distribution instead of a scalar gamma."""
        backward_policy = backward_policy.T  # Transpose the tensor

        # Add a small constant to avoid log(0)
        epsilon = 1e-10
        forward_policy = torch.clamp(forward_policy, min=epsilon)  # Clamp to avoid log(0)
        backward_policy = torch.clamp(backward_policy, min=epsilon)  # Clamp to avoid log(0)
        r_gamma = torch.clamp(r_gamma, min=epsilon)  # Clamp to avoid log(0)

        # Compute KL divergence
        kl_loss = torch.sum(forward_policy * (torch.log(forward_policy) - torch.log(backward_policy)), dim=-1)
        print(f"kl_loss shape: {kl_loss.shape}")
        print(f"r_gamma shape: {r_gamma.shape}")
        # Incorporate the distribution of gamma
        
        kl_loss *= r_gamma.mean()  # Element-wise multiplication with the distribution
        print(f"kl_loss shape after: {kl_loss.shape}")
        print(f"kl_loss: {kl_loss}")
        return torch.abs(kl_loss.sum())

    def dynamics_loss(self, policy, mu_pi, r_gamma):
        """Compute dynamics loss balancing exploration and exploitation."""
        H_pi = -torch.sum(policy * torch.log(policy), dim=-1)  # Policy entropy
        return -torch.sum(mu_pi * H_pi * (torch.log(r_gamma) + (1 - self.gamma) * (1 - H_pi) * torch.log(1 - r_gamma)))
    
    
    def train_step(self, input_batch):
        #info = {}
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()
        

        return loss, info
    
    
    
    
    # def get_dynamics_loss(self):
    #     """Compute dynamics loss from replay buffer samples."""
    #     info = {}
    #     strs, thought_strs, r = self.dynamics_buffer.sample(self.dynamics_sample_size)
    #     s = self.tokenizer.process(strs).to(self.device)
    #     thought_s = self.tokenizer.process(thought_strs).to(self.device)

    #     lens = [len(i) for i in strs]
        
    #     real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)
    #     inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
    #     inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
    #     inp[:, :inp_x.shape[1], :] = inp_x
    #     x = inp.reshape(s.shape[0], -1).to(self.device).detach()
    #     inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
    #     inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
    #     inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
    #     x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        
    #     forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
    #     forward_model_outs = forward_model_outs[:-1, :, :]
    #     forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))
    #     info['forward_dynamics_loss'] = forward_dynamics_loss
    #     forward_model_logits = forward_model_outs.detach().log_softmax(-1)
    #     info['forward_model_logits'] = forward_model_logits
        
        
    #     return forward_dynamics_loss, info

    def train_step_dy(self, input_batch):
        """Train the model, including KL divergence and dynamics loss."""
        # Sample input and compute main loss
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()

        rets = [loss.item()]
        
        # Compute dynamics loss with or without off-policy correction
        if self.dynamics_off_pol:
            total_dynamics_loss = 0
            for _ in range(self.dynamics_off_pol_rounds):
                dynamics_loss, dynamics_info = self.get_dynamics_loss()  # Ensure this returns the correct structure
                dynamics_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
                self.dynamics_opt.step()
                self.dynamics_opt.zero_grad()
                total_dynamics_loss += dynamics_loss.item()
            dynamics_loss = total_dynamics_loss / self.dynamics_off_pol_rounds
        else:
            dynamics_loss = info['forward_dynamics_loss']
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
            self.dynamics_opt.step()
            self.dynamics_opt.zero_grad()
            dynamics_loss = dynamics_loss.item()
        
        rets.append(dynamics_loss)
        

        return rets  # Ensure this returns the correct structure

    def get_loss(self, batch):
        """Compute total loss for policy and dynamics."""
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])
        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)
        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()
        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
        forward_model_outs = forward_model_outs[:-1, :, :]
        forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))
        forward_model_logits = forward_model_outs.detach().log_softmax(-1)

        forward_model_logits = forward_model_logits.gather(-1, real_actions.unsqueeze(-1)).squeeze(-1) 
        info['forward_dynamics_loss'] = forward_dynamics_loss
        info['forward_model_logits'] = forward_model_logits

        # Log-likelihood difference computation
        model_outs = self.model(x, None, return_all=True, lens=lens) 
        policy_logits = model_outs[:, :, :self.num_tokens] 
        policy_back_logits = model_outs[:, :, self.num_tokens:-1] 
        log_flows = model_outs[:, :, -1] 

        policy_logits = self.logsoftmax2(policy_logits)[:-1] 
        policy_back_logits = self.logsoftmax2(policy_back_logits)[1:] 

        # Reshape the policy logits and back logits to match the shape of the actions
        mask = s.eq(self.num_tokens)
        s = s.swapaxes(0, 1)
        thought_s = thought_s.swapaxes(0, 1)
        n = (s.shape[0] - 1) * s.shape[1]

        policy_logits = policy_logits.reshape((n, self.num_tokens))
        policy_logits = policy_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        policy_logits = policy_logits.reshape(s[1:].shape)
        policy_back_logits = policy_back_logits.reshape((n, self.num_tokens))
        policy_back_logits = policy_back_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        policy_back_logits = policy_back_logits.reshape(s[1:].shape)

        # Masking the end of the sequence
        mask = mask[:, 1:].swapaxes(0, 1).logical_not().float()

        # Log-likelihood difference computation
        ll_diff = torch.zeros((policy_logits.shape)).to(self.device)
        ll_diff += log_flows[:-1]
        ll_diff += policy_logits
        ll_diff += forward_model_logits
        log_flows = log_flows[1:].transpose(1, 0) 

        # Log-Flows and End-Log-Flows  
        r = r.clamp(min=self.reward_exp_min).log()
        r = r.unsqueeze(-1).repeat(1, log_flows.shape[1]) 
        lens = torch.tensor(lens).long()
        end_pos = lens - 1 - 1

        mask_for_backward = mask.clone().detach().transpose(1, 0) 
        if (end_pos >= mask_for_backward.size(0)).any():
            raise ValueError(f"end_pos contains out-of-bounds indices: {end_pos}")

        mask_for_backward[torch.arange(end_pos.shape[0], device=self.device), end_pos] -= 1

        end_log_flow = mask_for_backward * log_flows + (1 - mask_for_backward) * r
        end_log_flow = end_log_flow.transpose(1, 0)

        # Compute high and low entropy for current and next state using all quantiles
        H_high = torch.quantile(policy_logits, self.gamma, dim=-1, keepdim=True)  # Use all quantiles defined by self.gamma
        H_low = torch.quantile(policy_back_logits, 1 - self.gamma, dim=-1, keepdim=True)  # Use all quantiles defined by 1 - self.gamma
        print(f"H_high shape: {H_high.shape}")
        print(f"H_low shape: {H_low.shape}")
        # Ensure r_gamma is a tensor before returning
        r_gamma = self.entropy_ratio(H_high, H_low)

        if not isinstance(r_gamma, torch.Tensor):
            raise ValueError("r_gamma must be a tensor, but got: {}".format(type(r_gamma)))
        r_gamma = r_gamma.to(torch.float32)
        #print(f"Debug: r_gamma type after calculation: {type(r_gamma)}")  # Check the type of r_gamma
        
        print(f"r_gamma shape outside: {r_gamma.shape}")
        # Update log-likelihood difference
        ll_diff -= end_log_flow
        ll_diff -= policy_back_logits
        ll_diff *= mask

        # Compute KL divergence loss using the distribution
        kl_divergence_loss = self.kl_divergence_loss(end_log_flow, log_flows, r_gamma)
        
        info['kl_divergence_loss'] = kl_divergence_loss
        print(f"r_gamma shape inside: {r_gamma.mean(dim=-1)}")
        info['r_gamma'] = r_gamma.mean(dim=-1)  # Add r_gamma to the info dictionary
        loss = (ll_diff ** 2).sum() / mask.sum() + kl_divergence_loss
        info['gfn_loss'] = loss.item()

        return loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out      





class StochasticKL2GFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1


        # Initialize policy and forward dynamics models
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.forward_dynamics = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=self.num_tokens, 
            num_hid=args.dynamics_num_hid,
            num_layers=args.dynamics_num_layers,
            max_len=self.max_len + 1,
            dropout=0,
            partition_init=args.gen_partition_init, 
            causal=args.gen_do_explicit_Z 
        )
        self.forward_dynamics.to(args.device)

        # Initialize optimizers
        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.dynamics_opt = torch.optim.Adam(self.forward_dynamics.model_params(), args.dynamics_lr, weight_decay=args.dynamics_L2, betas=(0.9, 0.999))
        self.dynamics_clip = args.dynamics_clip
        
        self.dynamics_off_pol = args.dynamics_off_pol
        if self.dynamics_off_pol:
            self.dynamics_buffer = ReplayBuffer(self.max_len)
            self.dynamics_sample_size = args.dynamics_sample_size
            self.dynamics_off_pol_rounds = args.dynamics_off_pol_rounds

        # Entropy ratio estimation settings
        self.gamma = args.gamma
        self.device = args.device
        self.ce_loss = nn.CrossEntropyLoss()
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def entropy_ratio(self, H_high, H_low):
        """Compute entropy ratio and return as tensor."""
        r_gamma = H_high / (self.gamma * H_high + (1 - self.gamma) * H_low)
        return torch.tensor(r_gamma)

    def kl_divergence_loss(self, forward_policy, backward_policy, r_gamma):
        """KL Divergence Loss."""
        backward_policy = backward_policy.T  # Transpose the tensor

        # Add a small constant to avoid log(0)
        epsilon = 1e-10
        forward_policy = torch.clamp(forward_policy, min=epsilon)  # Clamp to avoid log(0)
        backward_policy = torch.clamp(backward_policy, min=epsilon)  # Clamp to avoid log(0)
        r_gamma = torch.clamp(r_gamma, min=epsilon)  # Clamp to avoid log(0)

        # print(f"forward_policy shape: {forward_policy.shape}")
        # print(f"backward_policy shape: {backward_policy.shape}")
        

        # Compute KL divergence
        kl_loss = torch.sum(forward_policy * (torch.log(forward_policy) - torch.log(backward_policy)))# + torch.log(r_gamma)))

        return kl_loss

    def dynamics_loss(self, policy, mu_pi, r_gamma):
        """Compute dynamics loss balancing exploration and exploitation."""
        H_pi = -torch.sum(policy * torch.log(policy), dim=-1)  # Policy entropy
        return -torch.sum(mu_pi * H_pi * (torch.log(r_gamma) + (1 - self.gamma) * (1 - H_pi) * torch.log(1 - r_gamma)))
    
    
    def train_step(self, input_batch):
        #info = {}
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()
        

        return loss, info
    
    
    
    
    # def get_dynamics_loss(self):
    #     """Compute dynamics loss from replay buffer samples."""
    #     info = {}
    #     strs, thought_strs, r = self.dynamics_buffer.sample(self.dynamics_sample_size)
    #     s = self.tokenizer.process(strs).to(self.device)
    #     thought_s = self.tokenizer.process(thought_strs).to(self.device)

    #     lens = [len(i) for i in strs]
        
    #     real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)
    #     inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
    #     inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
    #     inp[:, :inp_x.shape[1], :] = inp_x
    #     x = inp.reshape(s.shape[0], -1).to(self.device).detach()
    #     inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
    #     inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
    #     inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
    #     x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        
    #     forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
    #     forward_model_outs = forward_model_outs[:-1, :, :]
    #     forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))
    #     info['forward_dynamics_loss'] = forward_dynamics_loss
    #     forward_model_logits = forward_model_outs.detach().log_softmax(-1)
    #     info['forward_model_logits'] = forward_model_logits
        
        
    #     return forward_dynamics_loss, info

    def train_step_dy(self, input_batch):
        """Train the model, including KL divergence and dynamics loss."""
        # Sample input and compute main loss
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()

        rets = [loss.item()]
        
        # Compute dynamics loss with or without off-policy correction
        if self.dynamics_off_pol:
            total_dynamics_loss = 0
            for _ in range(self.dynamics_off_pol_rounds):
                dynamics_loss, dynamics_info = self.get_dynamics_loss()  # Ensure this returns the correct structure
                dynamics_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
                self.dynamics_opt.step()
                self.dynamics_opt.zero_grad()
                total_dynamics_loss += dynamics_loss.item()
            dynamics_loss = total_dynamics_loss / self.dynamics_off_pol_rounds
        else:
            dynamics_loss = info['forward_dynamics_loss']
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
            self.dynamics_opt.step()
            self.dynamics_opt.zero_grad()
            dynamics_loss = dynamics_loss.item()
        
        rets.append(dynamics_loss)
        

        return rets  # Ensure this returns the correct structure

    def get_loss(self, batch):
        """Compute total loss for policy and dynamics."""
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])
        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)
        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()
        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
        forward_model_outs = forward_model_outs[:-1, :, :]
        forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))
        forward_model_logits = forward_model_outs.detach().log_softmax(-1)

        forward_model_logits = forward_model_logits.gather(-1, real_actions.unsqueeze(-1)).squeeze(-1) 
        info['forward_dynamics_loss'] = forward_dynamics_loss
        info['forward_model_logits'] = forward_model_logits

        # Log-likelihood difference computation
        model_outs = self.model(x, None, return_all=True, lens=lens) 
        policy_logits = model_outs[:, :, :self.num_tokens] 
        policy_back_logits = model_outs[:, :, self.num_tokens:-1] 
        log_flows = model_outs[:, :, -1] 

        policy_logits = self.logsoftmax2(policy_logits)[:-1] 
        policy_back_logits = self.logsoftmax2(policy_back_logits)[1:] 

        # Reshape the policy logits and back logits to match the shape of the actions
        mask = s.eq(self.num_tokens)
        s = s.swapaxes(0, 1)
        thought_s = thought_s.swapaxes(0, 1)
        n = (s.shape[0] - 1) * s.shape[1]

        policy_logits = policy_logits.reshape((n, self.num_tokens))
        policy_logits = policy_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        policy_logits = policy_logits.reshape(s[1:].shape)
        policy_back_logits = policy_back_logits.reshape((n, self.num_tokens))
        policy_back_logits = policy_back_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        policy_back_logits = policy_back_logits.reshape(s[1:].shape)

        # Masking the end of the sequence
        mask = mask[:, 1:].swapaxes(0, 1).logical_not().float()

        # Log-likelihood difference computation
        ll_diff = torch.zeros((policy_logits.shape)).to(self.device)
        ll_diff += log_flows[:-1]
        ll_diff += policy_logits
        ll_diff += forward_model_logits
        log_flows = log_flows[1:].transpose(1, 0) 

        # Log-Flows and End-Log-Flows  
        r = r.clamp(min=self.reward_exp_min).log()
        r = r.unsqueeze(-1).repeat(1, log_flows.shape[1]) 
        lens = torch.tensor(lens).long()
        end_pos = lens - 1 - 1

        mask_for_backward = mask.clone().detach().transpose(1, 0) 
        if (end_pos >= mask_for_backward.size(0)).any():
            raise ValueError(f"end_pos contains out-of-bounds indices: {end_pos}")

        mask_for_backward[torch.arange(end_pos.shape[0], device=self.device), end_pos] -= 1

        end_log_flow = mask_for_backward * log_flows + (1 - mask_for_backward) * r
        end_log_flow = end_log_flow.transpose(1, 0)

        # Compute high and low entropy for current and next state
        H_high = torch.quantile(policy_logits, self.gamma, dim=-1, keepdim=True)
        H_low = torch.quantile(policy_back_logits, 1-self.gamma, dim=-1, keepdim=True)
        r_gamma = self.entropy_ratio(H_high, H_low)
            # Ensure r_gamma is a tensor before returning
        if not isinstance(r_gamma, torch.Tensor):
            raise ValueError("r_gamma must be a tensor, but got: {}".format(type(r_gamma)))
        r_gamma = r_gamma.to(torch.float32)
        #print(f"Debug: r_gamma type after calculation: {type(r_gamma)}")  # Check the type of r_gamma
        
        #print(f"r_gamma shape: {r_gamma.shape}")
        # Update log-likelihood difference
        ll_diff -= end_log_flow
        ll_diff -= policy_back_logits
        ll_diff *= mask
        # Compute KL divergence loss
        kl_divergence_loss = self.kl_divergence_loss(end_log_flow, log_flows, r_gamma)
        #print("kl_divergence_loss:", kl_divergence_loss)
        info['kl_divergence_loss'] = kl_divergence_loss
        info['r_gamma'] = r_gamma# Add r_gamma to the info dictionary
        loss = (ll_diff ** 2).sum() / mask.sum() + kl_divergence_loss
        info['gfn_loss'] = loss.item()
        #print("gfn-loss:", loss)
        return loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out      



class DBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.num_hid,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.device = args.device
        
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()
        return loss, info

    def get_loss(self, batch):
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens) 
        inp[:, :inp_x.shape[1], :] = inp_x
        
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        lens = [len(i) for i in strs]

        model_outs = self.model(x, None, return_all=True, lens=lens) 
        pol_logits = model_outs[:, :, :self.num_tokens] 
        pol_back_logits = model_outs[:, :, self.num_tokens:-1] 
        log_flows = model_outs[:, :, -1] 

        pol_logits = self.logsoftmax2(pol_logits)[:-1] 
        pol_back_logits = self.logsoftmax2(pol_back_logits)[1:] 
    
        mask = s.eq(self.num_tokens)

        s = s.swapaxes(0, 1) 
        thought_s = thought_s.swapaxes(0, 1)

        n = (s.shape[0] - 1) * s.shape[1]

        pol_logits = pol_logits.reshape((n, self.num_tokens)) 
        pol_logits = pol_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_logits = pol_logits.reshape(s[1:].shape) 
        pol_back_logits = pol_back_logits.reshape((n, self.num_tokens))
        pol_back_logits = pol_back_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_back_logits = pol_back_logits.reshape(s[1:].shape)

        mask = mask[:, 1:].swapaxes(0, 1).logical_not().float() 

        ll_diff = torch.zeros((pol_logits.shape)).to(self.device) 
        ll_diff += log_flows[:-1] 
        ll_diff += pol_logits
        log_flows = log_flows[1:].transpose(1, 0) 
        r = r.clamp(min=self.reward_exp_min).log()
        r = r.unsqueeze(-1).repeat(1, log_flows.shape[1]) 
        lens = torch.tensor(lens).long()
        end_pos = lens - 1 - 1
        mask_for_backward = mask.clone().detach().transpose(1, 0) 
        mask_for_backward[torch.arange(end_pos.shape[0], device=self.device), end_pos] -= 1
        end_log_flow = mask_for_backward * log_flows + (1 - mask_for_backward) * r
        end_log_flow = end_log_flow.transpose(1, 0)
        ll_diff -= end_log_flow

        ll_diff -= pol_back_logits
        ll_diff *= mask
        loss = (ll_diff ** 2).sum() / mask.sum()
        info = {'gfn_loss': loss.item()}

        return loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out    


class ReplayBuffer(object):
    def __init__(self, max_len, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.strs = np.zeros((max_size, max_len), dtype=int)
        self.thought_strs = np.zeros((max_size, max_len), dtype=int)
        self.rewards = np.zeros((max_size,))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, strs, thought_strs, rews):
        for i in range(len(strs)):
            curr_str, curr_thought_str, curr_rew = strs[i], thought_strs[i], rews[i]
            self.strs[self.ptr] = curr_str
            self.thought_strs[self.ptr] = curr_thought_str
            self.rewards[self.ptr] = curr_rew

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        sampled_strs = self.strs[ind]
        sampled_thought_strs = self.thought_strs[ind]
        sampled_rs = self.rewards[ind]

        return sampled_strs, sampled_thought_strs, sampled_rs
       

class StochasticDBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        
        self.stick = args.stick
        num_outputs = self.num_tokens + self.num_tokens + 1
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=num_outputs, 
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))
        self.device = args.device
        
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

        self.forward_dynamics = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=self.num_tokens, 
            num_hid=args.dynamics_num_hid,
            num_layers=args.dynamics_num_layers,
            max_len=self.max_len + 1,
            dropout=0,
            partition_init=args.gen_partition_init, 
            causal=args.gen_do_explicit_Z 
        )
        print (self.forward_dynamics)
        self.forward_dynamics.to(args.device)

        self.dynamics_opt = torch.optim.Adam(self.forward_dynamics.model_params(), args.dynamics_lr, weight_decay=args.dynamics_L2, betas=(0.9, 0.999))
        self.dynamics_clip = args.dynamics_clip

        self.ce_loss = nn.CrossEntropyLoss()

        self.dynamics_off_pol = args.dynamics_off_pol
        if self.dynamics_off_pol:
            self.dynamics_buffer = ReplayBuffer(self.max_len)
            self.dynamics_sample_size = args.dynamics_sample_size
            self.dynamics_off_pol_rounds = args.dynamics_off_pol_rounds

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()

        rets = [loss.item()]

        if self.dynamics_off_pol:
            total_dynamics_loss = 0.
            for dynamics_off_pol_round in range(self.dynamics_off_pol_rounds):
                dynamics_loss = self.get_dynamics_loss()
                dynamics_loss.backward()
                if self.dynamics_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
                self.dynamics_opt.step()
                self.dynamics_opt.zero_grad()
                total_dynamics_loss += dynamics_loss.item()
            dynamics_loss = total_dynamics_loss / self.dynamics_off_pol_rounds
        else:
            dynamics_loss = info['forward_dynamics_loss']
            dynamics_loss.backward()
            if self.dynamics_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.dynamics_clip)
            self.dynamics_opt.step()
            self.dynamics_opt.zero_grad()
            dynamics_loss = dynamics_loss.item()

        rets.append(dynamics_loss)

        return rets

    def get_dynamics_loss(self):
        info = {}
        strs, thought_strs, r = self.dynamics_buffer.sample(self.dynamics_sample_size)

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
        forward_model_outs = forward_model_outs[:-1, :, :]

        forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))

        forward_model_logits = forward_model_outs.detach().log_softmax(-1)

        return forward_dynamics_loss

    def get_loss(self, batch):
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        lens = [len(i) for i in strs]

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        inp_x_thought = F.one_hot(thought_s[:, 1:], num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp_thought = torch.zeros(thought_s.shape[0], self.max_len, self.num_tokens)
        inp_thought[:, :inp_x_thought.shape[1], :] = inp_x_thought
        x_thought = inp_thought.reshape(thought_s.shape[0], -1).to(self.device).detach()

        real_actions = s[:, 1:].clamp(0, self.num_tokens - 1).long().transpose(1, 0)

        if not self.dynamics_off_pol:
            forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
            forward_model_outs = forward_model_outs[:-1, :, :]

            forward_dynamics_loss = self.ce_loss(forward_model_outs.reshape(-1, forward_model_outs.shape[-1]), real_actions.reshape(-1))

            forward_model_logits = forward_model_outs.detach().log_softmax(-1)
        else:
            with torch.no_grad():
                forward_model_outs = self.forward_dynamics.forward_dynamics_model(x, x_thought, None, return_all=True, lens=lens)
            forward_model_outs = forward_model_outs[:-1, :, :]

            forward_dynamics_loss = 1e9

            forward_model_logits = forward_model_outs.log_softmax(-1)
        
        forward_model_logits = forward_model_logits.gather(-1, real_actions.unsqueeze(-1)).squeeze(-1) 

        info['forward_dynamics_loss'] = forward_dynamics_loss

        model_outs = self.model(x, None, return_all=True, lens=lens) 
        pol_logits = model_outs[:, :, :self.num_tokens] 
        pol_back_logits = model_outs[:, :, self.num_tokens:-1] 
        log_flows = model_outs[:, :, -1] 

        pol_logits = self.logsoftmax2(pol_logits)[:-1] 
        pol_back_logits = self.logsoftmax2(pol_back_logits)[1:] 

        mask = s.eq(self.num_tokens)

        s = s.swapaxes(0, 1)
        thought_s = thought_s.swapaxes(0, 1)

        n = (s.shape[0] - 1) * s.shape[1]

        pol_logits = pol_logits.reshape((n, self.num_tokens))
        pol_logits = pol_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_logits = pol_logits.reshape(s[1:].shape)
        pol_back_logits = pol_back_logits.reshape((n, self.num_tokens))
        pol_back_logits = pol_back_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
        pol_back_logits = pol_back_logits.reshape(s[1:].shape)

        mask = mask[:, 1:].swapaxes(0, 1).logical_not().float() 

        ll_diff = torch.zeros((pol_logits.shape)).to(self.device)
        ll_diff += log_flows[:-1]
        ll_diff += pol_logits
        ll_diff += forward_model_logits
        log_flows = log_flows[1:].transpose(1, 0) 
        r = r.clamp(min=self.reward_exp_min).log()
        r = r.unsqueeze(-1).repeat(1, log_flows.shape[1]) 
        lens = torch.tensor(lens).long()
        end_pos = lens - 1 - 1
        mask_for_backward = mask.clone().detach().transpose(1, 0) 
        mask_for_backward[torch.arange(end_pos.shape[0], device=self.device), end_pos] -= 1
        end_log_flow = mask_for_backward * log_flows + (1 - mask_for_backward) * r
        end_log_flow = end_log_flow.transpose(1, 0)
        ll_diff -= end_log_flow
        ll_diff -= pol_back_logits
        ll_diff *= mask
        loss = (ll_diff ** 2).sum() / mask.sum()

        return loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        assert not return_all

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out    
    

class TrajectoryBalanceGFlowNetGenerator(GeneratorBase):
    def __init__(self, num_quantiles=100, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.num_quantiles = num_quantiles
        self.quantiles = np.linspace(-1, 1, num_quantiles)  # Define quantile values
        self.quantile_counts = np.zeros(num_quantiles)  # Count of rewards in each quantile
        self.total_count = 0  # Total count of rewards for normalization

    def update_quantiles(self, reward):
        # Update the quantile counts based on the received reward
        bin_index = np.digitize(reward, self.quantiles) - 1
        if 0 <= bin_index < self.num_quantiles:
            self.quantile_counts[bin_index] += 1  # Increment count for the quantile
            self.total_count += 1  # Increment total count

    def sample_reward(self):
        # Sample a reward based on the quantile distribution
        probabilities = self.quantile_counts / self.total_count if self.total_count > 0 else np.ones(self.num_quantiles) / self.num_quantiles
        return np.random.choice(self.quantiles, p=probabilities)

    def calculate_trajectory_balance(self, trajectories):
        # Implement your logic to calculate the trajectory balance
        # This is a placeholder for the actual trajectory balance calculation
        balance = np.mean([np.sum(trajectory) for trajectory in trajectories])  # Example calculation
        return balance

    def calculate_reward(self, state, action, trajectories):
        # Calculate the reward based on the trajectory balance
        balance = self.calculate_trajectory_balance(trajectories)
        reward = get_acq_fn(state, action) + balance  # Combine acquisition and balance
        return reward  # Return the combined value as the reward

    def trajectory_balance_loss(self, state, action, trajectories):
        # Example of how to integrate trajectory balance into the loss function
        reward = self.calculate_reward(state, action, trajectories)

        # Update the quantiles with the new reward
        self.update_quantiles(reward)

        # Sample a stochastic reward from the quantile distribution
        stochastic_reward = self.sample_reward()

        # Return the reward and the sampled stochastic reward
        return reward, stochastic_reward





# # Example usage
# if __name__ == "__main__":
#     generator = TrajectoryBalanceGFlowNetGenerator(num_quantiles=100)
#     state = np.random.rand(10)  # Example state
#     action = np.random.choice([0, 1])  # Example action
#     trajectories = [np.random.rand(5) for _ in range(10)]  # Example trajectories
#     reward, stochastic_reward = generator.trajectory_balance_loss(state, action, trajectories)
#     print(f"Reward: {reward}, Stochastic Reward: {stochastic_reward}")