import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from lib.generator.base import GeneratorBase
from lib.model.mlp import MLP
import torch.nn.functional as F
import numpy as np


#SAC
class SACGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.gamma = 0.99
        self.alpha = 0.2 #More stochastic (temperature or entripy regularization)It controls the trade-off between exploration and exploitation in the policy.
        # Ensure state_dim matches the input size
        self.actor_args = {
            'state_dim': self.max_len * self.num_tokens,  # Ensure this matches the input size
            'hidden_dim': 128,
            'action_dim': self.num_tokens,
            'device': 'cpu',
            'lr': args.gen_learning_rate
        }
        self.critic_args = {
            'state_dim': self.max_len * self.num_tokens,  # Ensure this matches the input size
            'action_dim': self.num_tokens,
            'device': 'cpu',
            'lr': args.gen_learning_rate
        }
        
        self.device = args.device

        self.actor = nn.Sequential(
            nn.Linear(self.actor_args['state_dim'], self.actor_args['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.actor_args['hidden_dim'], self.actor_args['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.actor_args['hidden_dim'], self.actor_args['action_dim'])
        )
        self.critic1 = nn.Sequential(
            nn.Linear(self.critic_args['state_dim'] + self.critic_args['action_dim'], 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(self.critic_args['state_dim'] + self.critic_args['action_dim'], 1)
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + 
            list(self.critic1.parameters()) + 
            list(self.critic2.parameters()), 
            lr=self.actor_args['lr']
        )

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters())

    def train_step(self, input_batch):
        loss, info = self.learn_from(input_batch)
        return loss, info

    def learn_from(self, batch):
        strs, thought_strs, r = zip(*batch["bulk_trajs"])
        info = {}
        
        # Process input strings
        s = self.tokenizer.process(strs).to(self.args.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.args.device)

        # Convert rewards to tensor
        r = torch.tensor(r).to(self.args.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)

        # Prepare input for the model
        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens).to(self.args.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.args.device)  # Ensure x has the correct shape

        # Ensure x has the correct shape for the actor
        if x.shape[1] != self.actor_args['state_dim']:  # Check if the shape matches
            raise ValueError(f"Expected input shape {self.actor_args['state_dim']} but got {x.shape[1]}")

        # Adjust the input preparation to ensure correct dimensions
        if x.shape[1] < self.actor_args['state_dim']:
            # Pad x to match the expected state_dim
            padding = self.actor_args['state_dim'] - x.shape[1]
            x = F.pad(x, (0, padding), "constant", 0)  # Pad with zeros

        # Ensure s has the correct shape for the actor
        if s.shape[1] != self.actor_args['state_dim']:
            if s.shape[1] < self.actor_args['state_dim']:
                # Pad s to match the expected state_dim
                padding = self.actor_args['state_dim'] - s.shape[1]
                s = F.pad(s, (0, padding), "constant", 0)  # Pad with zeros
            else:
                s = s[:, :self.actor_args['state_dim']]  # Truncate if too large

        # Ensure thought_s has the correct shape for the actor
        if thought_s.shape[1] != self.actor_args['state_dim']:
            if thought_s.shape[1] < self.actor_args['state_dim']:
                padding = self.actor_args['state_dim'] - thought_s.shape[1]
                thought_s = F.pad(thought_s, (0, padding), "constant", 0)  # Pad with zeros
            else:
                thought_s = thought_s[:, :self.actor_args['state_dim']]  # Truncate if too large

        lens = [len(i) for i in strs]
        
        # Model output logic
        model_outs = self.actor(x)  # Ensure x matches the expected input size
        loss = 0  # Initialize loss
        accepted_samples = 0  # Count accepted samples

        # Calculate Q-values and losses
        current_q1 = self.critic1(torch.cat([x, model_outs], dim=-1))  # Ensure concatenation is correct
        current_q2 = self.critic2(torch.cat([x, model_outs], dim=-1))
        
        # Calculate target Q-values
        with torch.no_grad():
            thought_s = thought_s.to(self.actor[0].weight.dtype)  # Convert dtype
            next_action_probs = torch.softmax(self.actor(thought_s), dim=-1)
            next_q1 = self.critic1(torch.cat([thought_s, next_action_probs], dim=-1))
            next_q2 = self.critic2(torch.cat([thought_s, next_action_probs], dim=-1))
            next_q = torch.min(next_q1, next_q2)
            target_q = r + self.gamma * next_q

        
        # Convert current_q1, current_q2, and target_q to Float just before calculating loss
        current_q1 = current_q1.float()
        current_q2 = current_q2.float()
        target_q = target_q.float()
        
        
        # Calculate critic loss
        critic_loss1 = nn.MSELoss()(current_q1, target_q)
        critic_loss2 = nn.MSELoss()(current_q2, target_q)
        critic_loss = critic_loss1 + critic_loss2

        # Calculate actor loss
        s = s.to(self.actor[0].weight.dtype)  # Convert input to match actor's weight dtype
        action_probs = torch.softmax(self.actor(s), dim=-1)  # Ensure data type matches
        log_probs = torch.log(action_probs + 1e-8)
        min_q = torch.min(current_q1, current_q2)
        actor_loss = (self.alpha * log_probs - min_q).mean()

        # Combine losses
        loss = critic_loss + actor_loss

        # Policy Improvement acceptance criteria
        # Convert actor_loss and critic_loss to tensors before using them
        acceptance_ratio = torch.exp(torch.tensor(actor_loss.item()) - torch.tensor(critic_loss.item()))
        if torch.rand(1).item() < acceptance_ratio.item():  # Accept the sample
            accepted_samples += 1
            info['accepted'] = True
        else:
            info['accepted'] = False

        # Update optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return loss and additional info
        info['critic_loss'] = critic_loss.item()
        info['actor_loss'] = actor_loss.item()
        info['acceptance_prob'] = accepted_samples / (1 if accepted_samples > 0 else 1)  # Avoid division by zero

        return loss.item(), info

    def forward(self, x, lens, return_all=False):
        # Prepare input for the model
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)

        # Get action probabilities from the actor
        action_probs = self.actor(inp)

        # Calculate Q-values from critics
        current_q1 = self.critic1(torch.cat([inp, action_probs], dim=-1))
        current_q2 = self.critic2(torch.cat([inp, action_probs], dim=-1))

        if return_all:
            return action_probs, current_q1, current_q2
        else:
            return action_probs  # Return only action probabilities if not returning all


#MARS
class MARSGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)  # Call the base class constructor
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.device = args.device

        # Initialize MLP model
        self.net = MLP(
            num_tokens=self.num_tokens,
            num_outputs=self.num_tokens,
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        ).to(self.device)  # Move model to the specified device

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))

    def parameters(self):
        return self.net.parameters()

    def train_step(self, input_batch):
        loss, info = self.learn_from(input_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), info

    def learn_from(self, batch):
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        # Process input strings
        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        # Convert rewards to tensor
        r = torch.tensor(r).to(self.device).clamp(min=0)

        # Prepare input for the model
        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device)

        # Define or compute the mask as needed
        mask = None  # Ensure mask is defined
        action_logits = self.net(x, mask)  # Pass the mask argument
        log_probs = F.log_softmax(action_logits, dim=-1)

        # Calculate loss
        # Ensure thought_s has the correct shape for gathering
        thought_s = thought_s.unsqueeze(1)  # This should be [batch_size, 1]
        action_log_probs = log_probs.gather(1, thought_s.unsqueeze(1))  # This should work now
        loss = -torch.mean(action_log_probs * r)  # Use rewards for loss calculation

        return loss.item(), info

    def forward(self, x, lens, return_all=False):
        # Prepare input for the model
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)

        # Get action probabilities from the model
        action_logits = self.net(inp, None)  # Ensure to pass the required mask argument

        if return_all:
            return action_logits  # Return logits if return_all is True
        else:
            return F.softmax(action_logits, dim=-1)  # Return probabilities if not returning all

    def sample_many(self, env, n_samples):
        trajs = []
        for _ in range(n_samples):
            state = env.reset()
            done = False
            traj = {'states': [], 'actions': [], 'rewards': []}
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)  # Ensure state is on the correct device
                logits = self.net(state_tensor, None)  # Ensure to pass the required mask argument
                action = Categorical(logits=logits).sample().item()
                next_state, reward, done, _ = env.step(action)
                traj['states'].append(state)
                traj['actions'].append(action)
                traj['rewards'].append(reward)
                state = next_state
            trajs.append(traj)
        return trajs
    



#Metropolis-Hastings

# class MHAgent:
#     def __init__(self, args, envs):
#         self.envs = envs
#         self.batch = [i.reset() for i in envs] # The N MCMC chains
#         self.bufsize = args.bufsize
#         self.nactions = args.ndim*2
#         self.model = None

#     def parameters(self):
#         return []

#     def sample_many(self, mbsize, all_visited):
#         r = np.float32([i[1] for i in self.batch])
#         a = np.random.randint(0, self.nactions, self.bufsize)
#         # step: obs(s), r, s, reverse_a
#         steps = [self.envs[j].step(a[j], s=self.batch[j][2]) for j in range(self.bufsize)]
#         rp = np.float32([i[1] for i in steps])
#         A = rp / r
#         U = np.random.uniform(0,1,self.bufsize)
#         for j in range(self.bufsize):
#             if A[j] > U[j]: # Accept
#                 self.batch[j] = (None, rp[j], steps[j][2])
#                 all_visited.append(tuple(steps[j][2]))
#         return []

#     def learn_from(self, *a):
#         return None
    

#PPO
class PPOGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)

        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.device = args.device
        
        self.out_coef = 1.0  # Add this line to define out_coef

        # Initialize the model (policy network)
        self.model = MLP(
            num_tokens=self.num_tokens,
            num_outputs=self.num_tokens,
            num_hid=args.gen_num_hidden,
            num_layers=args.gen_num_layers,
            max_len=self.max_len,
            dropout=0,
            partition_init=args.gen_partition_init,
            causal=args.gen_do_explicit_Z
        )
        self.model.to(args.device)

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.model.model_params(), lr=args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))

        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # Clipping parameter for PPO
        self.ppo_epochs = 5  # Number of PPO epochs

    def parameters(self):
        return self.model.parameters()

    def train_step(self, input_batch):
        loss, info = self.learn_from(input_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), info

    def learn_from(self, batch):
        info = {}
        strs, thought_strs, r = zip(*batch["bulk_trajs"])

        # Process input strings
        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        # Convert rewards to tensor
        r = torch.tensor(r).to(self.device).clamp(min=0)

        # Prepare input for the model
        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens).to(self.device)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device)

        # Add a mask argument (you need to define what mask should be)
        mask = None  # Define or compute the mask as needed
        action_logits = self.model(x, mask)  # Pass the mask argument
        log_probs = F.log_softmax(action_logits, dim=-1)

        # Calculate advantages (using TD error or any other method)
        values = r  # Assuming rewards are used as values for simplicity
        advantages = values - values.mean()  # Simple advantage calculation

        # Adjust the shape of advantages to match ratio
        advantages = advantages.unsqueeze(1)  # Change shape from [32] to [32, 1]

        # Debugging: Print shapes of ratio and advantages
        print("Shape of log_probs:", log_probs.shape)  # Add this line
        print("Shape of advantages:", advantages.shape)  # Add this line

        # Calculate the surrogate loss
        ratio = torch.exp(log_probs - log_probs.detach())  # Importance sampling ratio
        surrogate_loss = ratio * advantages  # Now this should work without error
        clipped_surrogate_loss = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        # Debugging: Print shapes of ratio and clipped_surrogate_loss
        print("Shape of ratio:", ratio.shape)  # Add this line
        print("Shape of clipped_surrogate_loss:", clipped_surrogate_loss.shape)  # Add this line

        # Total loss
        actor_loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean()

        # Total loss
        total_loss = actor_loss

        info['actor_loss'] = actor_loss.item()

        return total_loss, info

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)

        # Ensure to pass the required mask argument
        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out




#DeterminsticDBGFlowNetGenerator

class DeterminsticDBGFlowNetGenerator(GeneratorBase):
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
            num_hid=128,
            num_layers=2,
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



    



#Random Trajectory
# import numpy as np

class RandomTrajGenerator:
    def __init__(self, args):
        self.args = args

    def parameters(self):
        return []  # Random trajectory generator doesn't have learnable parameters

    def learn_from(self, trajs):
        # Random trajectory generator doesn't learn
        return 0.0

    def sample_many(self, env, n_samples):
        trajs = []
        for _ in range(n_samples):
            state = env.reset()
            done = False
            traj = {'states': [], 'actions': [], 'rewards': []}
            while not done:
                action = np.random.randint(self.args.action_dim)
                next_state, reward, done, _ = env.step(action)
                traj['states'].append(state)
                traj['actions'].append(action)
                traj['rewards'].append(reward)
                state = next_state
            trajs.append(traj)
        return trajs



#MHStochasticDBGFlowNetGenerator



    
#Metropolis-Hastings generator
class MHGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.out_coef = args.gen_output_coef
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.pad_tok = 1
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        # Initialize MLP model
        self.model = MLP(
            num_tokens=self.num_tokens,  # Use the vocab size from args
            num_outputs=self.num_tokens,  # Match outputs to vocab size
            num_hid=128,
            num_layers=2,
            max_len=self.max_len,
            dropout=0,
            partition_init=True, 
            causal=True
        )
        self.device = args.device
        self.model.to(args.device)
        
        # Initialize any additional parameters or models as needed
        self.opt = torch.optim.Adam(self.model.parameters(), args.gen_learning_rate, weight_decay=args.gen_L2, betas=(0.9, 0.999))

    def sample_trajectories(self, mbsize, all_visited):
        # Implement sampling logic here
        r = np.float32([i[1] for i in self.batch])
        a = np.random.randint(0, self.nactions, self.bufsize)
        steps = [self.envs[j].step(a[j], s=self.batch[j][2]) for j in range(self.bufsize)]
        rp = np.float32([i[1] for i in steps])
        A = rp / r
        U = np.random.uniform(0, 1, self.bufsize)
        for j in range(self.bufsize):
            if A[j] > U[j]:  # Accept
                self.batch[j] = (None, rp[j], steps[j][2])
                all_visited.append(tuple(steps[j][2]))
        return []

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        if isinstance(loss, float):  # Check if loss is a float
            loss = torch.tensor(loss, requires_grad=True)  # Convert to tensor if necessary
        loss = torch.tensor(loss, requires_grad=True)  # Convert to tensor if necessary
        print("Loss requires grad:", loss.requires_grad) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()
        return loss.item(), info

    def get_loss(self, batch):
        strs, thought_strs, r = zip(*batch["bulk_trajs"])
        info = {}
        s = self.tokenizer.process(strs).to(self.device)
        thought_s = self.tokenizer.process(thought_strs).to(self.device)

        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)

        inp_x = F.one_hot(s, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens) 
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()

        lens = [len(i) for i in strs]
        
        s = s.swapaxes(0, 1)
        thought_s = thought_s.swapaxes(0, 1)
        n = (s.shape[0] - 1) * s.shape[1]
        print("n:", n)
        r = r.clamp(min=3).log()
        # Model output logic here
        model_outs = self.model(x, None, return_all=True, lens=lens) 
        loss = 0  # Initialize loss
        accepted_samples = 0  # Count accepted samples

        for i in range(len(model_outs)):
        # Assuming out is a tuple of (policy_logits, value)
            policy_logits = model_outs[:, :, :self.num_tokens] 
            value = model_outs[:, :, self.num_tokens:-1] 
            policy_logits = self.logsoftmax2(policy_logits)[:-1] 
            value = self.logsoftmax2(value)[1:] 

            if value.size(2) == 0:
                continue

            # Reshape policy_logits and value for loss calculation
            policy_logits = policy_logits.reshape((n, self.num_tokens))
            policy_logits = policy_logits[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
            policy_logits = policy_logits.reshape(s[1:].shape)
            value = value.reshape((n, self.num_tokens))
            value = value[torch.arange(n, device=self.device), (thought_s[1:,].reshape((-1,))).clamp(0, self.num_tokens - 1)]
            value = value.reshape(s[1:].shape)

            # Calculate policy loss
            policy_loss = F.cross_entropy(policy_logits, value, reduction='mean')
            # Calculate value loss
            value_loss = F.mse_loss(value, r[:, i], reduction='mean')
            
            # Combine losses
            loss += policy_loss + value_loss
            print("policy_loss:", policy_loss)
            print("value_loss:", value_loss)
            
            # Metropolis-Hastings acceptance criteria
            acceptance_ratio = torch.exp(policy_loss.item() - value_loss.item())  # Simplified acceptance ratio
            if torch.rand(1).item() < acceptance_ratio.item():  # Accept the sample
                accepted_samples += 1
                loss += policy_loss + value_loss  # Only add to loss if accepted
                info['accepted'] = True  # Mark as accepted
            else:
                info['accepted'] = False  # Mark as rejected

            info['policy_loss'] = policy_loss.item()
            info['value_loss'] = value_loss.item()
            
        if accepted_samples > 0:
            loss /= accepted_samples  # Average loss over accepted samples
        else:
            loss = torch.tensor(0.0, device=self.device)  # Handle case with no accepted samples

        info['acceptance_prob'] = accepted_samples / len(model_outs)  # Calculate acceptance probability
        info['loss'] = loss

        return loss, info
    
    def reset(self):
        # Reset logic for the generator
        self.batch = [i.reset() for i in self.envs]  # Reset the environment states

    def forward(self, x, lens, return_all=False):
        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)

        out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef

        return out








