import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from lib.generator.base import GeneratorBase
from lib.model.mlp import MLP
import torch.nn.functional as F
import numpy as np




#MARS
class MARSGenerator:
    def __init__(self, args):
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim)
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

    def parameters(self):
        return self.net.parameters()

    def learn_from(self, trajs):
        states = torch.cat([traj['states'] for traj in trajs])
        actions = torch.cat([traj['actions'] for traj in trajs])
        rewards = torch.cat([traj['rewards'] for traj in trajs])
        
        logits = self.net(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        loss = -torch.mean(action_log_probs * rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def sample_many(self, env, n_samples):
        trajs = []
        for _ in range(n_samples):
            state = env.reset()
            done = False
            traj = {'states': [], 'actions': [], 'rewards': []}
            while not done:
                logits = self.net(torch.tensor(state, dtype=torch.float32))
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
class PPOGenerator:
    def __init__(self, args):
        self.args = args
        self.actor = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def learn_from(self, trajs):
        states = torch.cat([traj['states'] for traj in trajs])
        actions = torch.cat([traj['actions'] for traj in trajs])
        rewards = torch.cat([traj['rewards'] for traj in trajs])
        
        old_logits = self.actor(states)
        old_log_probs = torch.log_softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1).detach()
        
        for _ in range(self.args.ppo_epochs):
            logits = self.actor(states)
            log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            values = self.critic(states).squeeze(-1)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratio * rewards
            surrogate2 = torch.clamp(ratio, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon) * rewards
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            critic_loss = nn.MSELoss()(values, rewards)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    def sample_many(self, env, n_samples):
        trajs = []
        for _ in range(n_samples):
            state = env.reset()
            done = False
            traj = {'states': [], 'actions': [], 'rewards': []}
            while not done:
                logits = self.actor(torch.tensor(state, dtype=torch.float32))
                action = Categorical(logits=logits).sample().item()
                next_state, reward, done, _ = env.step(action)
                traj['states'].append(state)
                traj['actions'].append(action)
                traj['rewards'].append(reward)
                state = next_state
            trajs.append(traj)
        return trajs    
    



#SAC
# 


class SACGenerator:
    def __init__(self, args):
        self.args = args
        self.actor = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim)
        )
        self.critic1 = nn.Sequential(
            nn.Linear(args.state_dim + args.action_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(args.state_dim + args.action_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=args.lr)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters())

    def learn_from(self, trajs):
        states = torch.cat([traj['states'] for traj in trajs])
        actions = torch.cat([traj['actions'] for traj in trajs])
        rewards = torch.cat([traj['rewards'] for traj in trajs])
        next_states = torch.cat([traj['next_states'] for traj in trajs])
        
        with torch.no_grad():
            next_action_probs = torch.softmax(self.actor(next_states), dim=-1)
            next_log_probs = torch.log(next_action_probs + 1e-8)
            next_q1 = self.critic1(torch.cat([next_states, next_action_probs], dim=-1))
            next_q2 = self.critic2(torch.cat([next_states, next_action_probs], dim=-1))
            next_q = torch.min(next_q1, next_q2) - self.args.alpha * next_log_probs
            target_q = rewards + self.args.gamma * next_q
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        action_probs = torch.softmax(self.actor(states), dim=-1)
        log_probs = torch.log(action_probs + 1e-8)
        q1 = self.critic1(torch.cat([states, action_probs], dim=-1))
        q2 = self.critic2(torch.cat([states, action_probs], dim=-1))
        min_q = torch.min(q1, q2)
        actor_loss = (self.args.alpha * log_probs - min_q).mean()
        
        loss = critic_loss + actor_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def sample_many(self, env, n_samples):
        trajs = []
        for _ in range(n_samples):
            state = env.reset()
            done = False
            traj = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}
            while not done:
                logits = self.actor(torch.tensor(state, dtype=torch.float32))
                action = Categorical(logits=logits).sample().item()
                next_state, reward, done, _ = env.step(action)
                traj['states'].append(state)
                traj['actions'].append(action)
                traj['rewards'].append(reward)
                traj['next_states'].append(next_state)
                state = next_state
            trajs.append(traj)
        return trajs    
    



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
        return loss, info

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
            acceptance_ratio = torch.exp(-value_loss)  # Simplified acceptance ratio
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