import argparse
import gzip
import pickle
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.utils.env import get_tokenizer, StochasticGFNEnvironment

import os, sys
import tempfile
import datetime
import shutil

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix


class MbStack:
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack])
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)

class RolloutWorker:
    def __init__(self, args, oracle, proxy, tokenizer, dataset):
        # Initialize with all the arguments
        self.args = args
        self.oracle = oracle
        self.proxy = proxy
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = args.max_len
        self.episodes_per_step = args.gen_episodes_per_step
        self.random_action_prob = args.gen_random_action_prob
        self.reward_exp = args.gen_reward_exp
        self.sampling_temperature = args.gen_sampling_temperature
        self.out_coef = args.gen_output_coef
        self.balanced_loss = args.gen_balanced_loss == 1
        self.reward_norm = args.gen_reward_norm
        self.reward_min = torch.tensor(float(args.gen_reward_min))
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.leaf_coef = args.gen_leaf_coef
        self.exp_ramping_factor = args.gen_reward_exp_ramping
        
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = args.device
        self.workers = MbStack(oracle)

        # Create the stochastic environment
        self.environment = StochasticGFNEnvironment(
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            oracle=self.oracle,
            stick=args.stick
        )

    def rollout(self, model, num_episodes, use_rand_policy=False):
        if model is None:
            raise ValueError("Model passed to rollout is None")
        model.eval()
        visited = []
        states = [self.environment.reset() for _ in range(num_episodes)]
        thought_states = [[] for _ in range(num_episodes)]
        traj_states = [[[]] for _ in range(num_episodes)]
        traj_actions = [[] for _ in range(num_episodes)]
        traj_rewards = [[] for _ in range(num_episodes)]
        traj_dones = [[] for _ in range(num_episodes)]

        for t in range(self.max_len):
            if len(states) > 0:
                
                print(f"Debug: Length of first state = {len(states[0])}")
            
            try:
                x = self.environment.tokenizer.process(states)
                if x.numel() > 0:  # Check if tensor is not empty
                    x = x.to(self.device)
                else:
                    print("Debug: Processed states resulted in an empty tensor")
                    
            except Exception as e:
                print(f"Debug: Exception occurred: {e}")
                print(f"Debug: States when exception occurred: {states}")
                raise e

          
            with torch.no_grad():
                logits = model(x, None)
            logits = logits[:, :self.args.vocab_size]

            
            if torch.isnan(logits).any():
                print("NaN values found in logits!")
                # Handle the NaN case, e.g., set logits to zero or some default value
                logits = torch.nan_to_num(logits)  # This will replace NaNs with 0

            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()
            
            #print("Actions:", actions)
            
            if use_rand_policy and self.random_action_prob > 0:
                rand_mask = torch.rand(actions.shape[0]) < self.random_action_prob
                actions[rand_mask] = torch.randint(0, logits.shape[1], (rand_mask.sum(),)).to(self.device)

            for i, a in enumerate(actions):
                if t == self.max_len - 1:
                    self.workers.push(states[i] + [a.item()], i)
                    r, d = 0, 1
                else:
                    r, d = 0, 0
                traj_states[i].append(states[i] + [a.item()])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i].append(a.item())
                thought_states[i].append(a.item())
        
        
        # After the rollout is complete, evaluate with the oracle
        # print("Debug: States after rollout:")   
        # for i, state in enumerate(states):
        #     print(f"  State {i}: {state}")
        #     #print(f"  Actions {i}: {actions}")
        # print("Debug: Thought states after rollout:")
        # for i, thought_state in enumerate(thought_states):
        #     print(f"  Thought state {i}: {thought_state}")
        # print("Debug: Traj states after rollout:")
        # for i, traj_state in enumerate(traj_states):
        #     print(f"  Traj state {i}: {traj_state}")
        # print(f"Debug: Traj actions after rollout: {traj_actions}")
        # print(f"Debug: Traj rewards after rollout: {traj_rewards}")
        # print(f"Debug: Traj dones after rollout: {traj_dones}")

        final_states = [s for s in states if len(s) == self.max_len]
        if final_states:
            oracle_scores = self.oracle(final_states)
            for i, (state, score) in enumerate(zip(final_states, oracle_scores)):
                visited.append((state, thought_states[i], score.item(), score.item()))

        return visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones

    def execute_train_episode_batch(self, generator, it, dataset, use_rand_policy=False):
        visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones = self.rollout(generator, self.episodes_per_step, use_rand_policy=use_rand_policy)
        
        bulk_trajs = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            s = states[mbidx]
            thought_s = thought_states[mbidx]
            visited.append((s, thought_s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, thought_s, traj_rewards[mbidx][-1].item()))
        
        if self.args.gen_data_sample_per_step > 0 and dataset is not None:
            x, y = dataset.sample(self.args.gen_data_sample_per_step)
            for seq, score in zip(x, y):
                curr_r = self.l2r(torch.tensor(score), it)
                bulk_trajs.append((seq, seq, curr_r))
        
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }

def calculate_diversity(sequences):
    unique_sequences = set(tuple(seq) for seq in sequences)
    return len(unique_sequences) / len(sequences)

def calculate_novelty(new_sequences, reference_sequences):
    # Ensure new_sequences is a 2D array
    new_sequences = np.array(new_sequences)
    if new_sequences.ndim == 1:
        new_sequences = new_sequences.reshape(-1, 1)  # Reshape to 2D if it's 1D

    # Convert sequences to a numerical format (e.g., integer encoding)
    unique_chars = np.unique(np.concatenate(new_sequences))  # Get unique characters
    char_to_int = {char: idx for idx, char in enumerate(unique_chars)}  # Create a mapping
    int_sequences = np.array([[char_to_int[char] for char in seq] for seq in new_sequences])  # Convert to integers

    reference_set = set(tuple(seq) for seq in reference_sequences)
    novel_count = sum(1 for seq in new_sequences if tuple(seq) not in reference_set)
    
    # Calculate diversity based on pairwise distances
    if len(int_sequences) > 1:
        distances = squareform(pdist(int_sequences, metric='hamming'))
        diversity_score = np.mean(distances)  # Average distance as a diversity measure
    else:
        diversity_score = 0

    # Check if new_sequences is empty using .size
    novelty_score = novel_count / int_sequences.shape[0] if int_sequences.size > 0 else 0
    return novelty_score + diversity_score  # Adjust the return value as needed

def count_elements(iterable):
    return sum(1 for _ in iterable)

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def train_generator(args, generator, oracle, proxy, tokenizer, dataset):
    print("Training generator")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, proxy, tokenizer, dataset)
    unique_sequences = set()
    total_steps = 0
    total_states_visited = set()  # To track unique states visited

    # Initialize lists for confusion matrix
    all_expected_actions = []
    all_expected_states = []
    all_r_gamma_values = []

    for it in tqdm(range(args.gen_num_iterations + 1)):
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset, use_rand_policy=False)
        visited.extend(rollout_artifacts["visited"])

        # Check if all_visited is iterable
        all_visited = rollout_artifacts["trajectories"]["states"]
        if not is_iterable(all_visited):
            raise ValueError("Expected 'all_visited' to be iterable, but got: {}".format(type(all_visited)))
        
        # Count elements in all_visited
        all_visited_count = count_elements(all_visited)
        #print("Number of elements in all_visited:", all_visited_count)
        
        total_states_visited.update(tuple(seq) for seq in all_visited)

        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])

        # if args.method == "SFGN-KL":
            # Collect actions taken during the training step
        actions_taken = rollout_artifacts["trajectories"]["traj_actions"]  # Assuming this contains the actions taken

        # Collect r_gamma values
        
         
        
        # Process r_gamma
        if 'r_gamma' in loss_info:
            r_gamma = loss_info['r_gamma']
        
            if isinstance(r_gamma, torch.Tensor):
                print("Shape of r_gamma:", r_gamma.shape)
                print("r_gamma:", r_gamma)
                
                if r_gamma.dim() == 1:
                    expected_actions = torch.argmax(r_gamma, dim=0)  # Use default dim=0 for 1D tensor
                elif r_gamma.dim() == 2:
                    if r_gamma.size(1) == 1:
                        expected_actions = torch.argmax(r_gamma.squeeze(1), dim=0)  # Squeeze the second dimension
                    else:
                        expected_actions = torch.argmax(r_gamma, dim=1)  # Handle other 2D cases
                else:
                    raise ValueError("Unexpected tensor dimensions for r_gamma")
                all_r_gamma_values.append(r_gamma.squeeze().detach().cpu().numpy())  # Collect r_gamma values
            else:
                print(f"Warning: r_gamma is not a tensor, but a {type(r_gamma)}. Skipping confusion matrix update.")
                expected_actions = torch.tensor([])  # Set to an empty tensor if not a tensor
                
        else:
            print("r_gamma not found in loss_info. Skipping r_gamma update.")
            
      
        
        

        # Calculate metrics
        rewards = [i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]
        avg_reward = np.mean(rewards)
        diversity = calculate_diversity(rollout_artifacts["trajectories"]["states"])
        novelty = calculate_novelty(rollout_artifacts["trajectories"]["states"], dataset.get_all_sequences())
        unique_sequences.update(tuple(seq) for seq in rollout_artifacts["trajectories"]["states"])
        total_steps += sum(len(traj) for traj in rollout_artifacts["trajectories"]["traj_states"])
       
        last_states = rollout_artifacts["trajectories"]["traj_states"][-1]
        last_actions = rollout_artifacts["trajectories"]["traj_actions"][-1]
        last_rewards = rollout_artifacts["trajectories"]["traj_rewards"][-1]

        
        
       

        # Log metrics to wandb
        wandb_log_dict = {
            "iteration": it,
            "loss": loss,
            "avg_reward": avg_reward,
            "diversity": diversity,
            "novelty": novelty,
            "unique_sequences": len(unique_sequences),
            "total_steps": total_steps,
            "total_states_visited": count_elements(total_states_visited),  # Use count_elements instead of len

        }

        
        
        # Handle different types of loss_info
        if isinstance(loss_info, dict):
            wandb_log_dict.update(loss_info)  # Update with all loss_info metrics
            # Log kl_divergence_loss specifically
            if 'kl_divergence_loss' in loss_info:
                wandb_log_dict["kl_divergence_loss"] = loss_info['kl_divergence_loss']  # Log KL divergence loss
                if it % 100 == 0:
                    print("kl_divergence_loss: ", loss_info['kl_divergence_loss'])
        elif isinstance(loss_info, (int, float)):
            wandb_log_dict["dynamic_loss"] = loss_info
        else:
            print(f"Warning: Unexpected type for loss_info: {type(loss_info)}")
        
        # if args.method == "SGFN-KL":
            # Log r_gamma if available in loss_info
        if isinstance(loss_info, dict) and 'r_gamma' in loss_info:
            wandb_log_dict["r_gamma"] = loss_info['r_gamma'].sum().item() # Log r_gamma
            if it % 100 == 0:
                print("r_gamma: ", loss_info['r_gamma'])
        
            if 'H_high' in loss_info:
                wandb_log_dict["H_high"] = loss_info['H_high'].sum().item() # Log H_high
                if it % 100 == 0:
                    print("H_high: ", loss_info['H_high'])
            if 'H_low' in loss_info:
                wandb_log_dict["H_low"] = loss_info['H_low'].sum().item() # Log H_low
                if it % 100 == 0:
                    print("H_low: ", loss_info['H_low'])
        else:
            print(f"Warning: Unexpected type for loss_info: {type(loss_info)}")
                
        wandb.log(wandb_log_dict)
       
        if it % 100 == 0:
            # Debugging information
            #args.logger.save(args.save_path, args)
            print(f"Iteration {it}:")
            print("Total step: ", total_steps)
            print("loss: ", loss)
            print("Shape of unique_sequence: ", len(unique_sequences))
            print("Shape of states_visited: ", count_elements(total_states_visited))
            print("Value of diversity: ", (diversity))
            print("Value of novelty: ", (novelty))


    return rollout_worker, generator

def calculate_dists(sequences):
    return sum(len(set(a) ^ set(b)) for a in sequences for b in sequences) / (len(sequences) * (len(sequences) - 1))

def filter_samples(args, samples, reference_set):
    filtered_samples = []
    for sample in samples:
        similar = False
        for example in reference_set:
            if is_similar(sample, example, args.filter_distance_type, args.filter_threshold):
                similar = True
                break
        if not similar:
            filtered_samples.append(sample)
    return filtered_samples

def sample_batch(args, rollout_worker, generator, current_dataset, oracle):
    print("Generating samples")
    samples = ([], [])
    scores = []
    
    while len(samples[0]) < args.num_sampled_per_round:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, dataset=current_dataset, use_rand_policy=False)
        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals)
        scores.extend(torch.tensor(rollout_artifacts["trajectories"]["traj_rewards"])[:, -1].numpy().tolist())
    
    idx_pick = np.argsort(scores)[::-1][:args.num_sampled_per_round]
    return (np.array(samples[0])[idx_pick].tolist(), np.array(samples[1])[idx_pick].tolist())

def construct_proxy(args, tokenizer, dataset=None):
    proxy = get_proxy_model(args, tokenizer)
    sigmoid = nn.Sigmoid()
    if args.proxy_type == "classification":
        l2r = lambda x: sigmoid(x.clamp(min=args.gen_reward_min)) / args.gen_reward_norm
    elif args.proxy_type == "regression":
        l2r = lambda x: x.clamp(min=args.gen_reward_min) / args.gen_reward_norm
    args.reward_exp_min = max(l2r(torch.tensor(args.gen_reward_min)), 1e-32)
    acq_fn = get_acq_fn(args)
    return acq_fn(args, proxy, l2r, dataset)

def mean_pairwise_distances(args, seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)

def log_overall_metrics(args, dataset, collected=False):
    k = 100
    top100 = dataset.top_k(k)
    top100_collected = dataset.top_k_collected(k) if collected else top100

    # Ensure dataset.train and dataset.valid are 2D
    train = dataset.train if dataset.train.ndim == 2 else dataset.train.reshape(-1, 1)
    valid = dataset.valid if dataset.valid.ndim == 2 else dataset.valid.reshape(-1, 1)

    # Ensure train and valid have the same number of columns
    max_cols = max(train.shape[1], valid.shape[1])
    if train.shape[1] < max_cols:
        train = np.pad(train, ((0, 0), (0, max_cols - train.shape[1])), mode='constant')
    if valid.shape[1] < max_cols:
        valid = np.pad(valid, ((0, 0), (0, max_cols - valid.shape[1])), mode='constant')

    # Concatenate the data
    all_sequences = np.concatenate((train, valid))

    # Calculate metrics
    max_100_collected_scores = np.max(top100_collected[1])
    novelty = calculate_novelty(top100_collected[0], all_sequences)
    top_100_collected_dists = calculate_dists(top100_collected[0])
    top_100_collected_scores = np.mean(top100_collected[1])
    top_100_dists = calculate_dists(top100[0])
    top_100_scores = np.mean(top100[1])
    
    # Calculate number of modes
    num_modes_collected = calculate_num_modes(top100_collected[0])
    num_modes_all = calculate_num_modes(top100[0])
    
    print("num_modes_collected: ", num_modes_collected)
    print("num_modes_all: ", num_modes_all)

    return {
        'max-100-collected-scores': max_100_collected_scores,
        'novelty': novelty,
        'top-100-collected-dists': top_100_collected_dists,
        'top-100-collected-scores': top_100_collected_scores,
        'top-100-dists': top_100_dists,
        'top-100-scores': top_100_scores,
        'num-modes-collected': num_modes_collected,
        'num-modes-all': num_modes_all
    }

def calculate_num_modes(sequences, distance_threshold=1):
    # Ensure sequences is a 2D numpy array
    sequences = np.array(sequences)
    if sequences.ndim == 1:
        sequences = sequences.reshape(-1, 1)
    
    # If sequences are strings, convert to integer representation
    if sequences.dtype.kind in ['U', 'S']:  # Unicode or byte string
        unique_chars = np.unique(sequences.ravel())
        char_to_int = {char: i for i, char in enumerate(unique_chars)}
        int_sequences = np.array([[char_to_int[char] for char in seq] for seq in sequences])
    else:
        int_sequences = sequences
    
    # Ensure int_sequences is 2D
    if int_sequences.ndim == 1:
        int_sequences = int_sequences.reshape(-1, 1)
    
    # Calculate pairwise Hamming distances
    distances = squareform(pdist(int_sequences, metric='hamming'))
    print("Pairwise distances:\n", distances)
    
    # Initialize modes
    modes = []
    for i in range(len(sequences)):
        # Check if the current sequence is a mode
        is_mode = True
        for j in modes:
            # Adjust the condition to allow for more flexibility
            if distances[i, j] < distance_threshold / len(sequences[0]):
                is_mode = False
                break
        if is_mode:
            modes.append(i)
    
    # Return the unique modes based on their indices
    unique_modes = np.unique(modes)
    return len(unique_modes)

def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)
    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    proxy.update(dataset)
    
    generator = get_generator(args, tokenizer)
    
    if generator is None:
        raise ValueError("Failed to initialize generator")
    
    rollout_worker, _ = train_generator(args, generator, oracle, proxy, tokenizer, dataset)

    # Wrap the training loop with tqdm
    with tqdm(total=args.gen_num_iterations, desc="Training Progress", unit="iteration") as pbar:
        for step in range(args.gen_num_iterations):
            batch = sample_batch(args, rollout_worker, generator, dataset, oracle)
        
            # Debug prints
            # print(f"Debug: dataset.train shape: {dataset.train.shape}")
            # print(f"Debug: dataset.train_scores shape: {dataset.train_scores.shape}")
            # print(f"Debug: batch[0] shape: {np.array(batch[0]).shape}")
            # print(f"Debug: batch[1] shape: {np.array(batch[1]).shape}")

            dataset.add(batch)
            curr_round_infos = log_overall_metrics(args, dataset, collected=True)
            
            wandb.log({
                "step": step,
                "max_100_collected_scores": curr_round_infos['max-100-collected-scores'],
                "novelty": curr_round_infos['novelty'],
                "top_100_collected_dists": curr_round_infos['top-100-collected-dists'],
                "top_100_collected_scores": curr_round_infos['top-100-collected-scores'],
                "top_100_dists": curr_round_infos['top-100-dists'],
                "top_100_scores": curr_round_infos['top-100-scores'],
                "num_modes_collected": curr_round_infos['num-modes-collected'],
                "num_modes_all": curr_round_infos['num-modes-all']
            })

            pbar.update(1)  # Update the progress bar
        
    args.logger.save(args.save_path, args)

