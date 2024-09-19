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

EXPERIMENT_NAME = "tfbind8_stochasticEnv"
WANDB_ENTITY = "nadhirvincenthassen"  # Your username set as default

parser = argparse.ArgumentParser()

# Arguments
parser.add_argument("--save_path", default=f'results/{EXPERIMENT_NAME}.pkl.gz')
parser.add_argument("--name", default=EXPERIMENT_NAME)
parser.add_argument("--load_scores_path", default='.')
parser.add_argument("--num_rounds", default=1, type=int)
parser.add_argument("--task", default="tfbind", type=str)
parser.add_argument("--num_sampled_per_round", default=2048, type=int) 
parser.add_argument('--vocab_size', type=int, default=4)
parser.add_argument('--max_len', type=int, default=8)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="none", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=80, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument('--stick', type=float, default=0.25, help='Stick parameter for StochasticDBGFlowNetGenerator')
# Generator arguments
parser.add_argument("--gen_learning_rate", default=1e-5, type=float)
parser.add_argument("--gen_num_iterations", default=5000, type=int)
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_reward_exp", default=3, type=float)
parser.add_argument("--gen_reward_min", default=0, type=float)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_random_action_prob", default=0.001, type=float)
parser.add_argument("--gen_sampling_temperature", default=2., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
parser.add_argument("--gen_reward_exp_ramping", default=3, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=10, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument('--method', type=str, default='db', help='Method to use for generator (e.g., tb, db)')
parser.add_argument('--num_tokens', type=int, default=4, help='Number of tokens in the vocabulary')
parser.add_argument('--gen_num_hidden', type=int, default=64, help='Number of hidden units in the generator')
parser.add_argument('--gen_num_layers', type=int, default=2, help='Number of layers in the generator')
parser.add_argument('--gen_dropout', type=float, default=0.1, help='Dropout rate for the generator')
parser.add_argument('--gen_partition_init', type=float, default=150.0, help='Partition initialization value for the generator')
parser.add_argument('--gen_do_explicit_Z', action='store_true', help='Enable explicit Z for the generator')
parser.add_argument('--gen_L2', type=float, default=0.0, help='L2 regularization coefficient for generator')
parser.add_argument('--dynamics_num_hid', type=int, default=128, help='Number of hidden units in the dynamics network')
parser.add_argument('--dynamics_num_layers', type=int, default=2, help='Number of layers in the dynamics network')
parser.add_argument('--dynamics_dropout', type=float, default=0.1, help='Dropout rate for the dynamics network')
parser.add_argument('--dynamics_partition_init', type=float, default=150.0, help='Partition initialization value for the dynamics network')
parser.add_argument('--dynamics_do_explicit_Z', action='store_true', help='Enable explicit Z for the dynamics network')
parser.add_argument('--dynamics_L2', type=float, default=0.0, help='L2 regularization coefficient for dynamics network')
parser.add_argument('--dynamics_lr', type=float, default=1e-3, help='Learning rate for the dynamics network')
parser.add_argument('--dynamics_clip', type=float, default=10.0, help='Clipping value for the dynamics network')
parser.add_argument('--dynamics_off_pol', type=float, default=0.0, help='Off-policy dynamics parameter')
# Proxy arguments
parser.add_argument("--proxy_type", default="regression")
parser.add_argument("--proxy_num_iterations", default=3000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)
parser.add_argument('--proxy_num_hid', type=int, default=128, help='Number of hidden units in the proxy model')
parser.add_argument('--proxy_num_layers', type=int, default=2, help='Number of layers in the proxy model')
parser.add_argument('--proxy_dropout', type=float, default=0.1, help='Dropout rate for the proxy model')
parser.add_argument('--proxy_learning_rate', type=float, default=1e-3, help='Learning rate for the proxy model')
parser.add_argument('--proxy_num_per_minibatch', type=int, default=32,
                    help='Number of samples per minibatch for proxy training')

# WandB arguments
parser.add_argument("--wandb_project", default=EXPERIMENT_NAME, help="WandB project name")
parser.add_argument("--wandb_run_name", default=None, help="WandB run name")
parser.add_argument("--wandb_entity", default=WANDB_ENTITY, help="WandB entity (username or team name)")

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

        print(f"Debug: Initial states = {states}")
        print(f"Debug: Number of episodes = {num_episodes}")
        print(f"Debug: Max length = {self.max_len}")

        for t in range(self.max_len):
            print(f"Debug: Timestep {t}")
            print(f"Debug: States before processing = {states}")
            print(f"Debug: Length of states = {len(states)}")
            if len(states) > 0:
                print(f"Debug: Length of first state = {len(states[0])}")
            
            try:
                x = self.environment.tokenizer.process(states).to(self.device)
            except IndexError as e:
                print(f"Debug: IndexError occurred. States: {states}")
                raise e

            with torch.no_grad():
                logits = model(x, None)
            logits = logits[:, :self.args.vocab_size]
            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()
            
            if use_rand_policy and self.random_action_prob > 0:
                rand_mask = torch.rand(actions.shape[0]) < self.random_action_prob
                actions[rand_mask] = torch.randint(0, logits.shape[1], (rand_mask.sum(),)).to(self.device)

            for i, a in enumerate(actions):
                print(f"Debug: Processing episode {i}, action {a.item()}")
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
            
            print(f"Debug: States after processing = {states}")
        
        # After the rollout is complete, evaluate with the oracle
        final_states = [s for s in states if len(s) == self.max_len]
        if final_states:
            oracle_scores = self.oracle(final_states)
            for i, (state, score) in enumerate(zip(final_states, oracle_scores)):
                visited.append((state, thought_states[i], score.item(), score.item()))

        return visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones

    def execute_train_episode_batch(self, generator, it, dataset, use_rand_policy=False):
        print(f"Debug: generator in execute_train_episode_batch: {generator}")
        visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones = self.rollout(generator, self.episodes_per_step, use_rand_policy=use_rand_policy)
        print(f"Debug: Rollout completed. States: {states}")
        
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
    reference_set = set(tuple(seq) for seq in reference_sequences)
    novel_count = sum(1 for seq in new_sequences if tuple(seq) not in reference_set)
    return novel_count / len(new_sequences)

def train_generator(args, generator, oracle, proxy, tokenizer, dataset):
    # Add this debug print
    print("Debug: generator before rollout:", generator)
    
    print(f"Debug: generator type: {type(generator)}")
    print("Training generator")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, proxy, tokenizer, dataset)
    print(f"Debug: rollout_worker created: {rollout_worker}")
    unique_sequences = set()
    total_steps = 0
    
    for it in tqdm(range(args.gen_num_iterations + 1)):
        # Add this debug print
        print(f"Debug: generator at iteration {it}:", generator)
        
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset, use_rand_policy=False)
        visited.extend(rollout_artifacts["visited"])

        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])
        print(it, loss, loss_info)

        # Calculate metrics
        rewards = [i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]
        avg_reward = np.mean(rewards)
        diversity = calculate_diversity(rollout_artifacts["trajectories"]["states"])
        novelty = calculate_novelty(rollout_artifacts["trajectories"]["states"], dataset.get_all_sequences())
        unique_sequences.update(tuple(seq) for seq in rollout_artifacts["trajectories"]["states"])
        total_steps += sum(len(traj) for traj in rollout_artifacts["trajectories"]["traj_states"])

        # Log metrics to wandb
        wandb.log({
            "iteration": it,
            "loss": loss,
            "dynamic_loss": loss_info.get("dynamic_loss", 0),
            "avg_reward": avg_reward,
            "diversity": diversity,
            "novelty": novelty,
            "unique_sequences": len(unique_sequences),
            "total_steps": total_steps,
        })

        if it % 5000 == 0:
            args.logger.save(args.save_path, args)
    
    return rollout_worker, generator

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
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, use_rand_policy=False)
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

def log_overall_metrics(args, dataset, collected=False, k=100):
    top100 = dataset.top_k(k)
    wandb.log({"top-{}-scores".format(k): np.mean(top100[1])})
    dist100 = mean_pairwise_distances(args, top100[0])
    wandb.log({"top-{}-dists".format(k): dist100})
    print("Score", np.mean(top100[1]))
    print("Dist", dist100)
    
    infos = [np.mean(top100[1]), dist100]

    if collected:
        top100_collected = dataset.top_k_collected(k)
        wandb.log({
            "top-{}-collected-scores".format(k): np.mean(top100_collected[1]),
            "max-{}-collected-scores".format(k): np.max(top100_collected[1])
        })
        dist100_collected = mean_pairwise_distances(args, top100_collected[0])
        infos.extend([np.mean(top100_collected[1]), dist100_collected, np.max(top100_collected[1]), np.percentile(top100_collected[1], 50)])
        wandb.log({"top-{}-collected-dists".format(k): dist100_collected})
        print("Collected Scores: mean={}, max={}, 50_percentile={}".format(np.mean(top100_collected[1]), np.max(top100_collected[1]), np.percentile(top100_collected[1], 50)))
        print("Collected Dist", dist100_collected)

    return infos

def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)
    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    proxy.update(dataset)
    
    generator = get_generator(args, tokenizer)
    
    
    if generator is None:
        raise ValueError("Failed to initialize generator")
    
    rollout_worker, _ = train_generator(args, generator, oracle, proxy, tokenizer, dataset)
    batch = sample_batch(args, rollout_worker, generator, dataset, oracle)
    args.logger.add_object("collected_seqs", batch[0])
    args.logger.add_object("collected_seqs_scores", batch[1])
    dataset.add(batch)
    curr_round_infos = log_overall_metrics(args, dataset, collected=True)
    print(curr_round_infos)
    args.logger.save(args.save_path, args)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use the experiment name in the wandb run name if not provided
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{EXPERIMENT_NAME}_{args.seed}"

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )

    args.logger = get_logger(args)
    args.device = torch.device('cpu')
    oracle = get_oracle(args)
    dataset = get_dataset(args, oracle)
    train(args, oracle, dataset)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)