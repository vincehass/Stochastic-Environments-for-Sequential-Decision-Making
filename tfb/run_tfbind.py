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

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.utils.env import get_tokenizer

import os, sys
import tempfile
import datetime
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_mlp.pkl.gz')
parser.add_argument("--tb_log_dir", default='results/test_mlp')
parser.add_argument("--name", default='test_mlp')
parser.add_argument("--load_scores_path", default='.')

# Multi-round
parser.add_argument("--num_rounds", default=1, type=int)
parser.add_argument("--task", default="tfbind", type=str)
parser.add_argument("--num_sampled_per_round", default=2048, type=int) 
parser.add_argument("--vocab_size", default=4)
parser.add_argument("--max_len", default=8)
parser.add_argument("--gen_max_len", default=8)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--enable_tensorboard", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="none", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=80, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument("--oracle_split", default="D2_target", type=str)
parser.add_argument("--proxy_data_split", default="D1", type=str)
parser.add_argument("--oracle_type", default="MLP", type=str)
parser.add_argument("--oracle_features", default="AlBert", type=str)
parser.add_argument("--medoid_oracle_dist", default="edit", type=str)
parser.add_argument("--medoid_oracle_norm", default=1, type=int)
parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)

# Generator
parser.add_argument("--gen_learning_rate", default=1e-5, type=float)
parser.add_argument("--gen_Z_learning_rate", default=5e-3, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
parser.add_argument("--gen_num_iterations", default=5000, type=int)  
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hidden", default=2048, type=int)  
parser.add_argument("--gen_num_layers", default=2, type=int) 
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=3, type=float)
parser.add_argument("--gen_reward_min", default=0, type=float)
parser.add_argument("--gen_L2", default=0, type=float)
parser.add_argument("--gen_partition_init", default=50, type=float)

# Soft-QLearning/GFlownet gen
parser.add_argument("--gen_reward_exp_ramping", default=3, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=10, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument("--gen_random_action_prob", default=0.001, type=float)
parser.add_argument("--gen_sampling_temperature", default=2., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
# PG gen
parser.add_argument("--gen_do_pg", default=0, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=1e-2, type=float)
# learning partition Z explicitly
parser.add_argument("--gen_do_explicit_Z", default=0, type=int)
parser.add_argument("--gen_model_type", default="mlp")

# Proxy
parser.add_argument("--proxy_learning_rate", default=1e-4)
parser.add_argument("--proxy_type", default="regression")
parser.add_argument("--proxy_arch", default="mlp")
parser.add_argument("--proxy_num_layers", default=2) 
parser.add_argument("--proxy_dropout", default=0.1)
parser.add_argument("--proxy_num_hid", default=2048, type=int) 
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
parser.add_argument("--proxy_early_stop_to_best_params", default=0, type=int)
parser.add_argument("--proxy_num_iterations", default=3000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)
parser.add_argument("--proxy_pos_ratio", default=0.9, type=float)
parser.add_argument("--proxy_betas", default="0.9,0.999") 

parser.add_argument("--dir", default='./results', type=str)
parser.add_argument("--method", default='db', type=str)
parser.add_argument("--stick", default=0.1, type=float)
parser.add_argument("--stochastic_alg", default=0, type=int)
parser.add_argument("--dynamics_lr", default=1e-5, type=float)
parser.add_argument("--dynamics_num_hid", default=2048, type=int)
parser.add_argument("--dynamics_num_layers", default=2, type=int)
parser.add_argument("--dynamics_clip", default=10, type=float)
parser.add_argument("--dynamics_L2", default=0, type=float)
parser.add_argument("--dynamics_off_pol", default=0, type=float)
parser.add_argument("--dynamics_off_pol_rounds", default=1, type=int)
parser.add_argument("--dynamics_sample_size", default=16, type=int)

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


def filter_len(x, y, max_len):
    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res


class RolloutWorker:
    def __init__(self, args, oracle, tokenizer):
        self.oracle = oracle
        self.max_len = args.max_len
        self.max_len = args.gen_max_len
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
        
        self.tokenizer = tokenizer
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = args.device
        self.args = args
        self.workers = MbStack(oracle)

        self.stick = args.stick

    def rollout(self, model, episodes, use_rand_policy=True):
        visited = []

        lists = lambda n: [list() for i in range(n)]

        states = [[] for i in range(episodes)]
        thought_states = [[] for i in range(episodes)]
        traj_states = [[[]] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)

        traj_logprob = np.zeros(episodes)

        for t in (range(self.max_len) if episodes > 0 else []):
            x = self.tokenizer.process(states).to(self.device)

            with torch.no_grad():
                logits = model(x, None)
            logits = logits[:, :self.args.vocab_size]
            
            try:
                cat = Categorical(logits=logits / self.sampling_temperature)
            except:
                print(states)
                print(x)
                print(logits)
                print(list(model.model.parameters()))
            
            actions = cat.sample()
            
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0, 1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(0, logits.shape[1])).to(self.device)

            noisy_actions = actions.clone().detach()
            rand_probs = np.random.rand(noisy_actions.shape[0])
            for i in range(noisy_actions.shape[0]):
                if rand_probs[i] < self.stick:
                    noisy_actions[i] = torch.tensor(np.random.randint(0, logits.shape[1])).to(self.device)

            i = 0
            for a, noisy_a in zip(actions, noisy_actions):
                if t == self.max_len - 1:
                    self.workers.push(states[i] + [noisy_a.item()], i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + [noisy_a.item()])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += [noisy_a.item()]
                thought_states[i] += [a.item()]
                i += 1
        
        return visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones

    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True):
        lists = lambda n: [list() for i in range(n)]
        
        visited, states, thought_states, traj_states, traj_actions, traj_rewards, traj_dones = self.rollout(model, self.episodes_per_step, use_rand_policy=use_rand_policy) 
        
        lens = np.mean([len(i) for i in traj_rewards])
        
        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            
            rq.append(r.item())

            s = states[mbidx]
            
            thought_s = thought_states[mbidx]
            visited.append((s, thought_s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, thought_s, traj_rewards[mbidx][-1].item()))
        
        if args.gen_data_sample_per_step > 0 and dataset is not None:
            n = args.gen_data_sample_per_step
            m = len(traj_states)
            
            x, y = dataset.sample_with_stochastic_data(n)
            
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            
            for idx in range(len(x)):
                curr_s, curr_thought_s = x[idx]
                curr_r = y[idx]

                curr_r = self.l2r(torch.tensor(curr_r), it)

                curr_data = (curr_s, curr_thought_s, curr_r)
                bulk_trajs.append(curr_data)

            for i in range(len(x)):
                traj_states[i + m].append([])
                for c, a in zip(x[i][0], self.tokenizer.process([x[i][1]]).reshape(-1)):
                    traj_states[i + m].append(traj_states[i + m][-1] + [c])
                    traj_actions[i + m].append(a)
                    traj_rewards[i + m].append(0 if len(traj_actions[i + m]) != self.max_len else self.l2r(torch.tensor(y[i]), it))
                    traj_dones[i + m].append(float(len(traj_rewards[i + m]) == self.max_len))
        
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


def train_generator(args, generator, oracle, tokenizer, dataset):
    print("Training generator")

    visited = []
    
    rollout_worker = RolloutWorker(args, oracle, tokenizer)
    
    for it in tqdm(range(args.gen_num_iterations + 1)):
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset)
        visited.extend(rollout_artifacts["visited"])

        if args.dynamics_off_pol:
            strs, thought_strs, r = zip(*rollout_artifacts["trajectories"]["bulk_trajs"])
            generator.dynamics_buffer.add(strs, thought_strs, r)

        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])
        print (it, loss, loss_info)        

        if it % 100 == 0:
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]).mean().item()
            args.logger.add_scalar("gen_reward", rs)
        if it % 5000 == 0:
            args.logger.save(args.save_path, args)
    
    return rollout_worker, None


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
    args.logger.add_scalar("top-{}-scores".format(k), np.mean(top100[1]), use_context=False)
    dist100 = mean_pairwise_distances(args, top100[0])
    args.logger.add_scalar("top-{}-dists".format(k), dist100, use_context=False)
    args.logger.add_object("top-{}-seqs".format(k), top100[0])
    print("Score", np.mean(top100[1]))
    print("Dist", dist100)
    
    infos = [np.mean(top100[1]), dist100]

    if collected:
        top100 = dataset.top_k_collected(k)
        print (top100[0])

        args.logger.add_scalar("top-{}-collected-scores".format(k), np.mean(top100[1]), use_context=False)
        args.logger.add_scalar("max-{}-collected-scores".format(k), np.max(top100[1]), use_context=False)
        
        dist100 = mean_pairwise_distances(args, top100[0])
        
        infos.extend([np.mean(top100[1]), dist100, np.max(top100[1]), np.percentile(top100[1], 50)])

        args.logger.add_scalar("top-{}-collected-dists".format(k), dist100, use_context=False)
        args.logger.add_object("top-{}-collected-seqs".format(k), top100[0])
        
        print("Collected Scores: mean={}, max={}, 50_percentile={}".format(np.mean(top100[1]), np.max(top100[1]), np.percentile(top100[1], 50)))
        print("Collected Dist", dist100)

    return infos


def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)

    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    proxy.update(dataset)
    
    generator = get_generator(args, tokenizer)

    rollout_worker, losses = train_generator(args, generator, proxy, tokenizer, dataset)

    batch = sample_batch(args, rollout_worker, generator, dataset, oracle)
    for sample_idx in range(len(batch[0])):
        curr_seq, curr_score = batch[0][sample_idx], batch[1][sample_idx]

    args.logger.add_object("collected_seqs", batch[0])
    args.logger.add_object("collected_seqs_scores", batch[1])

    dataset.add(batch)

    curr_round_infos = log_overall_metrics(args, dataset, collected=True)
    print (curr_round_infos)

    args.logger.save(args.save_path, args)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.logger = get_logger(args)
    args.device = torch.device('cpu')
    oracle = get_oracle(args)

    dataset = get_dataset(args, oracle)
    dataset.create_all_stochastic_datasets(args.stick)

    train(args, oracle, dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.method == 'db' and args.stick > 0.

    if args.stochastic_alg:
        args.gen_learning_rate = 1e-4

    main(args)