import argparse
import yaml
import wandb
import os
import sys
import torch
import numpy as np
# Ensure the 'tfb' module is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from tfb.run_tfbind import train  # Import the train function
from tfb.lib.generator import StochasticKL2GFlowNetGenerator, StochasticDBGFlowNetGenerator, DeterminsticDBGFlowNetGenerator,MARSGenerator, MHGenerator, PPOGenerator, SACGenerator, RandomTrajGenerator
from tfb.lib.oracle_wrapper import get_oracle
from tfb.lib.dataset import get_dataset
from lib.oracle_wrapper import get_oracle
from lib.logging import get_logger
# Mapping of method names to generator classes
generator_map = {
    'stochastic_dbg': StochasticDBGFlowNetGenerator,
    'stochastic_klg2': StochasticKL2GFlowNetGenerator,
    'deterministic_dbg': DeterminsticDBGFlowNetGenerator,
    'mars': MARSGenerator,
    'mh': MHGenerator,
    'ppo': PPOGenerator,
    'sac': SACGenerator,
    'random_traj': RandomTrajGenerator
}

EXPERIMENT_NAME = "TfbindE"
WANDB_ENTITY = "nadhirvincenthassen" 
WANDB_PROJECT = "TFBINDStochSEQ"
def run_experiment(args, experiment_name):  # Added experiment_name parameter
    # Initialize wandb for logging
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=experiment_name)
    
    oracle = get_oracle(args)
    dataset = get_dataset(args, oracle)
    
    # Call the train function with the modified args
    train(args, oracle, dataset)
    


def main_loop(config_file, methods_to_run, args):  # Added args parameter
    # Load configurations from the YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    for experiment in config['experiments']:
        if experiment['method'] in methods_to_run:
            # Set up the arguments for each experiment
            args = argparse.Namespace(
                method=experiment['method'],
                gen_num_iterations=experiment['gen_num_iterations'],
                gen_episodes_per_step=experiment['gen_episodes_per_step'],
                gen_reward_exp=experiment.get('gen_reward_exp', 3),
                gen_reward_min=experiment.get('gen_reward_min', 0),
                gen_reward_norm=experiment.get('gen_reward_norm', 1),
                gen_random_action_prob=experiment.get('gen_random_action_prob', 0.001),
                gen_sampling_temperature=experiment.get('gen_sampling_temperature', 2.0),
                gen_leaf_coef=experiment.get('gen_leaf_coef', 25),
                gen_reward_exp_ramping=experiment.get('gen_reward_exp_ramping', 3),
                gen_balanced_loss=experiment.get('gen_balanced_loss', 1),
                gen_output_coef=experiment.get('gen_output_coef', 10),
                gen_loss_eps=experiment.get('gen_loss_eps', 1e-5),
                num_tokens=experiment.get('num_tokens', 4),
                gen_num_hidden=experiment.get('gen_num_hidden', 64),
                gen_num_layers=experiment.get('gen_num_layers', 2),
                gen_dropout=experiment.get('gen_dropout', 0.1),
                gen_partition_init=experiment.get('gen_partition_init', 150.0),
                gen_do_explicit_Z=experiment.get('gen_do_explicit_Z', True),
                gen_L2=experiment.get('gen_L2', 0.0),
                dynamics_num_hid=experiment.get('dynamics_num_hid', 128),
                dynamics_num_layers=experiment.get('dynamics_num_layers', 2),
                dynamics_dropout=experiment.get('dynamics_dropout', 0.1),
                dynamics_partition_init=experiment.get('dynamics_partition_init', 150.0),
                dynamics_do_explicit_Z=experiment.get('dynamics_do_explicit_Z', True),
                dynamics_L2=experiment.get('dynamics_L2', 0.0),
                dynamics_lr=experiment.get('dynamics_lr', 1e-3),
                dynamics_clip=experiment.get('dynamics_clip', 10.0),
                dynamics_off_pol=experiment.get('dynamics_off_pol', 0.0),
                gen_data_sample_per_step=experiment.get('gen_data_sample_per_step', 16),
                proxy_num_iterations=experiment.get('proxy_num_iterations', 3000),
                proxy_num_dropout_samples=experiment.get('proxy_num_dropout_samples', 25),
                proxy_num_hid=experiment.get('proxy_num_hid', 128),
                proxy_num_layers=experiment.get('proxy_num_layers', 2),
                proxy_dropout=experiment.get('proxy_dropout', 0.1),
                proxy_learning_rate=experiment.get('proxy_learning_rate', 1e-3),
                proxy_num_per_minibatch=experiment.get('proxy_num_per_minibatch', 32),
                stick=experiment.get('stick', 0.25),
                wandb_project='TFBINDStochSEQ',  # Set your project name
                wandb_entity='nadhirvincenthassen',    
                wandb_run_name=f"{experiment['method']}_iter{experiment['gen_num_iterations']}_task{args.task}"
            )
            
            # Get the generator class based on the method
            generator_class = generator_map.get(experiment['method'])
            if generator_class is None:
                raise ValueError(f"Unknown method: {experiment['method']}")

            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--method', nargs='+', required=True, help='Methods to run')
    parser.add_argument("--task", default="tfbind", type=str)
    # WandB arguments
    parser.add_argument("--wandb_project", default=WANDB_PROJECT, help="WandB project name")
    parser.add_argument("--wandb_run_name", default=None, help="WandB run name")
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY, help="WandB entity (username or team name)")
    # Arguments
    parser.add_argument("--save_path", default=f'results/{EXPERIMENT_NAME}.pkl.gz')
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--load_scores_path", default='.')
    parser.add_argument("--num_rounds", default=1, type=int)

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
    parser.add_argument('--gen_data_sample_per_step', type=int, default=16, help='Number of data samples to generate per step')
    parser.add_argument('--gen_clip', type=float, default=10.0, help='Gradient clipping value for generator')
    parser.add_argument('--gamma', type=float, default=0.5, help='Discount factor for the reward')
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
    args = parser.parse_args()
    print(args)
    methods_to_run = args.method
    args.logger = get_logger(args)
    args.device = torch.device('cpu')
    
    
    
    # Call run_experiment before main_loop
    for method in methods_to_run:
        run_experiment(args, f"{method}_experiment")  # Pass the experiment name

    main_loop(args.config, methods_to_run, args)  # Pass args to main_loop

    

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
    



if __name__ == "__main__":
    main()
    wandb.finish()
    