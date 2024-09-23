import argparse
import yaml
import wandb
from tfb.run_tfbind import main  # Assuming main is the entry point for your training script

def run_experiment(args):
    # Initialize wandb for logging
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
    
    # Call the main training function
    main(args)
    
    # Finish the wandb run
    wandb.finish()

def main_loop(config_file, methods_to_run):
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
                wandb_project='YourProjectName',  # Set your project name
                wandb_entity='YourEntityName',     # Set your entity name
                wandb_run_name=f"{experiment['method']}_iter{experiment['gen_num_iterations']}"
            )
            
            # Run the experiment
            run_experiment(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments from a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--methods', type=str, nargs='+', help='List of methods to run (e.g., db mh random).', required=True)
    args = parser.parse_args()
    
    # Define methods_to_run based on command-line input
    methods_to_run = args.methods
    
    main_loop(args.config, methods_to_run)