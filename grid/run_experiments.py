import subprocess
import itertools

# Define the ranges for action and state dimensions
action_dims = [2, 3, 4]
state_dims = [2, 3, 4]

# Define the number of seeds for each configuration
num_seeds = 1

# Iterate over all combinations of action and state dimensions
for action_dim, state_dim in itertools.product(action_dims, state_dims):
    for seed in range(num_seeds):
        # Construct the command to run the main script
        cmd = [
            "python", "grid/main.py",
            "--action_dim", str(action_dim),
            "--state_dim", str(state_dim),
            "--seed", str(seed),
            "--progress"
        ]
        
        # Print the command being run
        print(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            continue  # Continue to the next iteration

print("All experiments completed!")