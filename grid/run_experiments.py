import subprocess
import itertools
import numpy as np

# Define the ranges for action and state dimensions
action_dims = [4]#[2, 3, 4]
state_dims = [2, 3, 4]

# Define the range for stick values
stick_values = np.arange(0.25, 0.95, 0.05).round(2)

# Define the number of seeds for each configuration
num_seeds = 1

# Iterate over all combinations of action and state dimensions, and stick values
for action_dim, state_dim, stick in itertools.product(action_dims, state_dims, stick_values):
    for seed in range(num_seeds):
        # Construct the command to run the main script
        cmd = [
            "python", "grid/main.py",
            "--action_dim", str(action_dim),
            "--state_dim", str(state_dim),
            "--stick", str(stick),
            "--seed", str(seed),
            "--progress"
        ]
        
        # Print the command being run
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # Run the command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment: {e}")
            continue  # Continue to the next iteration

print("All experiments completed!")