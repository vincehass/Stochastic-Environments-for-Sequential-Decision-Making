## Run experiments with different generators

### How to add more generators

1. First, let's update the base `Generator` class in `tfb/lib/generator/base_generator.py`:

```python:tfb/lib/generator/base_generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def forward(self, x):
        raise NotImplementedError

    def sample_batch(self, args, rollout_worker, dataset, oracle):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
```

2. Now, modify the `StochasticDBGFlowNetGenerator` in `tfb/lib/generator/gfn.py` to inherit from our base `Generator` class:

```python:tfb/lib/generator/gfn.py
from .base_generator import Generator

class StochasticDBGFlowNetGenerator(Generator):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        # Keep the existing initialization code
        # ...

    def forward(self, x):
        # Keep the existing forward method
        # ...

    def sample_batch(self, args, rollout_worker, dataset, oracle):
        # Keep the existing sample_batch method
        # ...

    def train_step(self, batch):
        # Implement the training step using the existing logic
        # This should include the loss calculation and optimization step
        # ...

    # Keep other existing methods
    # ...
```

3. Create similar classes for other generators, inheriting from the base `Generator` class:

```python:tfb/lib/generator/mars.py
from .base_generator import Generator

class MARSGenerator(Generator):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        # Initialize MARS-specific architecture

    def forward(self, x):
        # Implement forward pass

    def sample_batch(self, args, rollout_worker, dataset, oracle):
        # Implement batch sampling for MARS

    def train_step(self, batch):
        # Implement MARS training step
```

(Similarly, implement `MHGenerator`, `PPOGenerator`, `SACGenerator`, and `RandomTrajGenerator` in separate files)

4. Update the `get_generator` function in `tfb/lib/generator/__init__.py`:

```python:tfb/lib/generator/__init__.py
from .gfn import StochasticDBGFlowNetGenerator
from .mars import MARSGenerator
from .mh import MHGenerator
from .ppo import PPOGenerator
from .sac import SACGenerator
from .random_traj import RandomTrajGenerator

def get_generator(args, tokenizer):
    if args.method == 'db':
        return StochasticDBGFlowNetGenerator(args, tokenizer)
    elif args.method == 'mars':
        return MARSGenerator(args, tokenizer)
    elif args.method == 'mh':
        return MHGenerator(args, tokenizer)
    elif args.method == 'ppo':
        return PPOGenerator(args, tokenizer)
    elif args.method == 'sac':
        return SACGenerator(args, tokenizer)
    elif args.method == 'random':
        return RandomTrajGenerator(args, tokenizer)
    else:
        raise ValueError(f"Unknown generator method: {args.method}")
```

5. Modify the `train` function in `tfb/run_tfbind.py`:

```python:tfb/run_tfbind.py
from lib.generator import get_generator

def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)  # Implement this function to get the appropriate tokenizer
    generator = get_generator(args, tokenizer)

    for step in range(args.gen_num_iterations):
        batch = generator.sample_batch(args, rollout_worker, dataset, oracle)
        loss = generator.train_step(batch)

        # Log metrics
        curr_round_infos = log_overall_metrics(args, dataset, collected=True)

        wandb.log({
            "step": step,
            "loss": loss,
            "max_100_collected_scores": curr_round_infos['max-100-collected-scores'],
            "novelty": curr_round_infos['novelty'],
            "top_100_collected_dists": curr_round_infos['top-100-collected-dists'],
            "top_100_collected_scores": curr_round_infos['top-100-collected-scores'],
            "top_100_dists": curr_round_infos['top-100-dists'],
            "top_100_scores": curr_round_infos['top-100-scores'],
            "num_modes_collected": curr_round_infos['num-modes-collected'],
            "num_modes_all": curr_round_infos['num-modes-all']
        })

    # ... rest of the function ...
```

6. Update the `main` function to use the `method` argument:

```python:tfb/run_tfbind.py
def main(args):
    # ... existing setup code ...

    parser.add_argument('--method', type=str, default='db',
                        choices=['db', 'mars', 'mh', 'ppo', 'sac', 'random'],
                        help='Method to use for generation')

    # ... rest of the function ...
```

This implementation:

1. Maintains the existing `StochasticDBGFlowNetGenerator` structure.
2. Introduces a common base `Generator` class for all generators.
3. Keeps the `sample_batch` method, which is specific to this implementation.
4. Allows for easy addition of new generator types.
5. Uses the existing `method` argument to select the generator type.

To run experiments with different generators, you would use the `--method` argument:

```
python run_tfbind.py --method db
python run_tfbind.py --method mars
python run_tfbind.py --method ppo
```

This structure provides a more robust and consistent interface for all generators while maintaining compatibility with the existing codebase.
