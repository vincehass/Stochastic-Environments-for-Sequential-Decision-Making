from lib.generator.gfn import StochasticDBGFlowNetGenerator, StochasticKLGFlowNetGenerator, StochasticKL3GFlowNetGenerator, StochasticKL2GFlowNetGenerator
from lib.generator.generators_friends import DeterminsticDBGFlowNetGenerator, MARSGenerator, MHGenerator, PPOGenerator, SACGenerator, RandomTrajGenerator



def get_generator(args, tokenizer):
    method = args.method[0] if isinstance(args.method, list) else args.method  # Handle list input
    print(method)
    if method == 'GFN-DB':
        return DeterminsticDBGFlowNetGenerator(args, tokenizer)
    elif method == 'SGN-DB':
        return StochasticDBGFlowNetGenerator(args, tokenizer)
    elif method == 'SGFN-KL-gamma':
        return StochasticKL3GFlowNetGenerator(args, tokenizer)
    # elif method == 'stochastic_klg':
    #     return StochasticKLGFlowNetGenerator(args, tokenizer)
    elif method == 'SGFN-KL':
        return StochasticKL2GFlowNetGenerator(args, tokenizer)
    elif method == 'MARS':
        return MARSGenerator(args, tokenizer)
    elif method == 'MCMC':
        return MHGenerator(args, tokenizer)
    elif method == 'PPO':
        return PPOGenerator(args, tokenizer)
    elif method == 'SAC':
        return SACGenerator(args, tokenizer)
    elif method == 'RANDOM':
        return RandomTrajGenerator(args)
    else:
        raise ValueError(f"Unknown generator method: {method}")
