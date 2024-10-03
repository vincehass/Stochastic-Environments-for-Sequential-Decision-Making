from lib.generator.gfn import StochasticDBGFlowNetGenerator, StochasticKLGFlowNetGenerator, StochasticKL2GFlowNetGenerator
from lib.generator.generators_friends import DeterminsticDBGFlowNetGenerator, MARSGenerator, MHGenerator, PPOGenerator, SACGenerator, RandomTrajGenerator



def get_generator(args, tokenizer):
    method = args.method[0] if isinstance(args.method, list) else args.method  # Handle list input
    print(method)
    if method == 'deterministic_dbg':
        return DeterminsticDBGFlowNetGenerator(args, tokenizer)
    elif method == 'stochastic_dbg':
        return StochasticDBGFlowNetGenerator(args, tokenizer)
    elif method == 'stochastic_klg':
        return StochasticKLGFlowNetGenerator(args, tokenizer)
    elif method == 'stochastic_klg2':
        return StochasticKL2GFlowNetGenerator(args, tokenizer)
    elif method == 'mars':
        return MARSGenerator(args, tokenizer)
    elif method == 'mh':
        return MHGenerator(args, tokenizer)
    elif method == 'ppo':
        return PPOGenerator(args, tokenizer)
    elif method == 'sac':
        return SACGenerator(args, tokenizer)
    elif method == 'random':
        return RandomTrajGenerator(args)
    else:
        raise ValueError(f"Unknown generator method: {method}")
