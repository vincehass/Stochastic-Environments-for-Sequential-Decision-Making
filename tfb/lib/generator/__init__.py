from lib.generator.gfn import DBGFlowNetGenerator, StochasticDBGFlowNetGenerator
from lib.generator.generators_friends import MARSGenerator, MHGenerator, PPOGenerator, SACGenerator, RandomTrajGenerator



def get_generator(args, tokenizer):
    if args.method == 'db':
        return StochasticDBGFlowNetGenerator(args, tokenizer)
    elif args.method == 'static_gfn':
        return DBGFlowNetGenerator(args, tokenizer)
    elif args.method == 'mars':
        return MARSGenerator(args, tokenizer)
    elif args.method == 'mh':
        return MHGenerator(args, tokenizer)
    elif args.method == 'ppo':
        return PPOGenerator(args, tokenizer)
    elif args.method == 'sac':
        return SACGenerator(args, tokenizer)
    elif args.method == 'random':
        return RandomTrajGenerator(args)
    else:
        raise ValueError(f"Unknown generator method: {args.method}")
    
    

# #MARS
# class MARSGenerator:
#     def __init__(self, args):
#         self.args = args
#         self.net = nn.Sequential(
#             nn.Linear(args.state_dim, args.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, args.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, args.action_dim)