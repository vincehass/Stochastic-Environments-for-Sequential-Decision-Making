from lib.generator.gfn import DBGFlowNetGenerator, StochasticDBGFlowNetGenerator

def get_generator(args, tokenizer):
    if args.method == 'db':
        return StochasticDBGFlowNetGenerator(args, tokenizer)
    else:
        return DBGFlowNetGenerator(args, tokenizer)

