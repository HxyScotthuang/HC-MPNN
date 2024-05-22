import argparse

parser = argparse.ArgumentParser(description='Arguments')

parser.add_argument("--gpus", type=int, default=0,
                    help="gpus")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size")
parser.add_argument("--accum_iter", type=int, default=1,
                    help="gradient accumulation step")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed")
parser.add_argument("-d", "--dataset", type=str, default='FB-AUTO',
                    help="dataset to use")

# configuration for test mode
parser.add_argument("--test", action='store_true', default=False,
                    help="load stat from dir and directly test")
parser.add_argument("--test_by_arity", action='store_true', default=False,
                    help="test by arity")
parser.add_argument("--model_name", type=str, default="",
                    help="model name for test mode")

parser.add_argument("--model", type=str, default="HC-MPNN",
                    help="which model we are using?")

# configuration for stat training
parser.add_argument("--n_epoch", type=int, default=20,
                    help="number of minimum training epochs on each time step")
parser.add_argument("--eval_every", type=int, default=1,
                    help="evaluate every x epochs")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0,
                    help="set the weight decay for Adam optimizer")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="dropout rate for each layer")

parser.add_argument("--norm", type=str, default="layer_norm",choices=["layer_norm", "none"],
                    help="norm for each layer")
parser.add_argument("--neg_ratio", "-nr", type=int, default=10,
                    help="number of negative sample")   
parser.add_argument("--adversarial_temperature", type=float, default=0.5,
                    help="adversarial temperature setting")               

parser.add_argument('--batch_per_epoch', type=int, default=None,
                    help='number of batches per epoch, set to None to use all batches')

parser.add_argument('--positional_encoding','-pe', type=str, default="static", choices=["learnable", "static", "constant", "one-hot"],
                    help='the postional encoding method')
parser.add_argument("--initialization",'-i', type=str, default="standard", choices=["standard", "withoutpos", "withoutrel", "withoutposrel"],
                    help="initialization, for ablation study only")
parser.add_argument("--dependent", action='store_true', default=False,
                    help="whether to use query dependent message function")
parser.add_argument("--use_triton" , action='store_true', default=False, 
                    help="whether to use triton for training")

# configuration for evaluating
parser.add_argument("--metric", type=list, default=['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10'],
                    help="evaluating metrics")


# configuration for layers

parser.add_argument("--hidden_dim", type=int, default=100,
                    help="dimension list of hidden layers")
parser.add_argument("--num_layer", type=int, default=1,
                     help= "number of hidden layers")
parser.add_argument("--short_cut", action='store_true', default=True,
                    help="whether residual connection")
                   

args, unparsed = parser.parse_known_args()
print(args)  
# print(unparsed)  
