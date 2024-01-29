import numpy as np
import torch
import random
import os
import random
from .dataset import Dataset
from collections import deque

    
def set_rand_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # Disable hash randomization
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True




def get_device(args):
    if args.gpus != -1:
        device = torch.device("cuda:"+str(args.gpus))
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(args):
    random_num = random.randint(1, 1e8)
    model_name = "model:{}-data:{}-lr:{}-bsize:{}-neg:{}-dim:{}-{}"\
        .format(args.model, args.dataset, args.lr, args.batch_size, args.neg_ratio,
        args.hidden_dim, random_num)
    
    file_name = f"{model_name}_working_dir.tmp"
    
    working_dir = os.path.join("experiments", args.dataset)

    with open(file_name, "w") as fout:
        fout.write(working_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    with open(file_name, "r") as fin:
        working_dir = fin.read()
    os.remove(file_name)
    os.chdir(working_dir)
    return working_dir, model_name, random_num


def load_data(dataset, device):
    if dataset in ['FB-AUTO', 'JF17K']:
        return Dataset(ds_name=dataset, device=device)
    elif dataset in ["JF-IND", "WP-IND", "MFB-IND"]:
        return Dataset(ds_name=dataset, device=device, inductive_dataset=True)
    elif dataset in [f"{name}-IND-V{i}" for name in ["FB", "WN"] for i in range(1,5)]:
        return Dataset(ds_name=dataset, device=device, inductive_dataset=True, binary_dataset=True)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def get_rank(sim_scores):
    # Assumes the test fact is the first one
    rank = (sim_scores >= sim_scores[:, 0, np.newaxis]).sum(axis=-1)
    return list(rank)

def static_positional_encoding(max_arity, input_dim):
    """
    Generate a static positional encoding.

    Args:
    - max_arity (int): Maximum arity for which to create positional encodings.
    - input_dim (int): Dimension of the input feature vector.

    Returns:
    - torch.Tensor: A tensor containing positional encodings for each position.
    """
    # Initialize the positional encoding matrix
    position = torch.zeros(max_arity + 1, input_dim)

    # Compute the positional encodings
    for pos in range(max_arity + 1):
        # position[pos, pos] = 1
        for i in range(0, input_dim, 2):
            position[pos, i] = np.sin(pos / (10000 ** ((2 * i) / input_dim)))
            if i + 1 < input_dim:
                position[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))


    return position

def onehot_positional_encoding(max_arity, input_dim):
    """
    Generate a onehot positional encoding.

    Args:
    - max_arity (int): Maximum arity for which to create positional encodings.
    - input_dim (int): Dimension of the input feature vector.

    Returns:
    - torch.Tensor: A tensor containing positional encodings for each position.
    """
    # Initialize the positional encoding matrix
    position = torch.zeros(max_arity + 1, input_dim)

    # Compute the positional encodings
    for pos in range(max_arity + 1):
        position[pos, pos] = 1

    return position

