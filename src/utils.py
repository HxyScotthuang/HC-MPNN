import numpy as np
import torch
import random
import os
import random
from .dataset import Dataset
from collections import deque
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import index_sort

    
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
    if dataset in ['FB-AUTO', "Wikipeople"]:
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


def bfs_hypergraph_edges(node, edge_list, k, m):
    """
    Perform BFS on a hypergraph with downsampling, tracking the edges visited.

    :param node: Starting node for BFS.
    :param edge_list: List of hyperedges in the hypergraph.
    :param k: Depth of BFS traversal.
    :param m: Maximum number of neighbors to sample for each node.
    :return: Set of indices of edges visited during BFS.
    """
    visited_nodes = set([node])
    visited_edges = set()
    queue = deque([(node, 0)])

    while queue:
        current_node, depth = queue.popleft()
        if depth < k:
            for i, edge in enumerate(edge_list):
                if current_node in edge:
                    visited_edges.add(i)
                    neighbors = [neighbor for neighbor in edge if neighbor != current_node]

                    # Downsampling neighbors
                    if len(neighbors) > m:
                        neighbors = random.sample(neighbors, m)

                    for neighbor in neighbors:
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            queue.append((neighbor, depth + 1))
    return visited_edges

def create_edge_mask(visited_edges, total_edges):
    """
    Create a mask for edges in the hypergraph.

    :param visited_edges: Set of indices of edges visited.
    :param total_edges: Total number of edges in the hypergraph.
    :return: List representing the mask for edges.
    """
    return [i in visited_edges for i in range(total_edges)]


def generate_subgraph_union_edges_mask(nodes, edge_list, k, m=100):
    """
    Generate the union of subgraphs for a set of start nodes.

    :param nodes: List of start nodes for BFS.
    :param edge_list: List of hyperedges in the hypergraph.
    :param k: Depth of BFS traversal.
    :return: Set of indices of edges in the union of subgraphs.
    """
    visited_edges = set()
    for node in nodes:
        visited_edges |= bfs_hypergraph_edges(node, edge_list, k, m)
    mask = create_edge_mask(visited_edges, len(edge_list))
    return mask




def coo_to_csr(row, col, edge_types, num_nodes=None):

    # Row is the source node, col is the destination node. 
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    row, perm = index_sort(row, max_value=num_nodes)
    col = col[perm]
    types = edge_types[perm]
    rowptr = index2ptr(row, num_nodes)

    return rowptr, col, types


def coo_to_csr_hyper(row, col, edge_types, pos_index, num_nodes=None):
    # The only differenc is that now col is a 2D tensor
    # Row is the source node, col is the destination node list. 
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    row, perm = index_sort(row, max_value=num_nodes) # TODO: alternatively we can use stable
    col = col[:,perm] # 
    types = edge_types[perm]
    pos_index = pos_index[:, perm]
    rowptr = index2ptr(row, num_nodes)
    return rowptr, col, types, pos_index


def smart_split(edge_index):
    max_arity = edge_index.shape[0]
    file = torch.cat([
            torch.cat([edge_index[:arity,:], edge_index[arity+1:,:]], dim = 0) # exclude the current arity
            for arity in range(max_arity)]
            , dim = 1
            )
    return file

def preprocess_triton_hypergraph(edge_index, edge_type, num_node):
    max_arity = edge_index.shape[0]
    destination = edge_index.flatten()
    source = smart_split(edge_index)
    edge_type = edge_type.repeat(max_arity) # expand as if destination

    # Apply the sequence tensor to the non-zero elements
    pos_node_in_edge = torch.arange(1, max_arity+1).unsqueeze(1).repeat(1, edge_index.shape[1]).to(edge_index.device)
    pos_index = smart_split(pos_node_in_edge)
    # print("destination", destination, "source", source, "edge_type", edge_type, "pos_index", pos_index)


    assert pos_index.shape == source.shape, "pos_index and source should have the same shape"
    # Remove the destination node that is 0
    mask = destination != 0
    destination = destination[mask]
    source = source[:, mask]
    edge_type = edge_type[mask]
    pos_index  = pos_index[:, mask]

    rowptr, indices, etypes, pos_index = coo_to_csr_hyper(destination, source, edge_type, pos_index, num_node)
    return rowptr, indices, etypes, pos_index
