import torch
from torch import nn
from torch.nn import functional as F

from src.layers import HypergraphLayer
from src.utils import static_positional_encoding, onehot_positional_encoding

class HC_MPNN(nn.Module):
    def __init__(self, hidden_dims, num_nodes, num_relation, num_layer, positional_encoding = "learnable",
                 short_cut=True,  num_mlp_layer=2, max_arity=6, dropout=0.2,norm = "layer_norm", initialization = "standard", 
                dependent = False):
        super(HC_MPNN,self).__init__()
        self.name = "HC-MPNN"
        input_dim = hidden_dims
        self.dims = [input_dim] + [hidden_dims] * num_layer
        self.num_nodes = num_nodes + 1
        self.num_relation = num_relation
        self.short_cut = short_cut  # whether to use residual connections between layers
        
        self.max_arity  = max_arity

        self.query = nn.Embedding(self.num_relation, input_dim, padding_idx = 0)
        
        if positional_encoding in ["static", "constant", "one-hot"]:
            if positional_encoding == "static":
                static_encodings = static_positional_encoding(max_arity + 1, input_dim)
            elif positional_encoding == "constant":
                static_encodings = torch.ones(max_arity + 1, input_dim)
            elif positional_encoding == "one-hot":
                static_encodings = onehot_positional_encoding(max_arity + 1, input_dim)
            # Fix the encoding
            self.position = nn.Embedding.from_pretrained(static_encodings, freeze=True)

        elif positional_encoding == "learnable":
            self.position = nn.Embedding(max_arity + 1, input_dim)
        self.position.weight.data[0] = torch.zeros(input_dim)


        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(HypergraphLayer(self.dims[i], self.dims[i + 1], self.num_relation, dropout=dropout, norm = norm, positional_encoding = positional_encoding, dependent = dependent))

        self.feature_dim = hidden_dims + input_dim 
        self.initialization = initialization

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def inference(self, r_idx, entities_idx, arity, edge_list, rel_list):
        batch_size = r_idx.shape[0]
        entities_idx  = torch.einsum("ijk->ikj", entities_idx)

        all_idx = entities_idx

        mask_for_diff = torch.all(all_idx[:,:,0].unsqueeze(-1).expand(-1, -1, all_idx.size(-1)) == all_idx, dim=-1)
        
        # find for each batch, which position is searched (empty)
        pos_index_to_search = (mask_for_diff == False).int().argmax(dim=1)

        # Assertion checks
        assert torch.all(r_idx[:, 0].unsqueeze(-1).expand(-1,r_idx.size(1)) == r_idx), "All relation types should be the same in one batch"
        assert torch.all(torch.sum(mask_for_diff, dim=-1) >= all_idx.size(1) -1 ), "Is it exactly one of the ei_idx are different?"
        assert torch.all(arity[:, 0].unsqueeze(-1).expand(-1,arity.size(1)) == arity), "All arities should be the same in one batch"
        assert torch.all(pos_index_to_search <= arity[:,0]), "The position to search should be less than arity"

        query = self.query(r_idx[:, 0])  

        init_feature = torch.zeros(batch_size, self.num_nodes, self.dims[0], device=r_idx.device)
        result_tensor = torch.ones((batch_size, self.max_arity), dtype=torch.int, device=r_idx.device)
        range_tensor = torch.arange(self.max_arity, device=result_tensor.device).expand(batch_size, self.max_arity)
        arity_range = arity[:,0].unsqueeze(1).expand(-1, self.max_arity)
        
        mask = range_tensor < arity_range
        result_tensor *= mask
        zero_out_mask = range_tensor == pos_index_to_search.unsqueeze(1)
        result_tensor[zero_out_mask] = 0

        index_arity_without_self = all_idx[:,:,0] * result_tensor # Masking to find tensor
        index_arity_without_self = index_arity_without_self.unsqueeze(-1).expand(-1, -1, self.dims[0]).to(torch.int64)
        # add relational embedding

        if self.initialization in ["standard", "withoutpos"]:
            
            query_feature = torch.zeros(batch_size, self.num_nodes, self.dims[0], device=r_idx.device)

            query_feature.scatter_add_(dim=1, 
                                    index=index_arity_without_self,
                                    src=query.unsqueeze(1).expand(-1, self.max_arity, -1)
                                    )
            
            init_feature += query_feature

        if self.initialization in ["standard", "withoutrel"]:

            pos_src_index = result_tensor * torch.arange(1, self.max_arity+1, device=result_tensor.device).expand(batch_size, self.max_arity)

            pos_src = self.position(pos_src_index)

            pos_init_feature = torch.zeros(batch_size, self.num_nodes, self.dims[0], device=r_idx.device)
            pos_init_feature.scatter_add_(dim=1,
                                    index= index_arity_without_self,
                                    src = pos_src
                                    )
            init_feature += pos_init_feature

        if self.initialization in ["withoutposrel"]:
            constant_src = torch.ones(batch_size, self.num_nodes, self.dims[0], device=r_idx.device)
            constant_feature = torch.zeros(batch_size, self.num_nodes, self.dims[0], device=r_idx.device)
            constant_feature.scatter_add_(dim=1,
                                    index= index_arity_without_self,
                                    src = constant_src
                                    )
            init_feature += constant_feature

        init_feature[:, 0, :] = 0 # clear the padding node

        # Passing in the layer:
        layer_input = init_feature

        for layer in self.layers:
            hidden = F.relu(layer(layer_input, query, edge_list, rel_list))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        # Remind the model which query we are looking for
        output = torch.cat([output, query.unsqueeze(1).expand(-1, output.size(1), -1)], dim=-1)
        in_batch_tensor = all_idx * torch.logical_not(mask_for_diff).int().unsqueeze(-1).expand(-1, -1, all_idx.size(-1))
        
        collapsed_tensor = in_batch_tensor[torch.any(in_batch_tensor != 0, dim=2)]

        feature = output.gather(1, collapsed_tensor.unsqueeze(-1).expand(-1, -1, output.size(-1)))
        score = self.mlp(feature).squeeze(-1)

        return score

    def forward(self, batch, edge_list, rel_list):
        r_idx, entities_idx = batch.get_fact()
        arity = batch.get_arity()
        arity = torch.from_numpy(arity).to(r_idx.device)

        # In training mode, we remove the easy edges to reduce overfitting
        edge_list, rel_list = self.remove_easy_edge(r_idx, entities_idx, edge_list, rel_list)

        return self.inference(r_idx, entities_idx, arity, edge_list, rel_list)


    def remove_easy_edge(self, r_idx, entities_idx, edge_list, rel_list):
        # Remove the easy edges to reduce overfitting. Actually important for model to generalize!
        # Initialize an empty mask with the same size as the edge list
        all_edge_rel = torch.cat([edge_list, rel_list.unsqueeze(-1)], dim=-1)
        easy_edge = torch.cat([entities_idx, r_idx.unsqueeze(-1)], dim=-1).flatten(0,1)
        all_edge_rel, easy_edge = all_edge_rel.transpose(0,1), easy_edge.transpose(0,1)
        index = self.edge_match(all_edge_rel, easy_edge)[0]
        remove_mask = ~self.index_to_mask(index, len(edge_list))
        
        # Filter out the edges that are to be removed
        filtered_edge_list = edge_list[remove_mask,:]
        filtered_rel_list = rel_list[remove_mask]
        return filtered_edge_list, filtered_rel_list
    
    def edge_match(self, edge_index, query_index):
        # This is from NBFNet-PyG implementation
        # O((n + q)logn) time
        # O(n) memory
        # edge_index: big underlying graph
        # query_index: edges to match

        # preparing unique hashing of edges, base: (max_node, max_relation) + 1
        base = edge_index.max(dim=1)[0] + 1
        # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
        # idea: max number of edges = num_nodes * num_relations
        # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
        # given a tuple (h, r), we will search for all other existing edges starting from head h
        scale = base.cumprod(0)
        scale = scale[-1] // scale

        # hash both the original edge index and the query index to unique integers
        edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
        edge_hash, order = edge_hash.sort()
        query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

        # matched ranges: [start[i], end[i])
        start = torch.bucketize(query_hash, edge_hash)
        end = torch.bucketize(query_hash, edge_hash, right=True)
        # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
        num_match = end - start

        # generate the corresponding ranges
        offset = num_match.cumsum(0) - num_match
        range = torch.arange(num_match.sum(), device=edge_index.device)
        range = range + (start - offset).repeat_interleave(num_match)

        return order[range], num_match

    def index_to_mask(self, index, size):
        index = index.view(-1)
        size = int(index.max()) + 1 if size is None else size
        mask = index.new_zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

