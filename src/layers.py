# Running the provided HypergraphNetwork class with the corrected message_passing method
import torch
from torch import nn
import numpy as np
from torch.nn.init import xavier_normal_
from src.utils import static_positional_encoding, onehot_positional_encoding, preprocess_triton_hypergraph

from torch_geometric.utils import scatter

class HypergraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relation, max_arity = 6, dropout=0.2, norm = "layer_norm", positional_encoding = "learnable", dependent = False, use_triton = False):
        super(HypergraphLayer, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.num_relation = num_relation
        self.norm_type = norm
        self.dependent = dependent
        self.use_triton = use_triton
        if norm == "layer_norm":
            self.norm = nn.LayerNorm(out_channels)
        elif norm == "batch_norm":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()

        if self.dependent:
            self.relation_linear = nn.Linear(in_channels, num_relation * in_channels)
        else:
            self.rel_embedding = nn.Embedding(num_relation, in_channels)

        
        self.dropout = nn.Dropout(p = dropout,inplace=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.pe_mode = positional_encoding
        if self.pe_mode in ["static", "constant", "one-hot"]:
            if self.pe_mode == "static":
                static_encodings = static_positional_encoding(max_arity, in_channels)
            elif self.pe_mode == "constant":
                static_encodings = torch.ones(max_arity + 1, in_channels)
            elif self.pe_mode == "one-hot":
                static_encodings = onehot_positional_encoding(max_arity, in_channels)
            # Fix the encoding
            self.pos_embedding = nn.Embedding.from_pretrained(static_encodings, freeze=True)
            self.pos_embedding.weight.data[0] = torch.ones(in_channels)
        elif self.pe_mode == "learnable":
            self.pos_embedding = nn.Embedding(max_arity + 1, in_channels)
        else:
            raise ValueError("Unknown positional encoding type")
        
        
       
        

    def forward(self, node_features, query, edge_list, rel):
        self.pos_embedding.weight.data[0] = torch.ones(self.in_channels)
        batch_size, node_size, input_dim = node_features.shape
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation_vector = self.relation_linear(query).view(batch_size, self.num_relation, input_dim)
        else:
            relation_vector = None
            
        
        node_features[:, 0, :] = 0 # Clear the padding node for message agg

        if self.use_triton:
            from src.triton_rspmm import HyperRelConvSumAggr
            pos_embedding = self.pos_embedding.weight.unsqueeze(1).expand(-1, batch_size, -1).flatten(1) # expand the positional encoding for batch, and compress with the feature size
            if self.dependent:
                relation_vector = relation_vector.transpose(0,1).flatten(1).transpose(0,1)
            else:
                relation_vector = self.rel_embedding.weight.unsqueeze(1).expand(-1, batch_size, -1).flatten(1).transpose(0,1) # expand the relation embedding for batch, and compress with the feature size
            edge_list_trans = edge_list.transpose(0,1)
            node_features_flatten = node_features.transpose(0,1).flatten(1)
            rowptr, indices, etypes, pos_index = preprocess_triton_hypergraph(edge_list_trans, rel, num_node = node_size)
            out  = HyperRelConvSumAggr.apply(node_features_flatten, rowptr, indices, node_size, etypes, relation_vector, pos_embedding, pos_index, self.alpha, 0)
            out = out.view(node_size, batch_size, -1).transpose(0,1)
        else:
            message = self.messages(node_features, relation_vector, edge_list, rel)
            out = self.aggregates(message, edge_list, rel, node_features)
            out[:, 0, :] = 0 # Clear the padding node for learning

        out = (self.linear(torch.cat([out, node_features], dim=-1)))
        out = self.dropout(out)
    
        if self.norm_type == "batch_norm":

            out = torch.einsum("ijk->ikj", out)
            out = self.norm(out)
            out = torch.einsum("ijk->ikj", out)
        elif self.norm_type == "layer_norm":
            out = self.norm(out)
        else:
            pass   
        return out
    
    def messages(self, node_features, relation_vector, hyperedges, relations):
        device = node_features.device
        # Set the node feature of node 0 to be always 0 so that it does not contribute to the messages

        batch_size, node_size, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Create a batch index array
        batch_indices = torch.arange(batch_size, device=hyperedges.device)[:, None, None]  # Shape: [batch_size, 1, 1]

        # Repeat batch indices to match the shape of hyperedges
        # New shape of batch_indices: [batch_size, edge_size, max_arity]
        batch_indices = batch_indices.repeat(1, hyperedges.shape[0], hyperedges.shape[1]) # TODO: maybe replace with torch.expand

        # Use advanced indexing to gather node features
        # The resulting shape will be [batch_size, edge_size, max_arity, input_dim]
        sum_node_positional = node_features[batch_indices, hyperedges]


        # Compute positional encodings for nodes in each hyperedge
        # [batch_size, edge_size, max_arity, input_dim]
        positional_encodings = self.computer_pos_encoding(hyperedges, batch_size, device)

        # Sum node features and positional encodings
        # Final shape: [batch_size, edge_size, max_arity, input_dim]
        sum_node_positional = self.alpha* sum_node_positional + (1-self.alpha)*positional_encodings
        # sum_node_positional = sum_node_positional + positional_encodings

        # sum_node_positional is actually the ej+pj for each node that is located in each edge, indicated by its max_arity
        # we need to produce another [batch_size, edge_size, max_arity, input_dim], that compute *_{j \neq i}(e_j+p_j), which replace the i pos
        # We can do this by a clever "shift" operation. Compute the cumulative product in both directions [batch_size, edge_size, max_arity, input_dim]
        messages = self.all_but_one_trick(sum_node_positional, batch_size, edge_size, input_dim, device)
        
        # Get relation vectors for each edge and expand
        # Shape: [edge_size] -> [batch_size,  edge_size, max_arity, input_dim]
        if relation_vector is not None:
            assert self.dependent
            relation_vectors = relation_vector.index_select(1, relations)
            relation_vectors = relation_vectors.unsqueeze(2).expand(-1, -1, max_arity, -1)
        else:
            assert not self.dependent
            relation_vectors = self.rel_embedding(relations).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, max_arity, -1)
        
        messages = messages * relation_vectors

        # shape: [batch_size,  edge_size, max_arity, input_dim]
        return messages

    def aggregates(self, messages, hyperedges, relations, node_features):
        # Messages has shape [batch_size,  edge_size, max_arity, input_dim], where each edges stores max_arity messages, each belongs to the position of the node at that max_arity
        # hyperedges has shape [batch_size, edge_size, max_arity], where each edge stores the node index that belongs to that edge
        # relations has shape [batch_size, edge_size], where each edge stores the relation index that belongs to that edge
        # node_features has shape [batch_size, node_size, input_dim], where each node stores the feature vector of that node
        batch_size, node_size, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Expand and reshape messages for gathering
        # Shape: [batch_size, edge_size, max_arity, input_dim] -> [batch_size, edge_size * max_arity, input_dim]
        messages_expanded = messages.view(batch_size, edge_size * max_arity, input_dim)

        # Gather messages based on hyperedges indices
        # New shape after gather: [batch_size, node_size, input_dim]

        node_aggregate = scatter(messages_expanded, hyperedges.flatten(), dim = 1, reduce = "sum", dim_size=node_size)
        
        # The output is a tensor of shape [batch_size, node_size, input_dim], where each node stores the aggregated message from all the edges that it belongs to
        return node_aggregate


        
    def all_but_one_trick(self, sum_node_positional, batch_size, edge_size, input_dim, device):
        cumprod_forward = torch.cumprod(sum_node_positional, dim=2)
        cumprod_backward = torch.cumprod(sum_node_positional.flip(dims=[2]), dim=2).flip(dims=[2])

        # Shift and combine
        shifted_forward = torch.cat([torch.ones(batch_size, edge_size, 1, input_dim).to(device), cumprod_forward[:, :, :-1, :]], dim=2)
        shifted_backward = torch.cat([cumprod_backward[:, :, 1:, :], torch.ones(batch_size, edge_size, 1, input_dim).to(device)], dim=2)

        # Combine the two shifted products
        return shifted_forward * shifted_backward

    def computer_pos_encoding(self, hyperedges, batch_size, device):
        
        sequence_tensor = torch.arange(1, hyperedges.size(1) + 1, device = device).unsqueeze(0)
        # Apply the sequence tensor to the non-zero elements
        pos_node_in_edge = torch.where(hyperedges != 0, sequence_tensor, torch.zeros_like(hyperedges, device = device))

        # [batch_size, edge_size, max_arity, input_dim]
        return self.pos_embedding(pos_node_in_edge).unsqueeze(0).expand(batch_size, -1, -1, -1)


