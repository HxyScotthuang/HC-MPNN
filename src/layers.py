import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch_geometric.utils import scatter

from src.utils import static_positional_encoding, onehot_positional_encoding

class HypergraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relation, max_arity = 6, dropout=0.2, norm = "layer_norm", positional_encoding = "learnable", dependent = False):
        super(HypergraphLayer, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.num_relation = num_relation
        self.norm_type = norm
        self.dependent = dependent
        if norm == "layer_norm":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if self.dependent:
            self.relation_linear = nn.Linear(in_channels, num_relation * in_channels)
        else:
            self.rel_embedding = nn.Embedding(num_relation, in_channels, padding_idx = 0)
            self.rel_embedding.weight.data[0] = torch.ones(in_channels)
            xavier_normal_(self.rel_embedding.weight.data[1:])

        
        self.dropout = nn.Dropout(p = dropout,inplace=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        if positional_encoding in ["static", "constant", "one-hot"]:
            if positional_encoding == "static":
                static_encodings = static_positional_encoding(max_arity + 1, in_channels)
            elif positional_encoding == "constant":
                static_encodings = torch.ones(max_arity + 1, in_channels)
            elif positional_encoding == "one-hot":
                static_encodings = onehot_positional_encoding(max_arity + 1, in_channels)
            # Fix the encoding
            self.pos_embedding = nn.Embedding.from_pretrained(static_encodings, freeze=True)
            self.pos_embedding.weight.data[0] = torch.ones(in_channels)
        elif positional_encoding == "learnable":
            self.pos_embedding = nn.Embedding(max_arity + 1, in_channels)
        else:
            raise ValueError("Unknown positional encoding type")
        
        
       
        

    def forward(self, node_features, query, edge_list, rel):
        self.pos_embedding.weight.data[0] = torch.ones(self.in_channels)
        batch_size, _, input_dim = node_features.shape
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation_vector = self.relation_linear(query).view(batch_size, self.num_relation, input_dim)
        else:
            relation_vector = None
            self.rel_embedding.weight.data[0] = torch.ones(self.in_channels)

        node_features[:, 0, :] = 0 # Clear the padding node for message agg
        message = self.messages(node_features, relation_vector, edge_list, rel)
        out = self.aggregates(message, edge_list, node_features)
        out[:, 0, :] = 0 # Clear the padding node for learning

        out = (self.linear(torch.cat([out, node_features], dim=-1)))
        out = self.dropout(out)
        
        if self.norm_type == "layer_norm":
            out = self.norm(out)
        else:
            pass
        return out
    
    def messages(self, node_features, relation_vector, hyperedges, relations):
        device = node_features.device

        batch_size, _, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Create a batch index array
        batch_indices = torch.arange(batch_size, device=hyperedges.device)[:, None, None]  # Shape: [batch_size, 1, 1]

        # Repeat batch indices to match the shape of hyperedges
        batch_indices = batch_indices.repeat(1, hyperedges.shape[0], hyperedges.shape[1])

        # Use advanced indexing to gather node features
        hyperedges_gathered = node_features[batch_indices, hyperedges]


        # Compute positional encodings for nodes in each hyperedge
        positional_encodings = self.computer_pos_encoding(hyperedges, batch_size, device)

        # Sum node features and positional encodings
        sum_node_positional = self.alpha* hyperedges_gathered + (1-self.alpha)*positional_encodings

        # sum_node_positional is actually the ej+pj for each node that is located in each edge, indicated by its max_arity
        # we need to produce another [batch_size, edge_size, max_arity, input_dim], that compute *_{j \neq i}(e_j+p_j), which replace the i pos
        # We can do this by a clever "shift" operation. Compute the cumulative product in both directions 
        messages = self.all_but_one_trick(sum_node_positional, batch_size, edge_size, input_dim, device)
        
        # Get relation vectors for each edge and expand
        if relation_vector is not None:
            assert self.dependent
            relation_vectors = relation_vector.index_select(1, relations)
            relation_vectors = relation_vectors.unsqueeze(2).expand(-1, -1, max_arity, -1)
        else:
            assert not self.dependent
            relation_vectors = self.rel_embedding(relations).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, max_arity, -1)
        
        messages = messages * relation_vectors

        return messages

    def aggregates(self, messages, hyperedges, node_features):
        batch_size, node_size, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Expand and reshape messages for gathering
        messages_expanded = messages.view(batch_size, edge_size * max_arity, input_dim)

        # Gather messages based on hyperedges indices
        node_aggregate = scatter(messages_expanded, hyperedges.flatten(), dim = 1, reduce = "sum", dim_size=node_size)
        
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

        return self.pos_embedding(pos_node_in_edge).unsqueeze(0).expand(batch_size, -1, -1, -1)


