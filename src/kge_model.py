from collections.abc import Sequence
import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_

    
class MDistMult(nn.Module):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MDistMult, self).__init__()
        self.emb_dim = emb_dim
        self.name = "MDistMult"
        self.hidden_drop_rate = 0.2
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])

    def inference(self, r_idx, entities_idx):
        r = self.R(r_idx)
        e = self.E(entities_idx)
        x = r * torch.prod(e, dim = 2)
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=-1)
        return x
    
    def forward(self, batch, edge_list=None, rel_list=None):
        r_idx, entities_idx = batch.get_fact()
        return self.inference(r_idx, entities_idx)


