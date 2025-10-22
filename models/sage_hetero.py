import torch
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

class GraphSAGEModel(nn.Module):
    def __init__(self, metadata, config):
        super().__init__()
        hidden = config.model.hidden_dim
        # Define a homogeneous GraphSAGE (for to_hetero conversion)
        class HomoSAGE(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv((-1, -1), hidden)
                self.conv2 = SAGEConv((-1, -1), hidden)
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index)
                return x
        self.base_model = HomoSAGE()
        self.model = to_hetero(self.base_model, metadata, aggr='mean')
        self.classifier = nn.Linear(hidden, 1)
    def forward(self, x_dict, edge_index_dict):
        out_dict = self.model(x_dict, edge_index_dict)
        # We assume 'encounter' is a key in out_dict
        enc_out = out_dict['encounter']
        logit = self.classifier(enc_out).view(-1)
        return logit
