import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from models.heads import TypewiseInputProjector, EncounterClassifier

class RGCNModel(nn.Module):
    def __init__(self, metadata, config):
        super().__init__()
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        # Map each edge type to a relation ID
        self.relation_index = {edge_type: i for i, edge_type in enumerate(metadata[1])}
        num_relations = len(self.relation_index)
        self.node_types = metadata[0]
        self.proj = None
        # Define RGCNConv layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i>0 else self.hidden_dim
            out_dim = self.hidden_dim
            conv = RGCNConv(in_dim, out_dim, num_relations=num_relations, num_bases=config.model.rgcn_bases or None)
            self.convs.append(conv)
        self.classifier = EncounterClassifier(self.hidden_dim)
    def forward(self, x_dict, edge_index_dict):
        # lazy init projector
        if self.proj is None:
            input_dims = {}
            embed_dims = {}
            for node_type, x in x_dict.items():
                if node_type == 'encounter':
                    input_dims[node_type] = x.shape[1]
                else:
                    input_dims[node_type] = int(x.max().item()) + 1 if x.numel()>0 else 0
                    embed_dims[node_type] = self.hidden_dim
            self.proj = TypewiseInputProjector(input_dims, self.hidden_dim, embed_dims)
        x_dict_proj = self.proj(x_dict)
        # Concatenate all node features into a single tensor for RGCN
        node_offsets = {}
        all_x = []
        offset = 0
        for node_type in self.node_types:
            if node_type in x_dict_proj:
                feat = x_dict_proj[node_type]
                all_x.append(feat)
                node_offsets[node_type] = offset
                offset += feat.size(0)
        X = torch.cat(all_x, dim=0)
        # Build global edge_index and edge_type
        all_edge_indices = []
        all_edge_types = []
        for (src, rel, dst), e_idx in edge_index_dict.items():
            rel_id = self.relation_index[(src, rel, dst)]
            src_off = node_offsets[src]
            dst_off = node_offsets[dst]
            ei = e_idx + torch.tensor([[src_off],[dst_off]])
            num_e = ei.shape[1]
            all_edge_indices.append(ei)
            all_edge_types.append(torch.full((num_e,), rel_id, dtype=torch.long))
        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_type = torch.cat(all_edge_types, dim=0)
        # RGCN layers
        out = X
        for conv in self.convs:
            out = conv(out, edge_index, edge_type)
            out = torch.relu(out)
        # Extract encounter node outputs
        enc_off = node_offsets['encounter']
        enc_count = x_dict_proj['encounter'].size(0)
        enc_out = out[enc_off: enc_off+enc_count]
        logit = self.classifier(enc_out).view(-1)
        return logit
