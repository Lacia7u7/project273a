# models/rgcn.py
import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from models.heads import TypewiseInputProjector, EncounterClassifier

class RGCNModel(nn.Module):
    """
    Relational-GCN over a heterogeneous mini-batch.
    - Accepts the same constructor args your factory passes to other models.
    - Uses TypewiseInputProjector to:
        * Linear-project dense features for 'encounter'
        * Embed index features for all other node types
      so that every type outputs hidden_dim features.
    - Flattens the hetero batch into a single homogeneous graph with relation ids,
      applies stacked RGCNConv layers, and classifies encounter nodes.
    """
    def __init__(self,
                 metadata,
                 config,
                 enc_input_dim=None,
                 type_vocab_sizes=None,
                 **kwargs):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        # Stable mapping edge-type -> relation id (must match edge_types order)
        self.relation_index = {edge_type: i for i, edge_type in enumerate(edge_types)}
        num_relations = len(self.relation_index)

        # Hyperparams with safe fallbacks
        self.hidden_dim = int(getattr(config.model, "hidden_dim", 256))
        self.num_layers = int(getattr(config.model, "num_layers", 2))
        self.dropout_p = float(getattr(config.model, "dropout", 0.0))
        num_bases = getattr(config.model, "rgcn_bases", None)
        if isinstance(num_bases, int) and num_bases <= 0:
            num_bases = None

        # Optional eager projector init (preferred, matches your other models)
        self._lazy_projector = True
        self.proj = None
        if enc_input_dim is not None and type_vocab_sizes is not None:
            input_dims = {"encounter": int(enc_input_dim)}
            embed_dims = {}
            for nt in self.node_types:
                if nt == "encounter":
                    continue
                # how many discrete ids to embed for this type
                vocab_n = int(type_vocab_sizes.get(nt, 0))
                input_dims[nt] = vocab_n
                embed_dims[nt] = self.hidden_dim
            self.proj = TypewiseInputProjector(
                input_dims=input_dims,
                out_dim=self.hidden_dim,
                embed_dims=embed_dims,
                dropout=self.dropout_p,
                activation="relu",
            )
            self._lazy_projector = False

        # RGCN layers (projector already makes all node types -> hidden_dim)
        self.convs = nn.ModuleList([
            RGCNConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases
            )
            for _ in range(self.num_layers)
        ])

        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout_p)
        self.classifier = EncounterClassifier(self.hidden_dim)

    @torch.no_grad()
    def _build_projector_if_needed(self, x_dict):
        if not self._lazy_projector or self.proj is not None:
            return
        # Infer input dims from the current batch
        input_dims, embed_dims = {}, {}
        for nt, x in x_dict.items():
            if nt == "encounter":
                # dense features [N, F]
                input_dims[nt] = int(x.shape[1])
            else:
                # index features [N, 1] (max id + 1 is safe upper bound)
                vmax = int(x.max().item()) + 1 if x.numel() > 0 else 0
                input_dims[nt] = vmax
                embed_dims[nt] = self.hidden_dim
        self.proj = TypewiseInputProjector(
            input_dims=input_dims,
            out_dim=self.hidden_dim,
            embed_dims=embed_dims,
            dropout=self.dropout_p,
            activation="relu",
        )

    def _hetero_to_homo(self, x_dict_proj, edge_index_dict):
        """
        Concatenate typewise node features and remap hetero edges with offsets.
        Returns:
          X: [sum_N, hidden_dim]
          edge_index: [2, E_total] (on same device as X)
          edge_type:  [E_total] long (on same device as X)
          node_offsets: dict[type] -> starting row in X
        """
        device = next(iter(x_dict_proj.values())).device

        # Build concatenated node matrix + offsets
        node_offsets = {}
        all_x = []
        offset = 0
        for nt in self.node_types:
            if nt not in x_dict_proj:
                continue
            feat = x_dict_proj[nt]  # [N_t, hidden_dim]
            all_x.append(feat)
            node_offsets[nt] = offset
            offset += feat.size(0)

        X = torch.cat(all_x, dim=0) if len(all_x) > 0 else torch.empty(0, self.hidden_dim, device=device)

        # Collect edges and relation ids
        all_edge_indices = []
        all_edge_types = []
        for (src, rel, dst), e_idx in edge_index_dict.items():
            if src not in node_offsets or dst not in node_offsets:
                continue  # type not present in the current mini-batch
            if e_idx.numel() == 0:
                continue
            rel_id = self.relation_index[(src, rel, dst)]
            src_off = node_offsets[src]
            dst_off = node_offsets[dst]
            # Create per-dimension offsets on the correct device/dtype
            offs = e_idx.new_tensor([[src_off], [dst_off]])
            ei = e_idx + offs
            all_edge_indices.append(ei)
            all_edge_types.append(
                e_idx.new_full((ei.size(1),), fill_value=rel_id, dtype=torch.long)
            )

        if len(all_edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_type = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(all_edge_indices, dim=1)
            edge_type = torch.cat(all_edge_types, dim=0)

        return X, edge_index, edge_type, node_offsets

    def forward(self, x_dict, edge_index_dict):
        # Ensure projector exists (handles both eager and lazy init flows)
        self._build_projector_if_needed(x_dict)

        # Project per-type inputs to hidden_dim
        x_dict_proj = self.proj(x_dict)  # dict[type] -> [N_t, hidden_dim]

        # Build homogeneous graph view
        X, edge_index, edge_type, node_offsets = self._hetero_to_homo(x_dict_proj, edge_index_dict)

        # R-GCN stack
        h = X
        for conv in self.convs:
            h = conv(h, edge_index, edge_type)
            h = self.act(h)
            h = self.drop(h)

        # Slice back encounter nodes
        enc_off = node_offsets.get("encounter", 0)
        enc_count = x_dict_proj["encounter"].size(0)
        enc_h = h[enc_off: enc_off + enc_count]

        # Binary logit for each encounter
        logit = self.classifier(enc_h).view(-1)
        return logit
