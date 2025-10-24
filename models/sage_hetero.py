import torch
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

from models.wrapper import GridParam, SklearnLikeGNN


def _make_act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":  return nn.ReLU()
    if name == "gelu":  return nn.GELU()
    if name in ("silu", "swish"): return nn.SiLU()
    if name in ("lrelu", "leaky_relu"): return nn.LeakyReLU(0.1)
    if name == "elu":   return nn.ELU()
    return nn.ReLU()


class GraphSAGEModel(SklearnLikeGNN):
    def __init__(self, metadata, config, enc_input_dim: int, type_vocab_sizes: dict, device=None):
        """
        metadata: (node_types, edge_types)
        enc_input_dim: float-feature dim for 'encounter'
        type_vocab_sizes: {node_type: num_nodes} for index-based nodes (Embedding path)
        """
        super().__init__(
            metadata=metadata,
            config=config,
            enc_input_dim=enc_input_dim,
            type_vocab_sizes=type_vocab_sizes,
            device=device,
        )
        self.hidden = hidden = int(config.model.hidden_dim)
        node_types, edge_types = metadata

        act = _make_act(getattr(config.model, "act", "relu"))
        dropout_p = float(getattr(config.model, "dropout", 0.0))
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # ---------- 1) Per-type input encoders ----------
        # Encounter: always Linear
        self.input_encoders = nn.ModuleDict()
        self.input_encoders["encounter"] = nn.Linear(enc_input_dim, hidden)

        # Others: choose Embedding or (Lazy)Linear using config.graph.feature_dims
        feat_dims = getattr(config.graph, "feature_dims", {}) or {}
        for ntype in node_types:
            if ntype == "encounter":
                continue

            float_in_dim = feat_dims.get(ntype, None)
            if isinstance(float_in_dim, int) and float_in_dim > 0:
                # user declares float features with known dim
                self.input_encoders[ntype] = nn.Linear(float_in_dim, hidden)
            elif isinstance(float_in_dim, int) and float_in_dim == 0:
                # float features but let it infer at first forward
                self.input_encoders[ntype] = nn.LazyLinear(hidden)
            elif float_in_dim is None:
                # default: index features -> Embedding
                vocab_sz = int(type_vocab_sizes.get(ntype, 0))
                if vocab_sz <= 0:
                    raise ValueError(
                        f"Node type '{ntype}' has no feature_dims entry and no vocab size. "
                        f"Either set graph.feature_dims['{ntype}']=<in_dim or 0>, "
                        f"or provide type_vocab_sizes['{ntype}'] > 0."
                    )
                self.input_encoders[ntype] = nn.Embedding(vocab_sz, hidden)
            else:
                raise ValueError(f"feature_dims['{ntype}'] must be int (>=0) or omitted, got {float_in_dim!r}")

        # ---------- 2) Homogeneous SAGE stack -> hetero ----------
        num_layers = max(1, int(getattr(config.model, "num_layers", 2)))

        class HomoSAGE(nn.Module):
            def __init__(self, hidden, num_layers, act, drop):
                super().__init__()
                self.convs = nn.ModuleList([SAGEConv((-1, -1), hidden) for _ in range(num_layers)])
                self.act = act
                self.drop = drop
                self.num_layers = num_layers

            def forward(self, x, edge_index):
                for li, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if li < self.num_layers - 1:  # no act/drop after last layer
                        x = self.act(x)
                        x = self.drop(x)
                return x

        self.base = HomoSAGE(hidden, num_layers, act, self.drop)
        self.gnn = to_hetero(self.base, metadata, aggr="mean")

        # ---------- 3) Classifier over encounter nodes ----------
        self.classifier = nn.Linear(hidden, 1)

        if device is not None:
            self.to(device)

    def _encode_inputs(self, x_dict):
        """Apply the right encoder (Embedding or Linear) to each type."""
        h = {}
        for ntype, x in x_dict.items():
            enc = self.input_encoders[ntype]
            if isinstance(enc, nn.Embedding):
                if x.dtype != torch.long:
                    raise TypeError(
                        f"Expected Long indices for '{ntype}' (Embedding), got {x.dtype}. "
                        f"If you have float features, set config.graph.feature_dims['{ntype}'] to the input dim (or 0 for LazyLinear)."
                    )
                h[ntype] = enc(x)
            else:
                if not x.is_floating_point():
                    x = x.float()
                h[ntype] = enc(x)
        return h

    def forward(self, x_dict, edge_index_dict):
        h_dict = self._encode_inputs(x_dict)      # per-type to hidden
        out_dict = self.gnn(h_dict, edge_index_dict)  # hetero SAGE
        enc_out = out_dict["encounter"]           # [B_enc, hidden]
        return self.classifier(enc_out).view(-1)  # [B_enc]

    @classmethod
    def suggested_grid(cls):
        return {
            "model.hidden_dim": GridParam(
                values=[128, 192, 256, 320],
                description="Hidden channels for GraphSAGE layers.",
            ),
            "model.num_layers": GridParam(
                values=[2, 3, 4],
                description="Number of message passing layers.",
            ),
            "model.dropout": GridParam(
                values=[0.0, 0.1, 0.25],
                description="Dropout before GraphSAGE layers.",
            ),
            "model.act": GridParam(
                values=["relu", "gelu", "silu"],
                description="Activation function used inside GraphSAGE layers.",
            ),
        }
