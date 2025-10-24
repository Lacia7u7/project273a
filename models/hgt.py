# models/hgt.py
import inspect
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from models.heads import TypewiseInputProjector, EncounterClassifier
from models.wrapper import GridParam, SklearnLikeGNN

class HGTModel(SklearnLikeGNN):
    """
    HGT with:
      - global, eager input projector sizing (no batch-based sizing)
      - safe embeddings with padding_idx=0 and clamping
      - device move after full construction
    """
    def __init__(
        self,
        metadata,
        config,
        *,
        enc_input_dim: int,            # graph_train['encounter'].x.size(-1)
        type_vocab_sizes: dict,        # {ntype: graph_train[ntype].num_nodes} for non-encounter
        device: torch.device | None = None,
    ):
        super().__init__(
            metadata=metadata,
            config=config,
            enc_input_dim=enc_input_dim,
            type_vocab_sizes=type_vocab_sizes,
            device=device,
        )
        self.hidden_dim = int(config.model.hidden_dim)

        # -------- Input Projector (GLOBAL sizes) --------
        input_dims = {"encounter": int(enc_input_dim)}
        embed_dims = {}
        for ntype, n_nodes in type_vocab_sizes.items():
            n_nodes = int(n_nodes)
            if ntype != "encounter":
                # allow 0 in config; projector will internally use max(1, n_nodes)
                input_dims[ntype] = n_nodes
                embed_dims[ntype] = self.hidden_dim

        self.proj = TypewiseInputProjector(
            input_dims=input_dims,
            hidden_dim=self.hidden_dim,
            embed_dims=embed_dims,
            padding_idx=0,                               # UNKNOWN=0
            dropout=getattr(config.model, "dropout", 0.0),
            safe_clamp=True,
        )

        # -------- HGT conv stack --------
        self.convs = nn.ModuleList()
        num_layers = int(getattr(config.model, "num_layers", 2))
        heads = int(getattr(config.model, "heads", 4))
        for _ in range(num_layers):
            kwargs = {"heads": heads}
            # PyG compatibility (older versions don't have 'group')
            if "group" in inspect.signature(HGTConv.__init__).parameters:
                kwargs["group"] = "sum"
            self.convs.append(HGTConv(self.hidden_dim, self.hidden_dim, metadata, **kwargs))

        self.classifier = EncounterClassifier(self.hidden_dim, dropout=getattr(config.model, "dropout", 0.0))

        # -------- Move entire model to device (AFTER creating all submodules) --------
        if device is not None:
            self.to(device)

    @classmethod
    def suggested_grid(cls):
        return {
            "model.hidden_dim": GridParam(
                values=[128, 192, 256, 320],
                description="Hidden representation size for all node types.",
            ),
            "model.num_layers": GridParam(
                values=[2, 3, 4],
                description="Number of stacked HGT convolution layers.",
            ),
            "model.heads": GridParam(
                values=[2, 4, 6],
                description="Number of attention heads per HGT layer.",
            ),
            "model.dropout": GridParam(
                values=[0.0, 0.1, 0.25],
                description="Dropout applied to input projections and classifier.",
            ),
        }

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.proj(x_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return self.classifier(x_dict["encounter"]).view(-1)
