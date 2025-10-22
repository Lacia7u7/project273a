# models/heads.py
import torch
from torch import nn

class TypewiseInputProjector(nn.Module):
    """
    - 'encounter' -> Linear on dense float features
    - others      -> Embedding on integer indices (padding_idx=0, UNKNOWN=0)
    Safe for empty mini-batches; optional clamping to avoid out-of-range indices.
    """
    def __init__(
        self,
        input_dims: dict,                 # {"encounter": D_in, "<type>": num_embeddings, ...}
        hidden_dim: int,
        embed_dims: dict | None = None,   # {"<type>": emb_dim}; default -> hidden_dim
        padding_idx: int = 0,
        dropout: float = 0.0,
        safe_clamp: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.padding_idx = int(padding_idx)
        self.safe_clamp = bool(safe_clamp)

        self.embed_layers = nn.ModuleDict()
        self.lin_layers = nn.ModuleDict()
        self._num_embeddings: dict[str, int] = {}
        self.act = nn.ReLU()
        self.drop = nn.Dropout(float(dropout))

        for ntype, in_dim in input_dims.items():
            in_dim = int(in_dim)

            if ntype == "encounter":
                # Dense features -> Linear to hidden_dim
                self.lin_layers[ntype] = nn.Linear(in_dim, self.hidden_dim)
            else:
                # Indices -> Embedding (ensure at least 1 row for safety)
                num_emb = max(1, in_dim)
                emb_dim = int((embed_dims or {}).get(ntype, self.hidden_dim))
                self.embed_layers[ntype] = nn.Embedding(
                    num_embeddings=num_emb,
                    embedding_dim=emb_dim,
                    padding_idx=self.padding_idx,
                )
                self._num_embeddings[ntype] = num_emb
                if emb_dim != self.hidden_dim:
                    self.lin_layers[ntype] = nn.Linear(emb_dim, self.hidden_dim)

    def _safe_idx(self, idx: torch.Tensor, ntype: str) -> torch.Tensor:
        if not self.safe_clamp or idx.numel() == 0:
            return idx
        return idx.clamp_(0, self._num_embeddings[ntype] - 1)

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        for ntype, x in x_dict.items():
            if ntype in self.embed_layers:  # indices
                # Make 1-D long vector of indices
                idx = x.long().view(-1)
                if idx.numel() == 0:
                    out_nt = x.new_zeros((0, self.hidden_dim))
                else:
                    idx = self._safe_idx(idx, ntype)
                    emb = self.embed_layers[ntype](idx)  # [N, emb_dim]
                    out_nt = self.lin_layers[ntype](emb) if ntype in self.lin_layers else emb
            else:
                # Dense features for encounter
                xf = x if x.dtype.is_floating_point else x.float()
                if xf.numel() == 0:
                    out_nt = xf.new_zeros((0, self.hidden_dim))
                else:
                    out_nt = self.lin_layers[ntype](xf)  # [N, hidden_dim]

            out[ntype] = self.drop(self.act(out_nt))
        return out


class EncounterClassifier(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(float(dropout))
        self.lin = nn.Linear(int(in_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, 1))
        return self.lin(self.drop(x))
