import torch
from torch.nn import BCEWithLogitsLoss

def make_criterion(graph_train, config, device):
    pos_weight_cfg = getattr(config.model.loss, "pos_weight", 1.0) if hasattr(config.model, "loss") else 1.0

    if isinstance(pos_weight_cfg, str) and pos_weight_cfg.lower() == "balanced":
        y = graph_train["encounter"].y
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pw = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        return BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))

    if isinstance(pos_weight_cfg, (int, float)) and pos_weight_cfg != 1.0:
        return BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight_cfg), dtype=torch.float32, device=device))

    return BCEWithLogitsLoss()
