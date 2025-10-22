import torch
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

# --- helpers ---------------------------------------------------------------

def _move_to_device(obj, device, non_blocking: bool):
    """Recursively move tensors/containers to device with optional non_blocking."""
    if hasattr(obj, "to"):
        try:
            return obj.to(device, non_blocking=non_blocking)  # Tensor / PyG Batch
        except TypeError:
            return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device, non_blocking) for v in obj)
    return obj

def _forward_and_labels(model, batch, device):
    """
    Handles your hetero-graph case (x_dict/edge_index_dict) and a generic dict fallback.
    Returns (logits, labels, bs).
    """
    # --- PyG HeteroData mini-batch (your second snippet) ---
    if hasattr(batch, "x_dict") and hasattr(batch, "edge_index_dict"):
        logits = model(batch.x_dict, batch.edge_index_dict)  # shape: [B, ...]
        # encounter node labels:
        enc = batch["encounter"]
        # Prefer explicit batch_size if present, else infer from y length:
        bs = getattr(enc, "batch_size", getattr(enc, "y", None).size(0) if hasattr(enc, "y") else None)
        if bs is None:
            raise RuntimeError("Could not infer encounter batch size.")
        labels = enc.y[:bs].to(device).float()
        return logits, labels, bs

    # --- Generic dict-style batch fallback: expects "inputs"/"labels" or "x"/"y" ---
    if isinstance(batch, dict):
        x = batch.get("inputs", batch.get("x"))
        y = batch.get("labels", batch.get("y"))
        if x is None or y is None:
            raise RuntimeError("Batch dict must contain ('inputs'/'labels') or ('x'/'y').")
        logits = model(x)
        bs = y.shape[0]
        labels = y.to(device).float()
        return logits, labels, bs

    # --- If you have other batch formats, add branches here ---
    raise RuntimeError("Unsupported batch type for forward_and_labels().")

# --- unified training loop -------------------------------------------------

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    config,
    rt,                       # from apply_system_config(...)
    batch_bar=None,           # pass a tqdm wrapper if you like; else None
    grad_clip: float | None = None
):
    amp_enabled = rt["amp_enabled"]
    amp_dtype   = rt["amp_dtype"]
    scaler      = rt["scaler"]

    # tqdm or plain loader
    iterator = batch_bar if batch_bar is not None else train_loader

    # safer non_blocking only makes sense on CUDA
    non_blocking = bool(config.system.dataloader.non_blocking and torch.cuda.is_available())

    model.train()
    total_loss = 0.0
    total_seen = 0

    for batch in iterator:
        # 1) move to device (covers PyG HeteroData and dicts)
        batch = _move_to_device(batch, device, non_blocking=non_blocking)

        # 2) zero grad
        optimizer.zero_grad(set_to_none=True)

        # 3) forward + loss with AMP if enabled
        if amp_enabled:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, labels, bs = _forward_and_labels(model, batch, device)
                loss = criterion(logits[:bs], labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16 path (GradScaler disabled) or user disabled
                loss.backward()
                if grad_clip:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            logits, labels, bs = _forward_and_labels(model, batch, device)
            loss = criterion(logits[:bs], labels)
            loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 4) book-keeping
        total_loss += loss.item() * bs
        total_seen += bs

        # 5) tqdm postfix (lr + current loss)
        if batch_bar is not None:
            lr = optimizer.param_groups[0]["lr"]
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

    avg_loss = total_loss / max(1, total_seen)
    return avg_loss
