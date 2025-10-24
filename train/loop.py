from __future__ import annotations
from typing import Optional, Callable, Dict, Any
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import average_precision_score


class Trainer:
    """
    Minimal stateful trainer that you drive from your notebook/script loop.

    What it handles:
      - AMP/bf16 & GradScaler, grad clipping
      - non_blocking device moves
      - scheduler.step() each epoch
      - TB writer logging (train loss & val metric)
      - validation (AUPRC) + early stopping + best-state saving/loading
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: torch.device,
        config,
        rt: Dict[str, Any],
        *,
        scheduler=None,
        writer=None,
        logger=None,
        save_best_fn: Optional[Callable[[Dict[str, Any]], str]] = None,  # returns path
        early_stopping_patience: int = 10,
        val_every: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.writer = writer
        self.logger = logger
        self.save_best_fn = save_best_fn
        self.early_stopping_patience = int(early_stopping_patience)
        self.val_every = int(val_every)

        # runtime (from apply_system_config(...))
        self.amp_enabled: bool = rt["amp_enabled"]
        self.amp_dtype = rt["amp_dtype"]
        self.scaler = rt["scaler"]

        # non_blocking only makes sense on CUDA
        self.non_blocking: bool = bool(
            getattr(config.system.dataloader, "non_blocking", False) and torch.cuda.is_available()
        )

        # best/early-stop tracking
        self.best_metric: float = float("-inf")
        self.best_state: Optional[Dict[str, Any]] = None
        self.patience_counter: int = 0

    # ----------------- helpers -----------------

    def _move_to_device(self, obj):
        """Recursively move tensors/containers to device with optional non_blocking."""
        if hasattr(obj, "to"):
            try:
                return obj.to(self.device, non_blocking=self.non_blocking)
            except TypeError:
                return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(v) for v in obj)
        return obj

    def _forward_and_labels(self, batch):
        """Supports PyG HeteroData batches and a dict(x/y) fallback."""
        if hasattr(batch, "x_dict") and hasattr(batch, "edge_index_dict"):
            logits = self.model(batch.x_dict, batch.edge_index_dict)
            enc = batch["encounter"]
            bs = getattr(enc, "batch_size", getattr(enc, "y", None).size(0) if hasattr(enc, "y") else None)
            if bs is None:
                raise RuntimeError("Could not infer encounter batch size.")
            labels = enc.y[:bs].to(self.device).float()
            return logits, labels, bs

        if isinstance(batch, dict):
            x = batch.get("inputs", batch.get("x"))
            y = batch.get("labels", batch.get("y"))
            if x is None or y is None:
                raise RuntimeError("Batch dict must contain ('inputs'/'labels') or ('x'/'y').")
            logits = self.model(x)
            return logits, y.to(self.device).float(), y.shape[0]

        raise RuntimeError("Unsupported batch type.")

    # ----------------- public API -----------------

    def train_epoch(self, train_loader, *, batch_bar=None, grad_clip: Optional[float] = None) -> float:
        """Run one training epoch and return average loss."""
        iterator = batch_bar if batch_bar is not None else train_loader
        self.model.train()
        total_loss, total_seen = 0.0, 0

        for batch in iterator:
            batch = self._move_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_enabled:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    logits, labels, bs = self._forward_and_labels(batch)
                    loss = self.criterion(logits[:bs], labels)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
            else:
                logits, labels, bs = self._forward_and_labels(batch)
                loss = self.criterion(logits[:bs], labels)
                loss.backward()
                if grad_clip:
                    clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            total_loss += float(loss.item()) * bs
            total_seen += bs

            if batch_bar is not None:
                lr = self.optimizer.param_groups[0]["lr"]
                batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        avg_loss = total_loss / max(1, total_seen)

        # scheduler + train loss logging are per-epoch concerns
        if self.scheduler is not None:
            self.scheduler.step()
        if self.writer is not None:
            # caller provides the global step (epoch) when logging
            pass

        return avg_loss

    @torch.no_grad()
    def validate_auprc(self, val_graph) -> float:
        """Compute AUPRC on a PyG HeteroData graph."""
        self.model.eval()
        logits = self.model(val_graph.x_dict, val_graph.edge_index_dict)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = val_graph["encounter"].y.detach().cpu().numpy()
        return float(average_precision_score(labels, probs))

    def update_after_validation(self, metric: float, *, epoch: int) -> bool:
        """
        Update TB/early-stop/best-state after a validation.
        Returns True if early stopping should trigger.
        """
        if self.writer is not None:
            self.writer.add_scalar("Val/AUPRC", metric, epoch)

        improved = metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.patience_counter = 0
            self.best_state = {"model": self.model.state_dict(), "epoch": epoch, "val_auprc": metric}
            if self.save_best_fn is not None:
                path = self.save_best_fn(self.best_state)
                if self.logger:
                    self.logger.info(f"Saved best artifact to: {path}")
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.early_stopping_patience

    def log_train_loss(self, avg_loss: float, *, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar("Loss/train", avg_loss, epoch)

    def load_best(self):
        if self.best_state:
            self.model.load_state_dict(self.best_state["model"])
            if self.logger:
                self.logger.info(
                    f"Loaded best model from epoch {self.best_state['epoch']} "
                    f"with Val AUPRC={self.best_state['val_auprc']:.4f}"
                )
        return self.best_state
