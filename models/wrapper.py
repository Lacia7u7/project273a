from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_metrics


@dataclass(frozen=True)
class GridParam:
    """Metadata describing a grid-search friendly hyper-parameter."""

    values: Sequence[Any]
    description: str
    applies_to: Optional[str] = None  # e.g. "config.model.hidden_dim"


class SklearnLikeGNN(nn.Module):
    """Mixin that equips torch models with sklearn-like APIs."""

    def __init__(
        self,
        *,
        metadata: Any = None,
        config: Any = None,
        enc_input_dim: Optional[int] = None,
        type_vocab_sizes: Optional[Mapping[str, int]] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.config = config
        self.enc_input_dim = enc_input_dim
        self.type_vocab_sizes = type_vocab_sizes
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self._fitted: bool = False
        self.to(self._device)

    # ------------------------------------------------------------------
    # Fitting utilities
    # ------------------------------------------------------------------
    def _batch_to_device(self, batch: Any, device: torch.device) -> Any:
        if hasattr(batch, "to"):
            try:
                return batch.to(device)
            except TypeError:
                return batch.to(device, non_blocking=False)
        if isinstance(batch, dict):
            return {k: self._batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._batch_to_device(v, device) for v in batch)
        return batch

    def _forward_and_labels(self, batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(batch, "x_dict") and hasattr(batch, "edge_index_dict"):
            logits = self(batch.x_dict, batch.edge_index_dict)
            enc = batch["encounter"]
            y = getattr(enc, "y", None)
            if y is None:
                raise RuntimeError("Encounter node is missing labels ('y').")
            if hasattr(enc, "batch_size"):
                y = y[: enc.batch_size]
            return logits, y.to(device).float()
        if isinstance(batch, dict):
            x = batch.get("inputs", batch.get("x"))
            y = batch.get("labels", batch.get("y"))
            if x is None or y is None:
                raise RuntimeError("Dictionary batch must expose inputs/x and labels/y keys.")
            logits = self(x)
            return logits, y.to(device).float()
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            if hasattr(x, "to"):
                x = x.to(device)
            if hasattr(y, "to"):
                y = y.to(device)
            logits = self(x)
            return logits, y.float()
        raise RuntimeError("Unsupported batch type for forward pass.")

    def _ensure_dataloader(self, data: Any, shuffle: bool = False) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, Iterable):
            return DataLoader(data, batch_size=1, shuffle=shuffle)
        raise TypeError("Training data must be a DataLoader or Iterable of batches.")

    def fit(
        self,
        train_data: Any,
        y: Optional[Any] = None,
        *,
        val_data: Any | None = None,
        epochs: int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        grad_clip: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        verbose: bool = False,
        use_trainer: bool = False,
        trainer_kwargs: Optional[Mapping[str, Any]] = None,
        runtime: Optional[MutableMapping[str, Any]] = None,
        validation_metric: str = "auprc",
        load_best_state: bool = True,
    ) -> "SklearnLikeGNN":
        device = torch.device(device) if device is not None else self._device
        self._device = device
        self.to(device)
        loader = self._ensure_dataloader(train_data, shuffle=True)
        val_reference = val_data
        val_loader = (
            self._ensure_dataloader(val_reference, shuffle=False)
            if (val_reference is not None and not use_trainer)
            else None
        )

        criterion = criterion or nn.BCEWithLogitsLoss()
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-3)

        if use_trainer:
            self._fit_with_trainer(
                loader,
                val_data=val_reference,
                epochs=epochs,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                grad_clip=grad_clip,
                device=device,
                verbose=verbose,
                trainer_kwargs=dict(trainer_kwargs or {}),
                runtime=runtime,
                validation_metric=validation_metric,
                load_best_state=load_best_state,
            )
        else:
            self._fit_simple(
                loader,
                val_loader=val_loader,
                epochs=epochs,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                grad_clip=grad_clip,
                device=device,
                verbose=verbose,
            )
        self._fitted = True
        return self

    def _fit_simple(
        self,
        loader: DataLoader,
        *,
        val_loader: Optional[DataLoader],
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any],
        grad_clip: Optional[float],
        device: torch.device,
        verbose: bool,
    ) -> None:
        for epoch in range(int(epochs)):
            self.train()
            total_loss = 0.0
            n_samples = 0
            for batch in loader:
                batch = self._batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits, labels = self._forward_and_labels(batch, device)
                labels = labels.view_as(logits)
                loss = criterion(logits, labels)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * labels.numel()
                n_samples += int(labels.numel())
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss={total_loss / max(n_samples, 1):.4f}")
            if scheduler is not None:
                scheduler.step()
            if val_loader is not None:
                self.evaluate_loader(val_loader, device=device)

    def _fit_with_trainer(
        self,
        loader: DataLoader,
        *,
        val_data: Any,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any],
        grad_clip: Optional[float],
        device: torch.device,
        verbose: bool,
        trainer_kwargs: Dict[str, Any],
        runtime: Optional[MutableMapping[str, Any]],
        validation_metric: str,
        load_best_state: bool,
    ) -> None:
        try:
            from train.loop import Trainer
        except ImportError as exc:
            raise RuntimeError("Trainer class is unavailable; set use_trainer=False to fallback.") from exc

        trainer_runtime = self._build_runtime(device, runtime)
        trainer_config = self._resolve_trainer_config()

        early_stopping_patience = int(trainer_kwargs.pop("early_stopping_patience", 5))
        val_every = int(trainer_kwargs.pop("val_every", 1))

        trainer = Trainer(
            model=self,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=trainer_config,
            rt=trainer_runtime,
            scheduler=scheduler,
            **trainer_kwargs,
            early_stopping_patience=early_stopping_patience,
            val_every=val_every,
        )

        for epoch in range(1, int(epochs) + 1):
            avg_loss = trainer.train_epoch(loader, grad_clip=grad_clip)
            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")
            trainer.log_train_loss(avg_loss, epoch=epoch)

            if scheduler is not None:
                scheduler.step()

            if val_data is None or (epoch % val_every) != 0:
                continue

            metric_value = None
            metric_value = self._validate_for_trainer(
                trainer,
                val_data,
                device=device,
                validation_metric=validation_metric,
            )

            if metric_value is None:
                continue

            if trainer.update_after_validation(metric_value, epoch=epoch):
                if verbose:
                    print("Early stopping triggered by Trainer.")
                break

        if load_best_state:
            trainer.load_best()

    def _validate_for_trainer(
        self,
        trainer: Any,
        val_data: Any,
        *,
        device: torch.device,
        validation_metric: str,
    ) -> Optional[float]:
        if hasattr(val_data, "x_dict") and hasattr(val_data, "to"):
            return float(trainer.validate_auprc(val_data.to(device)))

        loader = self._ensure_dataloader(val_data, shuffle=False)
        metrics = self.evaluate_loader(loader, device=device)
        return metrics.get(validation_metric)

    def _resolve_trainer_config(self) -> Any:
        if self.config is not None and hasattr(self.config, "system"):
            system = getattr(self.config.system, "dataloader", None)
            if system is not None:
                return self.config
        return SimpleNamespace(system=SimpleNamespace(dataloader=SimpleNamespace(non_blocking=False)))

    def _build_runtime(
        self,
        device: torch.device,
        runtime: Optional[MutableMapping[str, Any]],
    ) -> MutableMapping[str, Any]:
        if runtime is not None:
            runtime.setdefault("amp_enabled", bool(runtime.get("amp_enabled", False)))
            runtime.setdefault("amp_dtype", runtime.get("amp_dtype", torch.float16))
            runtime.setdefault(
                "scaler",
                runtime.get("scaler", torch.cuda.amp.GradScaler(enabled=False)),
            )
            return runtime

        amp_enabled = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        return {
            "amp_enabled": amp_enabled,
            "amp_dtype": torch.float16,
            "scaler": scaler,
        }

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        data: Any,
        *,
        device: Optional[torch.device | str] = None,
        sigmoid: bool = True,
    ) -> np.ndarray:
        device = torch.device(device) if device is not None else self._device
        loader = self._ensure_dataloader(data, shuffle=False)
        self.eval()
        probs: List[np.ndarray] = []
        for batch in loader:
            batch = self._batch_to_device(batch, device)
            logits, _ = self._forward_and_labels(batch, device)
            out = torch.sigmoid(logits) if sigmoid else logits
            probs.append(out.detach().cpu().numpy().reshape(-1))
        return np.concatenate(probs, axis=0) if probs else np.empty(0)

    @torch.no_grad()
    def predict(
        self,
        data: Any,
        *,
        device: Optional[torch.device | str] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        probs = self.predict_proba(data, device=device, sigmoid=True)
        return (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_loader(
        self,
        data: Any,
        *,
        device: Optional[torch.device | str] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        device = torch.device(device) if device is not None else self._device
        loader = self._ensure_dataloader(data, shuffle=False)
        self.eval()
        y_true: List[np.ndarray] = []
        y_prob: List[np.ndarray] = []
        y_pred: List[np.ndarray] = []
        for batch in loader:
            batch = self._batch_to_device(batch, device)
            logits, labels = self._forward_and_labels(batch, device)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            y_true.append(labels.detach().cpu().numpy().reshape(-1))
            y_prob.append(probs.detach().cpu().numpy().reshape(-1))
            y_pred.append(preds.detach().cpu().numpy().reshape(-1))
        if not y_true:
            return {}
        y_true_arr = np.concatenate(y_true)
        y_prob_arr = np.concatenate(y_prob)
        y_pred_arr = np.concatenate(y_pred)
        base_metrics = compute_metrics(y_true_arr, y_prob_arr, threshold=threshold)
        metrics: Dict[str, float] = {
            "roc_auc": float(base_metrics.get("auroc", float("nan"))),
            "auprc": float(base_metrics.get("auprc", float("nan"))),
            "brier": float(base_metrics.get("brier", float("nan"))),
            "ece": float(base_metrics.get("ece", float("nan"))),
            "precision": float(base_metrics.get("precision_pos", float("nan"))),
            "recall": float(base_metrics.get("recall_pos", float("nan"))),
            "f1": float(base_metrics.get("f1_pos", float("nan"))),
            "balanced_accuracy": float(base_metrics.get("balanced_accuracy", float("nan"))),
            "accuracy": float(np.mean(y_true_arr == y_pred_arr)),
        }
        return metrics

    # ------------------------------------------------------------------
    # sklearn compatibility hooks
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "metadata": self.metadata,
            "config": self.config,
            "enc_input_dim": self.enc_input_dim,
            "type_vocab_sizes": self.type_vocab_sizes,
            "device": self._device,
        }
        if deep and hasattr(self.config, "dict"):
            cfg_dict = self.config.dict()
            params.update({f"config__{k}": v for k, v in cfg_dict.items()})
        return params

    def set_params(self, **params: Any) -> "SklearnLikeGNN":
        for key, value in params.items():
            if key.startswith("config__") and hasattr(self.config, "__setattr__"):
                _, attr = key.split("__", 1)
                if hasattr(self.config, attr):
                    setattr(self.config, attr, value)
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Grid search helpers
    # ------------------------------------------------------------------
    @classmethod
    def suggested_grid(cls) -> Mapping[str, GridParam]:
        return {}

    @classmethod
    def grid_search_params_space_names(cls) -> List[str]:
        return list(cls.suggested_grid().keys())


__all__ = ["GridParam", "SklearnLikeGNN"]
