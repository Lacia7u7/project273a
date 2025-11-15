import os
import subprocess
import warnings
from dataclasses import dataclass
from typing import Dict, Callable, Iterable, Mapping, Any, Tuple, Optional, Sequence, List

import pandas as pd
import torch
from joblib import delayed, Parallel
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from grid_search.gnn import GridSearchReport, _clone_config, _set_nested_attr
from models.rgcn import RGCNModel


DataFactory = Callable[[Any], Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]]


_DEFAULT_PARAM_GRID: Mapping[str, Sequence[Any]] = {
    "model.hidden_dim": [128, 192, 256],
    "model.num_layers": [2, 3, 4],
    "model.dropout": [0.0, 0.1, 0.2],
    "model.rgcn_bases": [-1, 0, 16],
    "train.optimizer.name": ["Adam", "AdamW"],
    "train.optimizer.lr": [1e-3, 5e-4, 3e-4],
    "train.optimizer.weight_decay": [0.0, 1e-4, 5e-4],
    "train.batching.batch_size_encounters": [64, 128, 256],
    "train.epochs": [20],
}


_OPTIMIZER_REGISTRY: Dict[str, Callable[[Iterable[torch.nn.Parameter], float, float], torch.optim.Optimizer]] = {
    "adam": lambda params, lr, wd: torch.optim.Adam(params, lr=lr, weight_decay=wd),
    "adamw": lambda params, lr, wd: torch.optim.AdamW(params, lr=lr, weight_decay=wd),
    "sgd": lambda params, lr, wd: torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd),
}


@dataclass
class FastGridSearchConfig:
    metadata: Any
    base_config: Any
    data_factory: DataFactory
    model_kwargs: Optional[Dict[str, Any]] = None
    param_grid: Optional[Mapping[str, Sequence[Any]]] = None
    metric_name: str = "val_auprc"
    greater_is_better: bool = True
    n_jobs: int = 4
    device: str = "cuda:0"
    start_cuda_mps: bool = True
    cuda_memory_fraction: float = 0.10


def _ensure_cuda_mps(pipe_dir: str = "/tmp/nvidia-mps", log_dir: str = "/tmp/nvidia-mps") -> None:
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", pipe_dir)
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", log_dir)
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # Gracefully continue when CUDA MPS tools are unavailable (e.g., on CPU-only systems).
        pass


def _configure_worker(device: str | torch.device, memory_fraction: float) -> torch.device:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA device requested but not available; falling back to CPU.",
                RuntimeWarning,
            )
            return torch.device("cpu")
        torch.cuda.set_device(requested)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=requested)
        except Exception:
            # Ignore environments that do not support fractional limits (e.g., older CUDA versions).
            pass
        return requested
    return requested


def _prepare_config(base_config: Any, params: Mapping[str, Any]) -> Any:
    config = _clone_config(base_config)
    for path, value in params.items():
        _set_nested_attr(config, path, value)
    return config


def _build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    opt_cfg = getattr(config.train, "optimizer", None)
    if opt_cfg is None:
        raise ValueError("Config must expose train.optimizer settings.")
    name = str(getattr(opt_cfg, "name", "adam")).lower()
    lr = float(getattr(opt_cfg, "lr", 1e-3))
    weight_decay = float(getattr(opt_cfg, "weight_decay", 0.0))
    builder = _OPTIMIZER_REGISTRY.get(name)
    if builder is None:
        raise ValueError(f"Unsupported optimizer '{name}'. Available: {sorted(_OPTIMIZER_REGISTRY)}")
    return builder(model.parameters(), lr, weight_decay)


def _run_single_trial(
    index: int,
    params: Mapping[str, Any],
    *,
    metadata: Any,
    base_config: Any,
    data_factory: DataFactory,
    model_kwargs: Optional[Dict[str, Any]],
    device: str | torch.device,
    metric_name: str,
    cuda_memory_fraction: float,
    keep_model: bool = False,
) -> Tuple[Dict[str, Any], Optional[RGCNModel]]:
    device_obj = _configure_worker(device, cuda_memory_fraction)
    config = _prepare_config(base_config, params)
    train_loader, val_loader, test_loader = data_factory(config)
    extra_kwargs = dict(model_kwargs or {})
    model = RGCNModel(metadata=metadata, config=config, device=device_obj, **extra_kwargs)
    optimizer = _build_optimizer(model, config)
    epochs = int(getattr(config.train, "epochs", 5))
    model.fit(
        train_loader,
        val_data=val_loader,
        epochs=epochs,
        optimizer=optimizer,
        device=device_obj,
        verbose=False,
    )

    row: Dict[str, Any] = {**params, "trial_index": index, "epochs_run": epochs}
    train_metrics = model.evaluate_loader(train_loader, device=device_obj)
    for metric, value in train_metrics.items():
        row[f"train_{metric}"] = value

    if val_loader is not None:
        val_metrics = model.evaluate_loader(val_loader, device=device_obj)
        for metric, value in val_metrics.items():
            row[f"val_{metric}"] = value
    else:
        val_metrics = None

    if test_loader is not None:
        test_metrics = model.evaluate_loader(test_loader, device=device_obj)
        for metric, value in test_metrics.items():
            row[f"test_{metric}"] = value

    if metric_name not in row:
        if val_metrics is None:
            raise RuntimeError(
                f"Metric '{metric_name}' was not produced. Ensure validation data is provided or change metric_name."
            )
        raise RuntimeError(f"Metric '{metric_name}' missing from evaluation results: {sorted(row)}")

    return row, (model if keep_model else None)


def run_fast_rgcn_grid_search(config: FastGridSearchConfig) -> GridSearchReport:
    if config.start_cuda_mps:
        _ensure_cuda_mps()

    param_grid = config.param_grid or _DEFAULT_PARAM_GRID
    parameter_grid = ParameterGrid(param_grid)
    if len(parameter_grid) == 0:
        raise ValueError("Parameter grid is empty; provide at least one hyper-parameter combination.")

    results: List[Dict[str, Any]] = []
    with tqdm_joblib(tqdm(total=len(parameter_grid), desc=f"Trials (n_jobs={config.n_jobs})")):
        parallel_results = Parallel(n_jobs=config.n_jobs, backend="loky")(
            delayed(_run_single_trial)(
                i,
                params,
                metadata=config.metadata,
                base_config=config.base_config,
                data_factory=config.data_factory,
                model_kwargs=config.model_kwargs,
                device=config.device,
                metric_name=config.metric_name,
                cuda_memory_fraction=config.cuda_memory_fraction,
                keep_model=False,
            )[0]
            for i, params in enumerate(parameter_grid)
        )
    results.extend(parallel_results)

    if not results:
        raise RuntimeError("Grid search did not evaluate any parameter combinations.")

    param_keys = sorted({k for grid in parameter_grid.param_grid for k in grid.keys()})
    results_df = pd.DataFrame(results)
    ascending = not config.greater_is_better
    results_df = results_df.sort_values(config.metric_name, ascending=ascending).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_params = {k: best_row[k] for k in param_keys if k in best_row}
    best_score = float(best_row[config.metric_name])

    # Re-train the best configuration to obtain the fitted model on the main process.
    _, best_model = _run_single_trial(
        index=int(best_row.get("trial_index", 0)),
        params=best_params,
        metadata=config.metadata,
        base_config=config.base_config,
        data_factory=config.data_factory,
        model_kwargs=config.model_kwargs,
        device=config.device,
        metric_name=config.metric_name,
        cuda_memory_fraction=config.cuda_memory_fraction,
        keep_model=True,
    )
    assert best_model is not None

    return GridSearchReport(
        results=results_df,
        best_params=best_params,
        best_score=best_score,
        best_model=best_model,
    )


__all__ = ["FastGridSearchConfig", "run_fast_rgcn_grid_search"]