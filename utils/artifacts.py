# --- Add these imports near the top of your script ---
import os, json, time, datetime
from pathlib import Path
import torch

# -------------------- helpers: save/load best artifact --------------------
def _config_to_dict(config):
    """Best-effort conversion of arbitrary config objects to a JSON-serializable dict."""
    # OmegaConf (Hydra) first
    try:
        from omegaconf import OmegaConf  # type: ignore
        return OmegaConf.to_container(config, resolve=True)
    except Exception:
        pass

    # Dataclasses
    try:
        import dataclasses
        if dataclasses.is_dataclass(config):
            return dataclasses.asdict(config)
    except Exception:
        pass

    # Generic object fallback
    def _recurse(x):
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (list, tuple, set)):
            return [_recurse(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _recurse(v) for k, v in x.items()}
        # objects: try __dict__, else repr
        return _recurse(getattr(x, "__dict__", repr(x)))
    return _recurse(config)


def save_best_artifact(best_state, config, *, artifacts_dir="artifacts", run_id=None):
    """
    Save best_state together with a (serializable) config snapshot.
    Returns the path to the saved .pt file.
    """
    if best_state is None:
        raise ValueError("best_state is None; nothing to save.")

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Choose run_id from config if available, else timestamp
    if run_id is None:
        run_id = getattr(config, "run_name", None) or getattr(getattr(config, "train", object()), "run_name", None)
        if not run_id:
            run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "timestamp": time.time(),
        "framework": "torch",
        "torch_version": torch.__version__,
        "config": _config_to_dict(config),
        "best_state": best_state,  # includes model SD, epoch, val metric
    }

    pt_path = run_dir / "best_artifact.pt"
    torch.save(payload, pt_path)

    # Also drop a human-readable config.json
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload["config"], f, indent=2, default=str)

    # Maintain a "latest" pointer for convenience (symlink or text file fallback)
    latest_ptr = artifacts_dir / "latest"
    try:
        if latest_ptr.exists() or latest_ptr.is_symlink():
            latest_ptr.unlink()
        latest_ptr.symlink_to(run_dir, target_is_directory=True)
    except Exception:
        # Filesystems without symlink support: write a pointer file
        with open(artifacts_dir / "LATEST", "w") as f:
            f.write(str(run_dir.resolve()))

    return str(pt_path)


def load_best_artifact(artifacts_dir="artifacts", run_id="latest", map_location="cpu"):
    """
    Load the saved artifact dict. If run_id='latest', resolves via symlink/LATEST file
    or picks the most recently modified run directory.
    """
    base = Path(artifacts_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_id == "latest":
        # Try symlink
        latest = base / "latest"
        if latest.exists():
            run_dir = latest.resolve()
        else:
            # Try pointer file
            latest_txt = base / "LATEST"
            if latest_txt.exists():
                run_dir = Path(latest_txt.read_text().strip())
            else:
                # Fallback to newest dir
                candidates = [p for p in base.iterdir() if p.is_dir()]
                if not candidates:
                    raise FileNotFoundError(f"No runs found under {artifacts_dir}")
                run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        run_dir = base / run_id

    artifact_path = run_dir / "best_artifact.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return torch.load(artifact_path, map_location=map_location)
# ------------------------------------------------------------------------
