import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, List, Dict, Any

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
)

# Try to import StratifiedGroupKFold (scikit-learn >= 1.1). We'll gracefully fall back if unavailable.
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
    _HAS_SGK = True
except Exception:
    _HAS_SGK = False


# ----------------------------- helpers -----------------------------
def _get_groups(df: pd.DataFrame, config) -> Optional[np.ndarray]:
    """Return group labels array or None based on config."""
    cfg_cols = getattr(config.data, "columns", None)
    pid_field = getattr(config.data.identifier_cols, "patient_id", None)
    group_by = getattr(config.data.splits, "group_by", None)

    if group_by == "patient" and pid_field in df.columns:
        return df[pid_field].to_numpy()
    if group_by == "hospital" and cfg_cols and getattr(cfg_cols, "hospital_col", None):
        hcol = cfg_cols.hospital_col
        if hcol in df.columns:
            return df[hcol].to_numpy()
    return None


def _target_series(df: pd.DataFrame, config) -> pd.Series:
    """Return target as a pandas Series."""
    tgt_name = config.data.target.name
    if tgt_name not in df.columns:
        raise KeyError(f"Target column '{tgt_name}' not found in dataframe.")
    return df[tgt_name]


def _can_stratify_labels(y: np.ndarray, k: int) -> bool:
    """Heuristic: stratification is feasible if each class has >= k members."""
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        return False
    return counts.min() >= k


def _group_labels_for_stratification(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Reduce sample-level labels to one label per group for stratified splitting over groups.
    Uses the group's majority class. For continuous targets this won't help—callers should guard.
    """
    df = pd.DataFrame({"group": groups, "y": y})
    maj = (
        df.groupby("group")["y"]
        .apply(lambda s: s.value_counts().idxmax())
        .reset_index(name="y_group")
    )
    _, inv = np.unique(groups, return_inverse=True)
    group_to_label = dict(zip(maj["group"].to_numpy(), maj["y_group"].to_numpy()))
    group_labels = np.array([group_to_label[g] for g in np.unique(groups)])
    return group_labels


def _three_way_from_cv_folds(
    folds: List[Tuple[np.ndarray, np.ndarray]], n_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (train, val, test) from at least 2 CV folds (each a (train_idx, test_idx) pair).
    We take val = folds[0].test, test = folds[1].test, and train = complement of (val ∪ test).
    """
    if len(folds) < 2:
        all_idx = np.arange(n_samples)
        return all_idx, all_idx, all_idx

    all_idx = np.arange(n_samples)
    val_idx = folds[0][1]
    test_idx = folds[1][1]
    mask_holdout = np.zeros(n_samples, dtype=bool)
    mask_holdout[val_idx] = True
    mask_holdout[test_idx] = True
    train_idx = np.where(~mask_holdout)[0]
    return train_idx, val_idx, test_idx


def _two_stage_plain_or_stratified(
    X_len: int,
    y: Optional[np.ndarray],
    stratify: bool,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-stage split without groups:
      1) train_val vs test
      2) train vs val (within train_val)
    """
    all_idx = np.arange(X_len)
    if stratify and y is not None and _can_stratify_labels(y, 2):
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(sss1.split(all_idx, y))
        y_trainval = y[trainval_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(sss2.split(trainval_idx, y_trainval))
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]
    else:
        if stratify:
            warnings.warn(
                "Requested stratification but target is unsuitable; falling back to non-stratified splits."
            )
        rs1 = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(rs1.split(all_idx))
        rs2 = ShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(rs2.split(trainval_idx))
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _two_stage_group_or_stratgroup(
    groups: np.ndarray,
    y: Optional[np.ndarray],
    stratify: bool,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-stage split with group disjointness:
      1) groups -> train_val_groups vs test_groups
      2) within train_val_groups -> train_groups vs val_groups
    If stratify=True and classification is feasible, perform stratification at the GROUP level.
    """
    uniq_groups, group_indices = np.unique(groups, return_inverse=True)
    n_groups = len(uniq_groups)

    if n_groups < 2:
        warnings.warn("Not enough groups to split; returning all indices for train/val/test.")
        idx = np.arange(len(groups))
        return idx, idx, idx

    if stratify and y is not None:
        y_group = _group_labels_for_stratification(groups, y)
        vals, counts = np.unique(y_group, return_counts=True)
        if len(vals) >= 2 and counts.min() >= 2:
            sss_g1 = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            g_trainval_idx, g_test_idx = next(
                sss_g1.split(np.arange(n_groups), y_group)
            )
            sss_g2 = StratifiedShuffleSplit(
                n_splits=1, test_size=val_size, random_state=seed
            )
            y_group_trainval = y_group[g_trainval_idx]
            g_train_idx, g_val_idx = next(
                sss_g2.split(g_trainval_idx, y_group_trainval)
            )
            g_train = uniq_groups[g_trainval_idx[g_train_idx]]
            g_val = uniq_groups[g_trainval_idx[g_val_idx]]
            g_test = uniq_groups[g_test_idx]
        else:
            warnings.warn(
                "Group-level stratification not feasible (insufficient groups per class); "
                "falling back to non-stratified GroupShuffleSplit."
            )
            stratify = False  # fall through to non-stratified below

    if not stratify:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_all = np.arange(len(groups))
        trainval_idx, test_idx = next(gss1.split(idx_all, groups=groups))

        groups_trainval = groups[trainval_idx]
        uniq_tv_groups = np.unique(groups_trainval)
        if len(uniq_tv_groups) < 2:
            warnings.warn("Not enough train/val groups; placing all train_val into train.")
            return trainval_idx, np.array([], dtype=int), test_idx

        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tv_idx_dummy = np.arange(len(trainval_idx))
        train_idx_rel, val_idx_rel = next(
            gss2.split(tv_idx_dummy, groups=groups_trainval)
        )
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    mask_train = np.isin(groups, g_train)
    mask_val = np.isin(groups, g_val)
    mask_test = np.isin(groups, g_test)
    train_idx = np.where(mask_train)[0]
    val_idx = np.where(mask_val)[0]
    test_idx = np.where(mask_test)[0]
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# --------------------------- public API ----------------------------
def create_splits(df: pd.DataFrame, config):
    """
    Split dataframe into (train_idx, val_idx, test_idx) with no overlap.
    Supports:
      - plain KFold / two-stage shuffle
      - StratifiedKFold / stratified two-stage
      - GroupKFold / group two-stage
      - StratifiedGroupKFold (if available) / stratified-group two-stage

    Config fields expected:
      config.data.splits.group_by: None | "patient" | "hospital"
      config.data.splits.stratify_by_target: bool
      config.data.splits.n_splits: int
      config.data.splits.seed: int
      (optional) config.data.splits.test_size: float in (0,1)
      (optional) config.data.splits.val_size: float in (0,1)
      config.data.identifier_cols.patient_id: str (if group_by == "patient")
      config.data.columns.hospital_col: str (if group_by == "hospital")
      config.data.target.name: str
    """
    cfg = config.data.splits
    n_splits = int(getattr(cfg, "n_splits", 5))
    seed = int(getattr(cfg, "seed", 42))
    stratify = bool(getattr(cfg, "stratify_by_target", False))
    test_size = float(getattr(cfg, "test_size", 0.2))
    val_size = float(getattr(cfg, "val_size", 0.2))

    y = _target_series(df, config).to_numpy()
    groups = _get_groups(df, config)
    n = len(df)

    # --------- grouped path ----------
    if groups is not None:
        uniq_groups = np.unique(groups)
        n_groups = len(uniq_groups)

        if n_splits >= 3 and n_groups >= n_splits:
            folds: List[Tuple[np.ndarray, np.ndarray]] = []
            if stratify:
                if _HAS_SGK:
                    try:
                        sgkf = StratifiedGroupKFold(n_splits=n_splits)
                        for tr, te in sgkf.split(np.zeros(n), y, groups):
                            folds.append((np.array(tr), np.array(te)))
                    except Exception:
                        warnings.warn(
                            "StratifiedGroupKFold failed; falling back to group-level majority-label "
                            "stratification over unique groups."
                        )
                        y_group = _group_labels_for_stratification(groups, y)
                        if _can_stratify_labels(y_group, n_splits):
                            skf_groups = StratifiedKFold(
                                n_splits=n_splits, shuffle=True, random_state=seed
                            )
                            group_indices = np.arange(n_groups)
                            uniq_groups, inv = np.unique(groups, return_inverse=True)
                            for g_tr, g_te in skf_groups.split(group_indices, y_group):
                                tr_mask = np.isin(inv, g_tr)
                                te_mask = np.isin(inv, g_te)
                                folds.append((np.where(tr_mask)[0], np.where(te_mask)[0]))
                        else:
                            warnings.warn(
                                "Cannot stratify groups across CV folds; using plain GroupKFold."
                            )
                            gkf = GroupKFold(n_splits=n_splits)
                            folds = [(np.array(tr), np.array(te))
                                     for tr, te in gkf.split(df, y, groups)]
                else:
                    y_group = _group_labels_for_stratification(groups, y)
                    if _can_stratify_labels(y_group, n_splits):
                        skf_groups = StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=seed
                        )
                        group_indices = np.arange(n_groups)
                        uniq_groups, inv = np.unique(groups, return_inverse=True)
                        for g_tr, g_te in skf_groups.split(group_indices, y_group):
                            tr_mask = np.isin(inv, g_tr)
                            te_mask = np.isin(inv, g_te)
                            folds.append((np.where(tr_mask)[0], np.where(te_mask)[0]))
                    else:
                        warnings.warn(
                            "Cannot stratify groups across CV folds; using plain GroupKFold."
                        )
                        gkf = GroupKFold(n_splits=n_splits)
                        folds = [(np.array(tr), np.array(te))
                                 for tr, te in gkf.split(df, y, groups)]
            else:
                gkf = GroupKFold(n_splits=n_splits)
                folds = [(np.array(tr), np.array(te))
                         for tr, te in gkf.split(df, y, groups)]

            train_idx, val_idx, test_idx = _three_way_from_cv_folds(folds, n)

        else:
            train_idx, val_idx, test_idx = _two_stage_group_or_stratgroup(
                groups=groups,
                y=y,
                stratify=stratify,
                seed=seed,
                test_size=test_size,
                val_size=val_size,
            )

    # --------- ungrouped path ----------
    else:
        if n_splits >= 3:
            if stratify and _can_stratify_labels(y, n_splits):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                folds = [(np.array(tr), np.array(te)) for tr, te in skf.split(np.zeros(n), y)]
            else:
                if stratify and not _can_stratify_labels(y, n_splits):
                    warnings.warn(
                        "Requested stratified CV but each class must have at least n_splits samples; "
                        "falling back to plain KFold."
                    )
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                folds = [(np.array(tr), np.array(te)) for tr, te in kf.split(np.zeros(n))]
            train_idx, val_idx, test_idx = _three_way_from_cv_folds(folds, n)
        else:
            train_idx, val_idx, test_idx = _two_stage_plain_or_stratified(
                X_len=n,
                y=y,
                stratify=stratify,
                seed=seed,
                test_size=test_size,
                val_size=val_size,
            )

    # Final safety: ensure disjointness and integer dtype
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    assert set(train_idx).isdisjoint(val_idx), "Train/Val overlap detected."
    assert set(train_idx).isdisjoint(test_idx), "Train/Test overlap detected."
    assert set(val_idx).isdisjoint(test_idx), "Val/Test overlap detected."

    return train_idx, val_idx, test_idx


def check_no_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config,
):
    """
    Ensure no patient or group leakage between splits.
    Checks:
      - index-level disjointness
      - patient-level disjointness (if column available)
      - group-level disjointness (if group_by specified)
    """
    # Index-level checks
    assert set(train_idx).isdisjoint(val_idx), "Index leakage: train/val overlap."
    assert set(train_idx).isdisjoint(test_idx), "Index leakage: train/test overlap."
    assert set(val_idx).isdisjoint(test_idx), "Index leakage: val/test overlap."

    # Patient-level (if present)
    pid_col = getattr(config.data.identifier_cols, "patient_id", None)
    if pid_col and pid_col in df.columns:
        train_pids = set(df.iloc[train_idx][pid_col])
        val_pids = set(df.iloc[val_idx][pid_col])
        test_pids = set(df.iloc[test_idx][pid_col])
        assert train_pids.isdisjoint(val_pids), "Patient leakage between train and val."
        assert train_pids.isdisjoint(test_pids), "Patient leakage between train and test."
        assert val_pids.isdisjoint(test_pids), "Patient leakage between val and test."

    # Group-level (if configured)
    groups = _get_groups(df, config)
    if groups is not None:
        g_train = set(groups[train_idx])
        g_val = set(groups[val_idx])
        g_test = set(groups[test_idx])
        assert g_train.isdisjoint(g_val), "Group leakage between train and val."
        assert g_train.isdisjoint(g_test), "Group leakage between train and test."
        assert g_val.isdisjoint(g_test), "Group leakage between val and test."


# ---------------------- NEW: distribution checker ----------------------
def check_target_distribution(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config,
    *,
    task: str = "auto",        # "classification" | "regression" | "auto"
    tol: float = 0.05,         # classification: max allowed absolute diff in proportions (5 percentage points)
    mean_z_tol: float = 0.25,  # regression: |mean - global_mean| <= 0.25 * global_std
    std_ratio_bounds: Tuple[float, float] = (0.5, 1.5),  # regression: split_std in [0.5, 1.5] * global_std
    q_tol: float = 0.20,       # regression: each of Q25/Q50/Q75 within 0.20 * global_IQR
    strict: bool = True,       # raise AssertionError when not OK
) -> Dict[str, Any]:
    """
    Validate that the target's distribution in train/val/test resembles the overall dataset.

    For classification:
      - Computes class proportions globally and per split.
      - Fails if any class proportion diff exceeds `tol`.

    For regression:
      - Compares mean/std and quartiles to the global distribution.
      - Fails if |mean - global_mean| > mean_z_tol * global_std,
              or std outside `std_ratio_bounds` * global_std,
              or any of Q25/Q50/Q75 deviates by more than q_tol * global_IQR.

    Returns a report dict with metrics; raises AssertionError if `strict=True` and checks fail.
    """
    y = _target_series(df, config).to_numpy()
    splits = {
        "train": np.asarray(train_idx, dtype=int),
        "val": np.asarray(val_idx, dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }

    def _is_classification_target(y_arr: np.ndarray) -> bool:
        # Heuristics: non-numeric / boolean → classification; or few unique values.
        if y_arr.dtype.kind in ("b", "O", "U", "S"):
            return True
        unique_vals = np.unique(y_arr)
        # Consider integer targets with few unique values as classification
        if y_arr.dtype.kind in ("i", "u") and len(unique_vals) <= 20:
            return True
        # Float with very few unique values often indicates encoded classes
        if y_arr.dtype.kind == "f" and len(unique_vals) <= 10:
            return True
        return False

    report: Dict[str, Any] = {"task": None}

    if task not in {"classification", "regression", "auto"}:
        raise ValueError("task must be 'classification', 'regression', or 'auto'")

    inferred_is_clf = _is_classification_target(y)
    do_clf = (task == "classification") or (task == "auto" and inferred_is_clf)
    report["task"] = "classification" if do_clf else "regression"

    # ---------- Classification path ----------
    if do_clf:
        global_vals, global_counts = np.unique(y, return_counts=True)
        n_total = len(y)
        global_props = {c: cnt / n_total for c, cnt in zip(global_vals, global_counts)}

        split_props: Dict[str, Dict[Any, float]] = {}
        split_counts: Dict[str, Dict[Any, int]] = {}
        max_abs_diff_by_split: Dict[str, float] = {}
        failed_reasons: List[str] = []

        for split_name, idx in splits.items():
            n_split = len(idx)
            if n_split == 0:
                split_props[split_name] = {}
                split_counts[split_name] = {}
                max_abs_diff_by_split[split_name] = 0.0
                continue

            vals, counts = np.unique(y[idx], return_counts=True)
            props = {c: cnt / n_split for c, cnt in zip(vals, counts)}
            split_props[split_name] = props
            split_counts[split_name] = {c: int(cnt) for c, cnt in zip(vals, counts)}

            # Evaluate absolute proportion deltas for the union of classes
            all_classes = set(global_props.keys())
            deltas = []
            for c in all_classes:
                p_global = global_props.get(c, 0.0)
                p_split = props.get(c, 0.0)
                deltas.append(abs(p_split - p_global))
            max_abs = float(np.max(deltas)) if deltas else 0.0
            max_abs_diff_by_split[split_name] = max_abs
            if max_abs > tol:
                failed_reasons.append(
                    f"{split_name}: max abs proportion delta {max_abs:.3f} exceeds tol {tol:.3f}"
                )

        ok = len(failed_reasons) == 0
        report.update(
            dict(
                global_props=global_props,
                split_props=split_props,
                split_counts=split_counts,
                max_abs_diff_by_split=max_abs_diff_by_split,
                ok=ok,
                reasons=failed_reasons,
            )
        )
        if strict and not ok:
            raise AssertionError(
                "Target distribution check failed (classification): "
                + "; ".join(failed_reasons)
            )
        return report

    # ---------- Regression path ----------
    eps = 1e-12
    y_global = y
    g_mean = float(np.mean(y_global))
    g_std = float(np.std(y_global, ddof=1)) if len(y_global) > 1 else 0.0
    g_q25, g_q50, g_q75 = [float(q) for q in np.quantile(y_global, [0.25, 0.5, 0.75])] if len(y_global) > 0 else (0.0, 0.0, 0.0)
    g_iqr = max(g_q75 - g_q25, eps)

    split_stats: Dict[str, Dict[str, float]] = {}
    failed_reasons: List[str] = []

    for split_name, idx in splits.items():
        if len(idx) == 0:
            split_stats[split_name] = dict(
                n=0, mean=np.nan, std=np.nan, q25=np.nan, q50=np.nan, q75=np.nan,
                mean_z=np.nan, std_ratio=np.nan, q25_delta_iqr=np.nan,
                q50_delta_iqr=np.nan, q75_delta_iqr=np.nan,
            )
            continue

        yi = y[idx]
        s_mean = float(np.mean(yi))
        s_std = float(np.std(yi, ddof=1)) if len(yi) > 1 else 0.0
        s_q25, s_q50, s_q75 = [float(q) for q in np.quantile(yi, [0.25, 0.5, 0.75])]
        mean_z = abs(s_mean - g_mean) / max(g_std, eps)
        std_ratio = s_std / max(g_std, eps) if g_std > 0 else np.nan
        q25_delta_iqr = abs(s_q25 - g_q25) / g_iqr
        q50_delta_iqr = abs(s_q50 - g_q50) / g_iqr
        q75_delta_iqr = abs(s_q75 - g_q75) / g_iqr

        split_stats[split_name] = dict(
            n=int(len(yi)),
            mean=s_mean,
            std=s_std,
            q25=s_q25,
            q50=s_q50,
            q75=s_q75,
            mean_z=mean_z,
            std_ratio=std_ratio,
            q25_delta_iqr=q25_delta_iqr,
            q50_delta_iqr=q50_delta_iqr,
            q75_delta_iqr=q75_delta_iqr,
        )

        # Evaluate thresholds
        if mean_z > mean_z_tol:
            failed_reasons.append(
                f"{split_name}: |mean - global_mean| > {mean_z_tol} * global_std (z={mean_z:.3f})"
            )
        if g_std > 0 and not (std_ratio_bounds[0] <= std_ratio <= std_ratio_bounds[1]):
            failed_reasons.append(
                f"{split_name}: std_ratio {std_ratio:.3f} outside bounds {std_ratio_bounds}"
            )
        if max(q25_delta_iqr, q50_delta_iqr, q75_delta_iqr) > q_tol:
            failed_reasons.append(
                f"{split_name}: quantile deltas exceed {q_tol} * IQR "
                f"(q25={q25_delta_iqr:.3f}, q50={q50_delta_iqr:.3f}, q75={q75_delta_iqr:.3f})"
            )

    ok = len(failed_reasons) == 0
    report.update(
        dict(
            global_stats=dict(
                mean=g_mean, std=g_std, q25=g_q25, q50=g_q50, q75=g_q75, iqr=g_iqr
            ),
            split_stats=split_stats,
            thresholds=dict(
                mean_z_tol=mean_z_tol, std_ratio_bounds=std_ratio_bounds, q_tol=q_tol
            ),
            ok=ok,
            reasons=failed_reasons,
        )
    )
    if strict and not ok:
        raise AssertionError("Target distribution check failed (regression): " + "; ".join(failed_reasons))
    return report
