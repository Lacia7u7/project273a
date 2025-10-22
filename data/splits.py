import numpy as np
import pandas as pd
from more_itertools.more import consecutive_groups
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold


def create_splits(df: pd.DataFrame, config):
    """Split dataframe into train/val/tests sets based on group and stratification settings."""
    cfg = config.data.splits
    groups = None
    if cfg.group_by == "patient":
        groups = df[config.data.identifier_cols.patient_id]
    elif cfg.group_by == "hospital":
        groups = df[config.data.columns.hospital_col].values if config.data.columns.hospital_col else None

    # default to stratified if no grouping specified
    splits = []
    if groups is None:
        if cfg.stratify_by_target:
            skf =  StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
            splits = list(skf.split(df, df[config.data.target.name]))
        else:
            # simple random split via permutation
            kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
            for train_idx, test_idx in kf.split(np.arange(len(df))):
                splits.append((train_idx, test_idx))
    else:
        gkf = GroupKFold(n_splits=cfg.n_splits)
        splits = list(gkf.split(df, df[config.data.target.name], groups))

    if len(splits) >= 2:
        train_idx, val_idx = splits[0]
        if len(splits) > 1:
            _, test_idx = splits[1]
        else:
            test_idx = val_idx
    else:
        # no split performed
        train_idx = val_idx = test_idx = np.arange(len(df))

    return train_idx, val_idx, test_idx

def check_no_leakage(df: pd.DataFrame, train_idx, val_idx, test_idx, config):
    """Ensure no patient or group leakage between splits"""
    pid_col = config.data.identifier_cols.patient_id
    train_pids = set(df.iloc[train_idx][pid_col])
    val_pids = set(df.iloc[val_idx][pid_col])
    test_pids = set(df.iloc[test_idx][pid_col])
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)
