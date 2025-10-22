from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

def set_target_column(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config
):
    tgt = config.data.target.name
    pos = set(config.data.target.positive_values)
    bin_name = getattr(config.data.target, "binarized_name", f"{tgt}_bin")
    # persist name back into config for later use
    config.data.target.binarized_name = bin_name

    df_train[bin_name] = df_train[tgt].isin(pos).astype(int)
    df_val[bin_name]   = df_val[tgt].isin(pos).astype(int)
    df_test[bin_name]  = df_test[tgt].isin(pos).astype(int)


def _concat_onehot(
    df: pd.DataFrame,
    cat_cols: List[str],
    ohe: OneHotEncoder,
    prefix: str = "oh__"
) -> pd.DataFrame:
    """Return df with one-hot columns added (original cat columns preserved)."""
    if not cat_cols:
        return df
    arr = ohe.transform(df[cat_cols])
    names = ohe.get_feature_names_out(cat_cols)
    # rename with explicit prefix to make selection easy later
    names = [f"{prefix}{n}" for n in names]
    ohe_df = pd.DataFrame(arr, index=df.index, columns=names)
    # cast numeric dtypes to float32 for torch
    for c in ohe_df.columns:
        ohe_df[c] = ohe_df[c].astype(np.float32)
    return pd.concat([df, ohe_df], axis=1)

def _collapse_rare_categories(df_train, df_val, df_test, cat_cols, unknown_label, min_freq: int):
    if min_freq is None or min_freq <= 1 or not cat_cols:
        return df_train, df_val, df_test

    for c in cat_cols:
        # Count on TRAIN only
        vc = df_train[c].astype(str).value_counts(dropna=False)
        rare_vals = set(vc[vc < min_freq].index.astype(str))

        if not rare_vals:
            continue

        # Replace rare on TRAIN with UNKNOWN
        df_train[c] = df_train[c].astype(str).where(~df_train[c].astype(str).isin(rare_vals), other=unknown_label)

        # Lock the allowed set to TRAIN post-collapse
        allowed = set(df_train[c].astype(str).unique())

        # Map VAL/TEST to UNKNOWN if not allowed
        df_val[c]  = df_val[c].astype(str).where(df_val[c].astype(str).isin(allowed),   other=unknown_label)
        df_test[c] = df_test[c].astype(str).where(df_test[c].astype(str).isin(allowed), other=unknown_label)

    return df_train, df_val, df_test

def preprocess_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Impute numeric & categorical, optional scaling, and ONE-HOT encode low-card categoricals.
    Keeps original categorical columns (for edges) and appends OHE columns (for encounter features).
    Returns: df_train, df_val, df_test, artifacts (scaler/ohe/feature_cols)
    """
    cfg = config.data

    # Work on copies
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    # Set binary target column
    set_target_column(df_train, df_val, df_test, config)

    num_cols = list(getattr(cfg.columns, "numeric", []) or [])
    cat_cols = list(getattr(cfg.columns, "categorical_low_card", []) or [])

    # ---------------------------
    # 1) Impute
    # ---------------------------
    num_imputer = SimpleImputer(strategy=cfg.preprocessing.numeric_imputer)
    if num_cols:
        df_train[num_cols] = num_imputer.fit_transform(df_train[num_cols])
        df_val[num_cols]   = num_imputer.transform(df_val[num_cols])
        df_test[num_cols]  = num_imputer.transform(df_test[num_cols])

    if cat_cols:
        SENTINELS = ["?", "None", None]
        for c in cat_cols:
            df_train[c] = df_train[c].replace(SENTINELS, np.nan)
            df_val[c]   = df_val[c].replace(SENTINELS, np.nan)
            df_test[c]  = df_test[c].replace(SENTINELS, np.nan)

        cat_imputer = SimpleImputer(
            strategy=cfg.preprocessing.categorical_imputer,
            fill_value=cfg.preprocessing.unknown_label,
        )
        df_train[cat_cols] = cat_imputer.fit_transform(df_train[cat_cols])
        df_val[cat_cols]   = cat_imputer.transform(df_val[cat_cols])
        df_test[cat_cols]  = cat_imputer.transform(df_test[cat_cols])

        # Unknown handling (keep original columns for edges, but normalize values)
        if getattr(cfg.preprocessing, "use_unknown_category", False):
            unk = cfg.preprocessing.unknown_label
            for c in cat_cols:
                train_vals = set(df_train[c].astype(str).unique())
                df_val[c]  = df_val[c].astype(str).where(df_val[c].astype(str).isin(train_vals), other=unk)
                df_test[c] = df_test[c].astype(str).where(df_test[c].astype(str).isin(train_vals), other=unk)

        # collapse rare categories based on config
        min_freq = getattr(cfg.preprocessing, "min_freq_for_category", 1)
        df_train, df_val, df_test = _collapse_rare_categories(
            df_train, df_val, df_test, cat_cols, cfg.preprocessing.unknown_label, min_freq
        )
    # ---------------------------
    # 2) One-Hot encode low-card categoricals (if requested)
    # ---------------------------
    artifacts: Dict[str, Any] = {"scaler": None, "ohe": None, "feature_cols": None}
    cat_handling = getattr(cfg.preprocessing, "categorical_handling", "onehot")
    ohe_feature_cols: List[str] = []
    if cat_cols and cat_handling == "onehot":
        # Fit on TRAIN only
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        ohe.fit(df_train[cat_cols])

        df_train = _concat_onehot(df_train, cat_cols, ohe, prefix="oh__")
        df_val   = _concat_onehot(df_val,   cat_cols, ohe, prefix="oh__")
        df_test  = _concat_onehot(df_test,  cat_cols, ohe, prefix="oh__")

        ohe_feature_cols = [f"oh__{n}" for n in ohe.get_feature_names_out(cat_cols)]
        artifacts["ohe"] = ohe
    elif cat_cols and cat_handling == "embedding":
        # Optional: leave as-is and let the model embed them by index.
        # If you pick this route, DO NOT include raw string cols in encounter.x.
        ohe_feature_cols = []  # none, because no OHE created
    else:
        ohe_feature_cols = []

    # ---------------------------
    # 3) Scale numeric features (optional)
    # ---------------------------
    scaler = None
    if getattr(cfg.preprocessing, "scaler", None):
        scaler = StandardScaler() if cfg.preprocessing.scaler == "standard" else RobustScaler()
        if num_cols:
            df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
            df_val[num_cols]   = scaler.transform(df_val[num_cols])
            df_test[num_cols]  = scaler.transform(df_test[num_cols])

    # Ensure numeric columns are float32 for torch
    for c in num_cols:
        df_train[c] = df_train[c].astype(np.float32)
        df_val[c]   = df_val[c].astype(np.float32)
        df_test[c]  = df_test[c].astype(np.float32)

    # Final encounter feature list = numeric + OHE categorical (NOT the raw string columns)
    encounter_feature_cols = num_cols + ohe_feature_cols
    # Persist for the builder
    setattr(cfg.columns, "encounter_features", encounter_feature_cols)

    artifacts["scaler"] = scaler
    artifacts["feature_cols"] = encounter_feature_cols
    return df_train, df_val, df_test, artifacts
