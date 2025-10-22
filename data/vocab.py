# data/vocab.py
import re
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd


def _canonicalize_icd(code: str, *, truncate_to_3: bool) -> Optional[str]:
    """
    Canonicalize ICD-9/ICD-10-ish codes:
      - strip whitespace
      - uppercase
      - remove dots
      - optionally truncate to 3-digit category (ICD-9 style):
          * 'E' or 'V' prefixed codes -> letter + 3 digits (e.g., E1234 -> E123)
          * others -> first 3 characters
    Returns None for empty/missing.
    """
    if code is None:
        return None
    s = str(code).strip()
    if not s or s.lower() == "nan":
        return None

    s = s.upper().replace(".", "")
    if not s:
        return None

    if truncate_to_3:
        if s[0] in ("E", "V"):
            return s[:4] if len(s) >= 4 else s
        return s[:3] if len(s) >= 3 else s

    return s


def _build_vocab_from_values(
    values: Iterable[str],
    *,
    add_unknown: bool,
    unknown_label: str,
    sort_values: bool = True,
) -> Dict[str, int]:
    unique = set(v for v in values if v is not None and str(v) != "nan")
    items = sorted(unique) if sort_values else list(unique)
    if add_unknown and unknown_label not in items:
        items = [unknown_label] + items
    # index is deterministic, UNKNOWN sits at 0 if present
    return {val: idx for idx, val in enumerate(items)}


def build_vocab(
    df: pd.DataFrame,
    col: str,
    *,
    add_unknown: bool = True,
    unknown_label: str = "UNKNOWN",
) -> Dict[str, int]:
    """Build a vocab from a single column."""
    values = df[col].astype(str).tolist()
    return _build_vocab_from_values(values, add_unknown=add_unknown, unknown_label=unknown_label)


def build_vocab_from_columns(
    df: pd.DataFrame,
    columns: List[str],
    *,
    transform=None,
    add_unknown: bool = True,
    unknown_label: str = "UNKNOWN",
) -> Dict[str, int]:
    """Build a vocab from multiple columns (e.g., diag_1, diag_2, diag_3)."""
    collected = []
    for col in columns:
        if col in df.columns:
            for v in df[col].dropna():
                vv = str(v)
                if transform:
                    vv = transform(vv)
                if vv is not None:
                    collected.append(vv)
    return _build_vocab_from_values(collected, add_unknown=add_unknown, unknown_label=unknown_label)


def make_vocabs(
    df_train_proc: pd.DataFrame,
    config
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, str]]]:
    """
    Build all vocabularies needed by the pipeline from TRAIN ONLY to avoid leakage.
    Returns:
      vocabs: dict of {node_type -> {token -> index}}
      mappings: dict of auxiliary string-to-string mappings (e.g., ICD -> ICD_GROUP token)
    """
    cfg = config.data
    use_unknown = getattr(cfg.preprocessing, "use_unknown_category", False)
    unknown_label = getattr(cfg.preprocessing, "unknown_label", "UNKNOWN")

    vocabs: Dict[str, Dict[str, int]] = {}
    mappings: Dict[str, Dict[str, str]] = {}

    # --- ICD codes ---
    icd_cols = list(getattr(cfg.columns, "icd_cols", []))
    truncate_flag = bool(getattr(cfg.preprocessing, "truncate_icd_to_3_digits", False))

    if icd_cols:
        # "icd" vocab respects the truncate flag (your earlier call intended this)
        vocabs["icd"] = build_vocab_from_columns(
            df_train_proc,
            icd_cols,
            transform=lambda x: _canonicalize_icd(x, truncate_to_3=truncate_flag),
            add_unknown=use_unknown,
            unknown_label=unknown_label,
        )

        # A grouped ICD vocab (always 3-digit category). Useful even if "icd" kept full codes.
        vocabs["icd_group"] = build_vocab_from_columns(
            df_train_proc,
            icd_cols,
            transform=lambda x: _canonicalize_icd(x, truncate_to_3=True),
            add_unknown=use_unknown,
            unknown_label=unknown_label,
        )

        # Provide a helper mapping from ICD token -> ICD_GROUP token strings
        icd_to_group = {}
        for token in vocabs["icd"].keys():
            if token == unknown_label and use_unknown:
                icd_to_group[token] = unknown_label
            else:
                icd_to_group[token] = _canonicalize_icd(token, truncate_to_3=True)
        mappings["icd_to_icd_group"] = icd_to_group
    else:
        vocabs["icd"] = {unknown_label: 0} if use_unknown else {}
        vocabs["icd_group"] = {unknown_label: 0} if use_unknown else {}
        mappings["icd_to_icd_group"] = {}

    # --- Drug vocab (from feature names) ---
    # If drug features are columns like "metformin", "insulin", etc., use the column list itself.
    drug_names = list(getattr(cfg.columns, "drug_cols", []))
    if drug_names:
        # Usually you DON'T want UNKNOWN here because the set is fixed by schema.
        vocabs["drug"] = {name: i for i, name in enumerate(sorted(drug_names))}
    else:
        vocabs["drug"] = {unknown_label: 0} if use_unknown else {}

    # Placeholder for drug class; keep empty unless you load a mapping {drug -> class}.
    vocabs["drug_class"] = {unknown_label: 0} if use_unknown else {}

    # --- Other categorical node types from TRAIN ---
    for attr_name, node_type in [
        ("admission_type_col", "admission_type"),
        ("discharge_disposition_col", "discharge_disposition"),
        ("admission_source_col", "admission_source"),
        ("specialty_col", "specialty"),
    ]:
        col_name = getattr(cfg.columns, attr_name, None)
        if col_name and col_name in df_train_proc.columns:
            values = df_train_proc[col_name].astype(str).tolist()
            vocabs[node_type] = _build_vocab_from_values(
                values, add_unknown=use_unknown, unknown_label=unknown_label
            )
        else:
            vocabs[node_type] = {unknown_label: 0} if use_unknown else {}

    return vocabs, mappings
