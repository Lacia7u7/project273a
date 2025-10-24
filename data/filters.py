import pandas as pd

from data import csv_mapper


def apply_filters(df: pd.DataFrame, config) -> pd.DataFrame:
    """Apply filtering criteria (LOS, discharge disposition, first encounter, etc.) to DataFrame"""
    cfg = config.data
    # Filter by length of staty if specified
    if cfg.filters.min_los is not None:
        df = df[df[csv_mapper.time_in_hospital] >= cfg.filters.min_los]
    if cfg.filters.max_los is not None:
        df = df[df[csv_mapper.time_in_hospital] <= cfg.filters.max_los]

    # Exclude certain discharge dispositions (e.g. hospice, expired)
    if cfg.filters.exclude_discharge_to_ids:
        dd_col = cfg.columns.discharge_disposition_col
        if dd_col in df.columns:
            df = df[~df[dd_col].isin(cfg.filters.exclude_discharge_to_ids)]

    #if first encounter per patient only
    if cfg.filters.first_encounter_per_patient:
        # sort by encounter_id (assuming it correlates with time) and drop duplicates keeping first
        pid_col = cfg.identifier_cols.patient_id
        eid_col = cfg.identifier_cols.encounter_id
        df = df.sort_values(eid_col).groupby(pid_col).head(1)
    return df.reset_index(drop=True)