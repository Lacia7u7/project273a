import pandas as pd

from data import splits


def test_group_split_no_leakage():
    # Create dummy data for two patients, multiple encounters
    df = pd.DataFrame({
        'patient_nbr': [1,1,2,2],
        'value': [10,20,30,40],
        'readmitted': ['<30','NO','<30','NO']
    })
    # Config stub
    class Conf: pass
    config = Conf()
    config.data = Conf()
    config.data.identifier_cols = Conf()
    config.data.identifier_cols.patient_id = 'patient_nbr'
    config.data.target = Conf()
    config.data.target.name = 'readmitted'
    config.data.splits = Conf()
    config.data.splits.group_by = 'patient'
    config.data.splits.n_splits = 2
    config.data.splits.seed = 42
    config.data.splits.stratify_by_target = False
    train_idx, val_idx, test_idx = splits.create_splits(df, config)
    # Check no patient overlap
    splits.check_no_leakage(df, train_idx, val_idx, test_idx, config)