import pandas as pd
from types import SimpleNamespace

from data import mappings, vocab
from graph import builder


def get_dummy_config():
    # Create a simple config-like object with needed fields
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace()
    cfg.data.columns = SimpleNamespace(numeric=[], categorical_low_card=[], icd_cols=['diag_1'], drug_cols=['drugA'], hospital_col=None, specialty_col=None, admission_source_col=None, admission_type_col=None, discharge_disposition_col=None)
    cfg.data.target = SimpleNamespace(name='target', positive_values=['1'])
    cfg.data.identifier_cols = SimpleNamespace(encounter_id='encounter_id', patient_id='patient_id')
    cfg.data.preprocessing = SimpleNamespace(truncate_icd_to_3_digits=True, unknown_label="UNKNOWN")
    cfg.graph = SimpleNamespace(node_types_enabled={'encounter': True, 'icd': True, 'icd_group': True, 'drug': True, 'drug_class': True}, edge_types_enabled={'encounter__has_icd__icd': True, 'icd__is_a__icd_group': True, 'encounter__has_drug__drug': True, 'drug__belongs_to__drug_class': True, 'reverse_edges': True}, edge_featureing={'has_drug': {'relation_subtypes_by_status': False, 'edge_attr_status': False}})
    return cfg
