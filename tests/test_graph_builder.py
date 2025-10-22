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

def test_edge_counts_and_reverse():
    cfg = get_dummy_config()
    # dummy data: 2 encounters, with diag and drug
    data = pd.DataFrame({
        'encounter_id': [1, 2],
        'patient_id': [100, 101],
        'diag_1': ['250.01', '250.02'],
        'drugA': ['Up', 'No'],
        'target': ['1', '0']
    })
    # build vocabs
    vocabs = {}
    vocabs['icd'] = vocab.build_entity_vocab_from_df(data, cfg.data.columns.icd_cols, truncate_icd=cfg.data.preprocessing.truncate_icd_to_3_digits)
    vocabs['icd_group'] = {mappings.map_icd_to_group(code): i for i, code in enumerate(vocabs['icd'].keys())}
    vocabs['drug'] = {drug: i for i, drug in enumerate(cfg.data.columns.drug_cols)}
    vocabs['drug_class'] = {mappings.map_drug_to_class(drug): i for i, drug in enumerate(vocabs['drug'].keys())}
    vocabs['encounter'] = {}  # not used explicitly
    data_graph = builder.build_heterodata(data, vocabs, cfg, include_target=True)
    # Check that reverse edges count equals forward edges count for defined relations
    for (src, rel, dst) in list(data_graph.edge_index_dict.keys()):
        if rel.startswith('rev_'):
            fwd_rel = rel.replace('rev_', '')
            # get corresponding forward edge index
            fwd_key = (dst, fwd_rel, src)
            if fwd_key in data_graph.edge_index_dict:
                num_fwd = data_graph.edge_index_dict[fwd_key].shape[1]
                num_rev = data_graph.edge_index_dict[(src, rel, dst)].shape[1]
                assert num_fwd == num_rev
