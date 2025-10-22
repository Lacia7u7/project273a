import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

from data import mappings


def build_heterodata(df: "pd.DataFrame", vocabs: dict, config, include_target: bool = False):
    data = HeteroData()
    cfg = config
    # Assign encounter nodes (one per row)
    feat_cols = getattr(cfg.data.columns, "encounter_features", None)
    if feat_cols is None or len(feat_cols) == 0:
        # fallback (will fail if categoricals are strings)
        feat_cols = cfg.data.columns.numeric + cfg.data.columns.categorical_low_card

    X = df[feat_cols].astype(np.float32).to_numpy()
    data['encounter'].x = torch.from_numpy(X)

    if include_target:
        data['encounter'].y = torch.tensor((df[cfg.data.target.name].isin(cfg.data.target.positive_values)).astype(int).values, dtype=torch.long)
    # create mappings from entity to node index using provided vocabs
    # Prepare containers for edges
    edge_index_dict = defaultdict(lambda: [[], []])
    edge_attr_dict = {}
    # iterate encounters
    for enc_idx, row in df.iterrows():
        # Edges: encounter -> icd
        if cfg.graph.node_types_enabled.get('icd', False):
            for col in cfg.data.columns.icd_cols:
                if pd.isna(row[col]) or row[col] == '?' or row[col] == '':
                    continue
                code = str(row[col])
                if cfg.data.preprocessing.truncate_icd_to_3_digits:
                    code = code[:3] if len(code) >= 3 else code
                icd_idx = vocabs['icd'].get(code, vocabs['icd'].get(cfg.data.preprocessing.unknown_label))
                edge_index_dict[('encounter', 'has_icd', 'icd')][0].append(enc_idx)
                edge_index_dict[('encounter', 'has_icd', 'icd')][1].append(icd_idx)
                # icd -> icd_group
                if cfg.graph.node_types_enabled.get('icd_group', False) and cfg.graph.edge_types_enabled.get('icd__is_a__icd_group', False):
                    group = mappings.map_icd_to_group(code)
                    if group:
                        grp_idx = vocabs['icd_group'].get(group, vocabs['icd_group'].get(cfg.data.preprocessing.unknown_label))
                        edge_index_dict[('icd', 'is_a', 'icd_group')][0].append(icd_idx)
                        edge_index_dict[('icd', 'is_a', 'icd_group')][1].append(grp_idx)
        # Edges: encounter -> drug
        if cfg.graph.node_types_enabled.get('drug', False):
            for drug in cfg.data.columns.drug_cols:
                val = str(row[drug])
                if val is None or val == 'No' or val == 'nan':
                    continue
                drug_idx = vocabs['drug'].get(drug, vocabs['drug'].get(cfg.data.preprocessing.unknown_label))
                if cfg.graph.edge_types_enabled.get('encounter__has_drug__drug', False):
                    if cfg.graph.edge_featureing.get('has_drug', {}).get('relation_subtypes_by_status', False):
                        rel = f"has_drug_{val.lower()}"
                        edge_index_dict[('encounter', rel, 'drug')][0].append(enc_idx)
                        edge_index_dict[('encounter', rel, 'drug')][1].append(drug_idx)
                    else:
                        edge_index_dict[('encounter', 'has_drug', 'drug')][0].append(enc_idx)
                        edge_index_dict[('encounter', 'has_drug', 'drug')][1].append(drug_idx)
                        if cfg.graph.edge_featureing.get('has_drug', {}).get('edge_attr_status', False):
                            status_val = 1 if val.lower() == 'up' else (-1 if val.lower() == 'down' else 0)
                            edge_attr_dict.setdefault(('encounter','has_drug','drug'), []).append(status_val)
                # drug -> drug_class
                if cfg.graph.node_types_enabled.get('drug_class', False) and cfg.graph.edge_types_enabled.get('drug__belongs_to__drug_class', False):
                    cls = mappings.map_drug_to_class(drug)
                    cls_idx = vocabs['drug_class'].get(cls, vocabs['drug_class'].get(cfg.data.preprocessing.unknown_label))
                    edge_index_dict[('drug', 'belongs_to', 'drug_class')][0].append(drug_idx)
                    edge_index_dict[('drug', 'belongs_to', 'drug_class')][1].append(cls_idx)
        # Edges: encounter -> hospital
        if cfg.data.columns.hospital_col and cfg.graph.node_types_enabled.get('hosp', False):
            hosp_id = str(row[cfg.data.columns.hospital_col])
            hosp_idx = vocabs['hosp'].get(hosp_id, vocabs['hosp'].get(cfg.data.preprocessing.unknown_label))
            edge_index_dict[('encounter', 'at_hospital', 'hosp')][0].append(enc_idx)
            edge_index_dict[('encounter', 'at_hospital', 'hosp')][1].append(hosp_idx)
        # Edges: encounter -> specialty
        if cfg.data.columns.specialty_col and cfg.graph.node_types_enabled.get('specialty', False):
            spec = str(row[cfg.data.columns.specialty_col])
            spec_idx = vocabs['specialty'].get(spec, vocabs['specialty'].get(cfg.data.preprocessing.unknown_label))
            edge_index_dict[('encounter', 'has_specialty', 'specialty')][0].append(enc_idx)
            edge_index_dict[('encounter', 'has_specialty', 'specialty')][1].append(spec_idx)
        # Edges: encounter -> admission_source
        if cfg.data.columns.admission_source_col and cfg.graph.node_types_enabled.get('admission_source', False):
            src_id = str(int(row[cfg.data.columns.admission_source_col])) if not pd.isna(row[cfg.data.columns.admission_source_col]) else cfg.data.preprocessing.unknown_label
            src_idx = vocabs['admission_source'].get(src_id, vocabs['admission_source'].get(cfg.data.preprocessing.unknown_label))
            edge_index_dict[('encounter', 'has_admission_source', 'admission_source')][0].append(enc_idx)
            edge_index_dict[('encounter', 'has_admission_source', 'admission_source')][1].append(src_idx)
        # Edges: encounter -> admission_type
        if cfg.data.columns.admission_type_col and cfg.graph.node_types_enabled.get('admission_type', False):
            typ_id = str(int(row[cfg.data.columns.admission_type_col])) if not pd.isna(row[cfg.data.columns.admission_type_col]) else cfg.data.preprocessing.unknown_label
            typ_idx = vocabs['admission_type'].get(typ_id, vocabs['admission_type'].get(cfg.data.preprocessing.unknown_label))
            edge_index_dict[('encounter', 'has_admission_type', 'admission_type')][0].append(enc_idx)
            edge_index_dict[('encounter', 'has_admission_type', 'admission_type')][1].append(typ_idx)
        # Edges: encounter -> discharge_disposition
        if cfg.data.columns.discharge_disposition_col and cfg.graph.node_types_enabled.get('discharge_disposition', False):
            dd_id = str(int(row[cfg.data.columns.discharge_disposition_col])) if not pd.isna(row[cfg.data.columns.discharge_disposition_col]) else cfg.data.preprocessing.unknown_label
            dd_idx = vocabs['discharge_disposition'].get(dd_id, vocabs['discharge_disposition'].get(cfg.data.preprocessing.unknown_label))
            edge_index_dict[('encounter', 'has_discharge', 'discharge_disposition')][0].append(enc_idx)
            edge_index_dict[('encounter', 'has_discharge', 'discharge_disposition')][1].append(dd_idx)
    # Assign node features for non-encounter types: just an index as feature (will be embedded in model)
    for node_type, vocab in vocabs.items():
        if node_type == 'encounter':
            continue
        num_nodes = len(vocab)
        data[node_type].num_nodes = num_nodes
        # we use index as feature placeholder
        data[node_type].x = torch.arange(num_nodes, dtype=torch.long)
    # Set edge indices in HeteroData
    for (src, rel, dst), (s_list, d_list) in edge_index_dict.items():
        if len(s_list) == 0:
            continue
        data[(src, rel, dst)].edge_index = torch.tensor([s_list, d_list], dtype=torch.long)
        # attach edge attr if present
        if (src, rel, dst) in edge_attr_dict:
            data[(src, rel, dst)].edge_attr = torch.tensor(edge_attr_dict[(src, rel, dst)], dtype=torch.float)
        # If reverse edges enabled, add them
        if cfg.graph.edge_types_enabled.get('reverse_edges', False):
            rev_rel = "rev_" + rel
            data[(dst, rev_rel, src)].edge_index = torch.tensor([d_list, s_list], dtype=torch.long)
    return data
