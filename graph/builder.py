import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from typing import Dict, Any

from data import mappings
from utils.config import Config, EdgeFeatConfig


def build_heterodata(df: "pd.DataFrame",
                     vocabs: Dict[str, Dict[str, int]],
                     config: Config,
                     include_target: bool = False) -> HeteroData:
    """
    Build a PyG HeteroData graph from a dataframe and vocabularies.

    - Respects GraphConfig.edge_featuring (Pydantic models, not dicts).
    - Robust handling of NaNs / unknowns.
    - Adds reverse edges if enabled.
    """
    data = HeteroData()
    cfg = config

    # -------- features for encounter nodes --------
    feat_cols = getattr(cfg.data.columns, "encounter_features", None)
    if not feat_cols:  # fallback (warn: will fail if categoricals are strings)
        feat_cols = cfg.data.columns.numeric + cfg.data.columns.categorical_low_card

    X = df[feat_cols].astype(np.float32).to_numpy()
    data["encounter"].x = torch.from_numpy(X)

    if include_target:
        y = (df[cfg.data.target.name].isin(cfg.data.target.positive_values)).astype(int).values
        data["encounter"].y = torch.tensor(y, dtype=torch.long)

    # Short-hands
    unk = cfg.data.preprocessing.unknown_label
    nodes_enabled = cfg.graph.node_types_enabled
    edges_enabled = cfg.graph.edge_types_enabled
    add_reverse = edges_enabled.get("reverse_edges", False)

    # Edge containers
    edge_index_dict: Dict[Any, list] = defaultdict(lambda: [[], []])
    edge_attr_dict: Dict[Any, list] = {}

    # Cache edge featuring config for drugs (Pydantic model!)
    drug_feat_cfg: EdgeFeatConfig = cfg.graph.edge_featuring.get("has_drug", EdgeFeatConfig())
    relation_subtypes = getattr(drug_feat_cfg, "relation_subtypes_by_status", False)
    edge_attr_status = getattr(drug_feat_cfg, "edge_attr_status", False)

    # -------- iterate encounters/rows --------
    for enc_idx, row in df.iterrows():

        # ---- encounter -> icd ----
        if nodes_enabled.get("icd", False) and edges_enabled.get("encounter__has_icd__icd", True):
            for col in cfg.data.columns.icd_cols:
                raw = row[col]
                if pd.isna(raw) or raw in ("?", ""):
                    continue
                code = str(raw)
                if cfg.data.preprocessing.truncate_icd_to_3_digits and len(code) >= 3:
                    code = code[:3]
                icd_idx = vocabs["icd"].get(code, vocabs["icd"].get(unk, 0))
                edge_index_dict[("encounter", "has_icd", "icd")][0].append(enc_idx)
                edge_index_dict[("encounter", "has_icd", "icd")][1].append(icd_idx)

                # icd -> icd_group
                if (nodes_enabled.get("icd_group", False)
                        and edges_enabled.get("icd__is_a__icd_group", False)):
                    group = mappings.map_icd_to_group(code)
                    if group:
                        grp_idx = vocabs["icd_group"].get(group, vocabs["icd_group"].get(unk, 0))
                        edge_index_dict[("icd", "is_a", "icd_group")][0].append(icd_idx)
                        edge_index_dict[("icd", "is_a", "icd_group")][1].append(grp_idx)

        # ---- encounter -> drug (+ optional attr / relation subtypes) ----
        if nodes_enabled.get("drug", False) and edges_enabled.get("encounter__has_drug__drug", True):
            for drug in cfg.data.columns.drug_cols:
                raw = row[drug]
                # Robust skip: NaN or explicit "No"
                if pd.isna(raw) or raw == "No":
                    continue
                val = str(raw).strip().lower()  # up / down / steady / etc.

                drug_idx = vocabs["drug"].get(drug, vocabs["drug"].get(unk, 0))

                if relation_subtypes:
                    rel = f"has_drug_{val}"
                    edge_index_dict[("encounter", rel, "drug")][0].append(enc_idx)
                    edge_index_dict[("encounter", rel, "drug")][1].append(drug_idx)
                else:
                    edge_index_dict[("encounter", "has_drug", "drug")][0].append(enc_idx)
                    edge_index_dict[("encounter", "has_drug", "drug")][1].append(drug_idx)
                    if edge_attr_status:
                        status_val = 1 if val == "up" else (-1 if val == "down" else 0)
                        edge_attr_dict.setdefault(("encounter", "has_drug", "drug"), []).append(status_val)

                # drug -> drug_class
                if (nodes_enabled.get("drug_class", False)
                        and edges_enabled.get("drug__belongs_to__drug_class", False)):
                    cls = mappings.map_drug_to_class(drug)
                    cls_idx = vocabs["drug_class"].get(cls, vocabs["drug_class"].get(unk, 0))
                    edge_index_dict[("drug", "belongs_to", "drug_class")][0].append(drug_idx)
                    edge_index_dict[("drug", "belongs_to", "drug_class")][1].append(cls_idx)

        # ---- encounter -> hospital ----
        if (cfg.data.columns.hospital_col and nodes_enabled.get("hosp", False)
                and edges_enabled.get("encounter__at_hospital__hosp", True)):
            hosp_id = str(row[cfg.data.columns.hospital_col])
            hosp_idx = vocabs["hosp"].get(hosp_id, vocabs["hosp"].get(unk, 0))
            edge_index_dict[("encounter", "at_hospital", "hosp")][0].append(enc_idx)
            edge_index_dict[("encounter", "at_hospital", "hosp")][1].append(hosp_idx)

        # ---- encounter -> specialty ----
        if (cfg.data.columns.specialty_col and nodes_enabled.get("specialty", False)
                and edges_enabled.get("encounter__has_specialty__specialty", True)):
            spec = str(row[cfg.data.columns.specialty_col])
            spec_idx = vocabs["specialty"].get(spec, vocabs["specialty"].get(unk, 0))
            edge_index_dict[("encounter", "has_specialty", "specialty")][0].append(enc_idx)
            edge_index_dict[("encounter", "has_specialty", "specialty")][1].append(spec_idx)

        # ---- encounter -> admission_source ----
        if (cfg.data.columns.admission_source_col and nodes_enabled.get("admission_source", False)
                and edges_enabled.get("encounter__has_admission_source__admission_source", True)):
            if pd.isna(row[cfg.data.columns.admission_source_col]):
                src_id = unk
            else:
                src_id = str(int(row[cfg.data.columns.admission_source_col]))
            src_idx = vocabs["admission_source"].get(src_id, vocabs["admission_source"].get(unk, 0))
            edge_index_dict[("encounter", "has_admission_source", "admission_source")][0].append(enc_idx)
            edge_index_dict[("encounter", "has_admission_source", "admission_source")][1].append(src_idx)

        # ---- encounter -> admission_type ----
        if (cfg.data.columns.admission_type_col and nodes_enabled.get("admission_type", False)
                and edges_enabled.get("encounter__has_admission_type__admission_type", True)):
            if pd.isna(row[cfg.data.columns.admission_type_col]):
                typ_id = unk
            else:
                typ_id = str(int(row[cfg.data.columns.admission_type_col]))
            typ_idx = vocabs["admission_type"].get(typ_id, vocabs["admission_type"].get(unk, 0))
            edge_index_dict[("encounter", "has_admission_type", "admission_type")][0].append(enc_idx)
            edge_index_dict[("encounter", "has_admission_type", "admission_type")][1].append(typ_idx)

        # ---- encounter -> discharge_disposition ----
        if (cfg.data.columns.discharge_disposition_col and nodes_enabled.get("discharge_disposition", False)
                and edges_enabled.get("encounter__has_discharge__discharge_disposition", True)):
            if pd.isna(row[cfg.data.columns.discharge_disposition_col]):
                dd_id = unk
            else:
                dd_id = str(int(row[cfg.data.columns.discharge_disposition_col]))
            dd_idx = vocabs["discharge_disposition"].get(dd_id, vocabs["discharge_disposition"].get(unk, 0))
            edge_index_dict[("encounter", "has_discharge", "discharge_disposition")][0].append(enc_idx)
            edge_index_dict[("encounter", "has_discharge", "discharge_disposition")][1].append(dd_idx)

    # -------- assign non-encounter node features --------
    for node_type, vocab in vocabs.items():
        if node_type == "encounter":
            continue
        num_nodes = len(vocab)
        data[node_type].num_nodes = num_nodes
        # simple id feature placeholder (use Embedding in the model)
        data[node_type].x = torch.arange(num_nodes, dtype=torch.long)

    # -------- finalize edges (+ reverse) --------
    for (src, rel, dst), (s_list, d_list) in edge_index_dict.items():
        if not s_list:
            continue
        idx = torch.tensor([s_list, d_list], dtype=torch.long)
        data[(src, rel, dst)].edge_index = idx

        if (src, rel, dst) in edge_attr_dict:
            data[(src, rel, dst)].edge_attr = torch.tensor(edge_attr_dict[(src, rel, dst)], dtype=torch.float)

        if add_reverse:
            rev_rel = "rev_" + rel
            data[(dst, rev_rel, src)].edge_index = torch.tensor([d_list, s_list], dtype=torch.long)

    return data
