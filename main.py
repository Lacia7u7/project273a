#%%
import sys

import utils.io

if 'google.colab' in sys.modules:
    # Install pyg-lib for accelerated neighborhood sampling
    # NOTE: Make sure you are using the correct torch version
    # https://pytorch.org/get-started/locally/
    # Install torch_geometric and its dependencies
    import os
    import torch

    # Install torch-scatter and torch-sparse
    # NOTE: Make sure you are using the correct torch version
    # https://pytorch.org/get-started/locally/
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.', '')

    !pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch_geometric
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.', '')

    !pip install pyg-lib -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
#%%
import sys
import os

# Run this only if running on Colab
if 'google.colab' in sys.modules:
    repo_dir = 'project273a'
    if not os.path.exists(repo_dir):
        !git clone https://github.com/carloea2/project273a.git
    %cd {repo_dir}
    !git pull origin master # Pull the latest changes from the master branch
    sys.path.append('/content/project273a')
#%%
%load_ext autoreload
%autoreload 1
#%%
# Imports for utilities
import os

from utils.config import Config  # Pydantic data model for config
from utils.logging import init_logging
from utils.seed import set_seed

# Define configuration as a nested dictionary (or use JSON format)
config_dict = {
    "system": {
        "numexpr_threads": 4,
        "deterministic": False,
        "cpu": {
          "intra_op_threads": None,
          "inter_op_threads": None,
          "omp_num_threads": None,
          "mkl_num_threads": None,
          "kmp_affinity": "granularity=fine,compact,1,0",
          "start_method": "forkserver",
          "pin_affinity_cores": None
        },
        "dataloader": {
          "num_workers": os.cpu_count()//2,           # auto -> max(2, cores-1)
          "prefetch_factor": 4,
          "persistent_workers": True,
          "pin_memory": True,
          "pin_memory_device": "cuda",
          "non_blocking": True
        },
        "cuda": {
          "enabled": True,
          "device_ids": None,            # auto -> all visible
          "allow_tf32": True,
          "matmul_precision": "high",
          "cudnn_benchmark": True,
          "cudnn_deterministic": None,
          "amp": False,
          "amp_dtype": "bf16",           # prefer bf16 on Ampere+ if available
          "grad_scaler_enabled": True,
          "compile_mode": "reduce-overhead",
          "compile_fullgraph": True,
          "uva": True
        },
        "ddp": {
          "enabled": False,
          "backend": "nccl",
          "find_unused_parameters": False,
          "gradient_as_bucket_view": True,
          "broadcast_buffers": False,
          "static_graph": False
        }
    },
    "data": {
        "csv_path": "raw/diabetic_data.csv",            # raw data file
        "ids_mapping_path": "raw/IDS_mapping.csv",          # ID mapping file for codes
        "target": {"name": "readmitted", "positive_values": ["<30"], "binarized_name": "readmitted_binary"},  # predict 30-day readmit
        "identifier_cols": {"encounter_id": "encounter_id", "patient_id": "patient_nbr"},
        "filters": {
            "exclude_discharge_to_ids": [11, 13, 14, 19, 20, 21],
            "first_encounter_per_patient": True,
        },
        "columns": {
            # Numeric features (counts, etc.)
            "numeric": ["time_in_hospital", "num_lab_procedures", "num_procedures",
                        "num_medications", "number_outpatient", "number_emergency",
                        "number_inpatient", "number_diagnoses"],
            # Low-cardinality categoricals (will be one-hot or label encoded as features)
            "categorical_low_card": ["race", "gender", "age", "max_glu_serum", "A1Cresult", "change", "diabetesMed"],
            # High-cardinality categorical columns to be turned into separate nodes
            "icd_cols": ["diag_1", "diag_2", "diag_3"],          # diagnosis code columns
            "drug_cols": ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
                          "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
                          "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
                          "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
                          "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"],
            "hospital_col": None,                                # (dataset has no explicit hospital ID column)
            "specialty_col": "medical_specialty",                # physician specialty
            "admission_type_col": "admission_type_id",
            "discharge_disposition_col": "discharge_disposition_id",
            "admission_source_col": "admission_source_id"
        },
        "preprocessing": {
            "numeric_imputer": "mean",           # impute missing numeric with mean
            "categorical_imputer": "most_frequent",  # impute missing categoricals with mode
            "unknown_label": "UNKNOWN",          # label for unseen or rare categories
            "use_unknown_category": True,        # add an "UNKNOWN" category for unseen values
            "min_freq_for_category": 5,          # rare category threshold (below this -> UNKNOWN)
            "truncate_icd_to_3_digits": True     # use only first 3 digits of ICD codes to group
        },
        "splits": {
            "group_by": "patient",    # group splits by patient_id to avoid leakage:contentReference[oaicite:6]{index=6}
            "n_splits": 5,           # use 5-fold split (first fold for train/val, second for test)
            "stratify_by_target": True,
            "seed": 42
        }
    },
    "graph": {
        # Enable various node and edge types in the heterogeneous graph
        "node_types_enabled": {
            "encounter": True, "icd": True, "icd_group": True, "drug": True, "drug_class": True,
            "specialty": True, "admission_type": True, "discharge_disposition": True, "admission_source": True,
            "hosp": True
        },
        "edge_types_enabled": {
            "encounter__has_icd__icd": True,
            "icd__is_a__icd_group": True,
            "encounter__has_drug__drug": True,
            "drug__belongs_to__drug_class": True,
            "encounter__has_specialty__specialty": True,
            "encounter__has_admission_type__admission_type": True,
            "encounter__has_discharge__discharge_disposition": True,
            "encounter__has_admission_source__admission_source": True,
            "reverse_edges": True    # add reverse of every relation for undirected information flow
        },
        "edge_featuring": {
            "has_drug": {
                "relation_subtypes_by_status": True,  # separate edge types for Up/Down/Steady drug status
                "edge_attr_status": True              # include an edge attribute indicating drug change
            }
        },
        "fanouts": {
            # Neighbor sampling fanout per edge type per GNN layer (2-layer example):
            "encounter__has_icd__icd": [10, 5, 3],
            "encounter__has_drug__drug": [10, 5, 3],
            "encounter__has_specialty__specialty": [-1],  # -1 means take all neighbors (specialty has 1 neighbor per encounter)
            "encounter__has_admission_type__admission_type": [-1],
            "encounter__has_discharge__discharge_disposition": [-1],
            "encounter__has_admission_source__admission_source": [-1],
            "icd__is_a__icd_group": [-1],
            "drug__belongs_to__drug_class": [-1],
            "reverse_edges": [10, 5, 3]  # sample some reverse edges if needed
        }
    },
    "model": {
        "arch": "RGCN",           # model architecture: "HGT", "RGCN", or "GraphSAGE"
        "hidden_dim": 128,        # hidden embedding size
        "num_layers": 3,         # number of GNN layers
        "heads": 4,              # number of attention heads (for HGT)
        "rgcn_bases": 4,       # number of bases for RGCN
        "dropout": 0.45, #0.25
        "loss":{
            "pos_weight": "none"
        }
    },
    "train": {
        "epochs": 60,
        "early_stopping_patience": 8,
        "val_every": 1,          # evaluate on val every epoch
        "gradient_clip_norm": 2.0,
        "optimizer": {
            "name": "AdamW",
            "lr":  0.0005,
            #"lr":  0.01,
            "weight_decay": 0.1,
        },
        "batching": {
            "batch_size_encounters": 1024
        }
    },
    "inference": {
        "output_predictions_path": "artifacts/predictions.csv"
    },
    "evaluation":{
        "metrics_primary": ["auprc", "auroc", "f1_pos"],
        "metrics_secondary": [
            "precision_pos", "recall_pos", "specificity",
            "balanced_accuracy", "brier", "logloss"
        ],

        "threshold_tuning": {
            "optimize_for": "f1_pos",
            "grid": []                       # [] -> let find_best_threshold choose; or e.g., [0.1,0.2,...,0.9]
        },
        "plots": {
            "roc": True,
            "pr": True, "calibration":
            True, "confusion": True,
            "decision_curves": True
        }
    },
    "baseline": {},
    "path": {
        "artifacts_dir": "artifacts/",
        "tb_log_dir": "artifacts/tb_logs/",
        "logging_path": "logs/"
    },
      "metrics_primary": ["auprc", "auroc", "f1_pos"],
    "metrics_secondary": [
        "precision_pos", "recall_pos", "specificity",
        "balanced_accuracy", "brier", "logloss"
    ],

    # optional: used when you call evaluate_predictions(..., tune_threshold=True)
    "threshold_tuning": {
        "optimize_for": "f1_pos",        # must be a key your compute_metrics() returns
        "grid": []                       # [] -> let find_best_threshold choose; or e.g., [0.1,0.2,...,0.9]
    },

    # optional: only if you pass `metadata` with these columns to evaluate_predictions(...)
    "subgroup_metrics": ["gender", "race", "age"],

    # `plots` exists in your schema with defaults; omit or set explicitly if you like
    # "plots": {"roc": True, "pr": True, "calibration": True, "confusion": True, "decision_curves": True}
}

# Initialize config object
config = Config(**config_dict)
#%%
from utils.system import apply_system_config  # or from the cell above
rt = apply_system_config(config)
device = rt["device"]

# Set random seeds for reproducibility
set_seed(config, 42)

# Initialize logging and TensorBoard writer
logger, writer = init_logging(config.path.logs_dir)
logger.info("Configuration and logging initialized.")
#%%
from data.filters import apply_filters
import data

# Load the datasets
df = utils.io.load_csv(config.data.csv_path)
df = apply_filters(df, config)

logger.info(f"Raw data shape: {df.shape}")
logger.info(f"Columns: {list(df.columns)}")
#%%
from data import preprocess

# Create train/val/test splits first (to fit imputer/scaler on train only)
from data.splits import create_splits, check_no_leakage, check_target_distribution

train_idx, val_idx, test_idx = create_splits(df, config)

# 1) Always verify leakage first
check_no_leakage(df, train_idx, val_idx, test_idx, config)

# 2) Then verify target distribution (auto-detects task type)
report = check_target_distribution(
    df, train_idx, val_idx, test_idx, config,
    task="auto",        # or "classification" / "regression"
    tol=0.05,           # for classification
    mean_z_tol=0.25,    # for regression
    std_ratio_bounds=(0.5, 1.5),
    q_tol=0.20,
    strict=True         # set False to just get a report without raising
)
print(report["ok"], report.get("reasons", []))
#%%
df_train = df.iloc[train_idx].copy().reset_index(drop=True)
df_val = df.iloc[val_idx].copy().reset_index(drop=True)
df_test = df.iloc[test_idx].copy().reset_index(drop=True) if test_idx is not None else None

logger.info(f"Split sizes -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Preprocess the splits
df_train, df_val, df_test, scaler = preprocess.preprocess_data(df_train, df_val, df_test, config)

# The scaler and any encodings from train are now ready for use in inference too
logger.info("Preprocessing complete. Sample of processed features:")
logger.info(df_train[config.data.columns.numeric + config.data.columns.categorical_low_card].head(3))
#%%
RUN_DATA_EXPLORATION = False

if RUN_DATA_EXPLORATION:
    from evaluation.data_exploration import DataExplorer
    try:
        from IPython.display import display
    except Exception:  # pragma: no cover - fallback for non-IPython envs
        def display(obj):  # type: ignore[redefinition]
            print(obj)

    explorer = DataExplorer(df_train, target_column=config.data.target.binarized_name)
    data_summary = explorer.summary_table()
    class_balance_fig = explorer.class_balance_plot()
    violin_fig = explorer.violin_plot()

    display(data_summary.head())
    class_balance_fig.show()
    violin_fig.show()
#%%
from data import vocab

vocabs, mappings = vocab.make_vocabs(df_train, config)
logger.info("Vocab sizes: " + ", ".join(f"{k}: {len(v)}" for k,v in vocabs.items()))
#%%
# Verify no patient overlap between train, val, test
train_patients = set(df_train[config.data.identifier_cols.patient_id])
val_patients = set(df_val[config.data.identifier_cols.patient_id])
test_patients = set(df_test[config.data.identifier_cols.patient_id])

overlap_train_val = train_patients.intersection(val_patients)
overlap_train_test = train_patients.intersection(test_patients)
overlap_val_test = val_patients.intersection(test_patients)
logger.info(f"Patient overlap - Train/Val: {len(overlap_train_val)}, Train/Test: {len(overlap_train_test)}, Val/Test: {len(overlap_val_test)}")

# Ensure target stratification roughly preserved
mean_train = df_train[config.data.target.binarized_name].mean(); mean_val = df_val[config.data.target.binarized_name].mean(); mean_test = df_test[config.data.target.binarized_name].mean()
logger.info(f"Readmit rate - Train: {mean_train:.3f}, Val: {mean_val:.3f}, Test: {mean_test:.3f}")
#%%
from graph import builder

# Build heterogeneous graphs for each split
graph_train = builder.build_heterodata(df_train, vocabs, config, include_target=True)
graph_val   = builder.build_heterodata(df_val, vocabs, config, include_target=True)
graph_test  = builder.build_heterodata(df_test, vocabs, config, include_target=True)

# Log graph statistics
logger.info(f"Graph (Train) node types: {list(graph_train.node_types)}")
for ntype in graph_train.node_types:
    logger.info(f"  {ntype}: {graph_train[ntype].num_nodes} nodes")
logger.info(f"Graph (Train) edge types: {list(graph_train.edge_types)}")
for etype in graph_train.edge_types:
    logger.info(f"  {etype}: {graph_train[etype].edge_index.size(1)} edges")
#%%
# Jupyter cell (with tqdm bars)
import torch

from train.model_factory import get_model_class, setup_and_compile_model
from train.optim import make_optimizer, make_scheduler
from data.sampling import build_num_neighbors
from train.losses import make_criterion
from train.device import get_device
# sizes from the TRAIN graph
enc_input_dim = graph_train['encounter'].x.size(-1)
type_vocab_sizes = {nt: graph_train[nt].num_nodes for nt in graph_train.node_types if nt != 'encounter'}

# optional one-time sanity check
def validate_indices(g):
    for nt in g.node_types:
        if nt == 'encounter':
            continue
        x = g[nt].x
        if x.numel() == 0:
            continue
        vmax = int(x.max().item())
        n = int(g[nt].num_nodes)
        assert vmax < n, f"[{nt}] max index {vmax} >= num_nodes {n}"
validate_indices(graph_train)

# build model (move to device after full construction)
device = get_device()
ModelClass = get_model_class(config)
metadata = (list(graph_train.node_types), list(graph_train.edge_types))
model = ModelClass(
    metadata,
    config,
    enc_input_dim=enc_input_dim,
    type_vocab_sizes=type_vocab_sizes,
).to(device)  # or pass device=device in the constructor instead
model = setup_and_compile_model(model, config, logger)
#%%
model
#%%
from train.loader import make_neighbor_loader
from tqdm.auto import tqdm, trange
from train.loop import Trainer
from utils.artifacts import save_best_artifact

# --- build optim, sched, loss as before ---
optimizer = make_optimizer(model.parameters(), config)
scheduler = make_scheduler(optimizer, config)
criterion = make_criterion(graph_train, config, device)

# --- NeighborLoader as before ---
num_layers = int(getattr(config.model, "num_layers", 2))
num_neighbors = build_num_neighbors(graph_train, config, num_layers)
train_loader = make_neighbor_loader(
    graph_train,
    input_nodes=("encounter", torch.arange(graph_train["encounter"].num_nodes)),
    num_neighbors=num_neighbors,
    config=config,
    train=True,
    shuffle=True
)

val_data = graph_val.to(device)
#%%
RUN_GNN_GRID_SEARCH = False

if RUN_GNN_GRID_SEARCH:
    from grid_search.gnn import GridSearchGNN
    try:
        from IPython.display import display
    except Exception:  # pragma: no cover
        def display(obj):  # type: ignore[redefinition]
            print(obj)

    grid_search = GridSearchGNN(
        ModelClass,
        base_config=config,
        model_kwargs={
            "metadata": metadata,
            "enc_input_dim": enc_input_dim,
            "type_vocab_sizes": type_vocab_sizes,
            "device": device,
        },
        train_kwargs={
            "epochs": 5,
            "use_trainer": True,
            "runtime": rt,
            "trainer_kwargs": {
                "early_stopping_patience": int(getattr(config.train, "early_stopping_patience", 5)),
                "val_every": int(getattr(config.train, "val_every", 1)),
            },
        },
        metric_name="auprc",
    )

    gnn_grid_report = grid_search.run(train_loader, val_data=val_data, device=device)
    display(gnn_grid_report.results.head())
    logger.info(f"Best grid search params: {gnn_grid_report.best_params} -> {gnn_grid_report.best_score:.4f}")

# --- Trainer instance ---
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    config=config,
    rt=rt,
    scheduler=scheduler,
    writer=writer if "writer" in globals() else None,
    logger=logger if "logger" in globals() else None,
    save_best_fn=lambda payload: save_best_artifact(payload, config, artifacts_dir="artifacts"),
    early_stopping_patience=int(config.train.early_stopping_patience),
    val_every=int(getattr(config.train, "val_every", 1)),
)

epochs   = int(config.train.epochs)
val_every = trainer.val_every
grad_clip = getattr(config.train, "gradient_clip_norm", None)

epoch_bar = trange(1, epochs + 1, desc="Epochs", dynamic_ncols=True, leave=True)

#%%

for epoch in epoch_bar:
    # per-epoch inner bar
    batch_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}", dynamic_ncols=True, leave=False)

    avg_loss = trainer.train_epoch(train_loader, batch_bar=batch_bar, grad_clip=grad_clip)
    trainer.log_train_loss(avg_loss, epoch=epoch)

    val_auprc = None
    if epoch % val_every == 0:
        val_auprc = trainer.validate_auprc(val_data)
        early_stop = trainer.update_after_validation(val_auprc, epoch=epoch)
    else:
        early_stop = False

    # progress + logs
    if val_auprc is not None:
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", val_auprc=f"{val_auprc:.4f}")
        logger.info(f"Epoch {epoch} - Train loss: {avg_loss:.4f} | Val AUPRC: {val_auprc:.4f}")
    else:
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        logger.info(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")

    if early_stop:
        tqdm.write("Early stopping triggered.")
        break

# Load best weights at the end (and the best artifact was already saved when it improved)
best_state = trainer.load_best()
#%%
artifact_path = save_best_artifact(best_state, config, artifacts_dir="artifacts")
logger.info(f"Saved best artifact to: {artifact_path}")
#%%
from utils.artifacts import load_best_artifact

# Restore latest (or pass a specific run_id):
artifact = load_best_artifact(artifacts_dir="artifacts", run_id="latest", map_location="cpu")

# Rebuild your model (same architecture as training) and load weights:
model.load_state_dict(artifact["best_state"]["model"])

# Access stored metadata:
print("Run:", artifact["run_id"])
print("Best epoch:", artifact["best_state"]["epoch"])
print("Best Val AUPRC:", artifact["best_state"]["val_auprc"])
restored_config = artifact["config"]  # JSON-serializable dict
#%%
from evaluation.metrics import compute_metrics, find_best_threshold
from evaluation.calibration import calibrate_probabilities, apply_calibration

# Evaluate on validation set
model.eval()
with torch.no_grad():
    val_out = model(val_data.x_dict, val_data.edge_index_dict)
    test_data = graph_test.to(device)
    test_out = model(test_data.x_dict, test_data.edge_index_dict)
val_probs = torch.sigmoid(val_out).cpu().numpy()
test_probs = torch.sigmoid(test_out).cpu().numpy()
val_labels = graph_val['encounter'].y.cpu().numpy()
test_labels = graph_test['encounter'].y.cpu().numpy()

# Compute metrics at default 0.5 threshold
val_metrics = compute_metrics(val_labels, val_probs)
test_metrics = compute_metrics(test_labels, test_probs)
logger.info("Validation metrics at 0.5 threshold: " + ", ".join(f"{k}={v:.4f}" for k,v in val_metrics.items()))
logger.info("Test metrics at 0.5 threshold: " + ", ".join(f"{k}={v:.4f}" for k,v in test_metrics.items()))

# Find best threshold on validation for F1 score
best_thr, metric_name, best_f1 = find_best_threshold(val_labels, val_probs, optimize_for='f1_pos')
logger.info(f"Best threshold for F1 on val = {best_thr:.2f}, F1 at best thr = {best_f1:.4f}")
# Apply this threshold to test set
test_pred_opt = (test_probs >= best_thr).astype(int)
from sklearn.metrics import f1_score, precision_score, recall_score
f1_test_opt = f1_score(test_labels, test_pred_opt, pos_label=1)
precision_test_opt = precision_score(test_labels, test_pred_opt, pos_label=1, zero_division=0)
recall_test_opt = recall_score(test_labels, test_pred_opt, pos_label=1)
logger.info(f"Test F1={f1_test_opt:.4f}, Precision={precision_test_opt:.4f}, Recall={recall_test_opt:.4f} at threshold {best_thr:.2f}")

# Probability calibration (using validation set)
cal_model = calibrate_probabilities(val_probs, val_labels, method="platt")  # or "isotonic"
cal_test_probs = apply_calibration(cal_model, test_probs)
cal_metrics = compute_metrics(test_labels, cal_test_probs)
logger.info("Test metrics after calibration: " + ", ".join(f"{k}={v:.4f}" for k,v in cal_metrics.items()))

#%%
best_thr, metric_name, best_f1
#%%
RUN_MODEL_EVALUATOR = False

if RUN_MODEL_EVALUATOR:
    from evaluation.model_evaluator import Evaluator
    try:
        from IPython.display import display
    except Exception:  # pragma: no cover
        def display(obj):  # type: ignore[redefinition]
            print(obj)

    test_data = [graph_test]
    test_labels_np = graph_test["encounter"].y.cpu().numpy()

    evaluator = Evaluator({"gnn_model": model})
    evaluator.evaluate(test_data, test_labels_np)

    display(evaluator.metrics_summary_table())
    display(evaluator.threshold_metrics_table())
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix
from sklearn.calibration import CalibrationDisplay
# ROC Curve
RocCurveDisplay.from_predictions(test_labels, test_probs)
plt.title("ROC Curve (Test)")
plt.show()

# Precision-Recall Curve
PrecisionRecallDisplay.from_predictions(test_labels, test_probs)
plt.title("Precision-Recall Curve (Test)")
plt.show()

# Calibration curve (reliability diagram)
CalibrationDisplay.from_predictions(test_labels, test_probs, n_bins=10, strategy='uniform')
plt.title("Calibration Curve (Test)")
plt.show()

# Confusion Matrix at optimal threshold
cm = confusion_matrix(test_labels, test_pred_opt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Readmit","Readmit<30"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (Thr={best_thr:.2f})")
plt.show()

# Decision curve analysis (net benefit vs threshold)
from evaluation.threshold import decision_curve_analysis
dc_df = decision_curve_analysis(test_labels, test_probs)
plt.plot(dc_df['threshold'], dc_df['net_benefit'], label='GNN Model')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve (Test Set)")
plt.legend()
plt.show()

#%%
from benchmarks.baselines import train_and_eval_baselines
# Prepare NumPy arrays for baseline models
# Prepend "oh__" to categorical_low_card columns as they are likely one-hot encoded
oh_categorical_cols = ['oh__race_AfricanAmerican', 'oh__race_Asian',
       'oh__race_Caucasian', 'oh__race_Hispanic', 'oh__race_Other',
       'oh__gender_Female', 'oh__gender_Male', 'oh__gender_UNKNOWN',
       'oh__age_[0-10)', 'oh__age_[10-20)', 'oh__age_[20-30)',
       'oh__age_[30-40)', 'oh__age_[40-50)', 'oh__age_[50-60)',
       'oh__age_[60-70)', 'oh__age_[70-80)', 'oh__age_[80-90)',
       'oh__age_[90-100)', 'oh__max_glu_serum_>200', 'oh__max_glu_serum_>300',
       'oh__max_glu_serum_Norm', 'oh__A1Cresult_>7', 'oh__A1Cresult_>8',
       'oh__A1Cresult_Norm', 'oh__change_Ch', 'oh__change_No',
       'oh__diabetesMed_No', 'oh__diabetesMed_Yes']
feature_cols = config.data.columns.numeric  + oh_categorical_cols
X_train_tab = df_train[feature_cols].to_numpy()
y_train_tab = df_train[config.data.target.binarized_name].to_numpy()
X_val_tab = df_val[feature_cols].to_numpy()
y_val_tab = df_val[config.data.target.binarized_name].to_numpy()
X_test_tab = df_test[feature_cols].to_numpy()
y_test_tab = df_test[config.data.target.binarized_name].to_numpy()
#%%
RUN_STABILITY_STUDY = False

if RUN_STABILITY_STUDY:
    from sklearn.linear_model import LogisticRegression
    from evaluation.model_evaluator import RepeatedTrainingStudy
    try:
        from IPython.display import display
    except Exception:  # pragma: no cover
        def display(obj):  # type: ignore[redefinition]
            print(obj)

    def make_log_reg() -> LogisticRegression:
        return LogisticRegression(max_iter=500, solver="lbfgs")

    stability = RepeatedTrainingStudy(
        make_log_reg,
        metric_names=["auroc", "auprc", "f1_pos", "precision_pos", "recall_pos"],
    )

    runs_df, summary_df = stability.run(
        X_train_tab,
        y_train_tab,
        X_test_tab,
        y_test_tab,
        n_runs=5,
        random_state=42,
    )

    display(runs_df)
    display(summary_df)

print("Starting")
# Train and evaluate baseline models
baseline_results = train_and_eval_baselines(X_train_tab, y_train_tab, X_val_tab, y_val_tab, X_test_tab, y_test_tab, config=config)
print("Finishing")
# Display baseline evaluation results
for model_name, metrics_dict in baseline_results.items():
    test_met = metrics_dict["test"]
#%%
