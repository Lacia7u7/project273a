#%% md
# # Heterogeneous GNN Pipeline for 30-Day Readmission Prediction
# 
# This notebook implements an **end-to-end pipeline** for predicting 30-day hospital readmissions using a **heterogeneous graph neural network (GNN)** built from a tabular diabetes readmission dataset. It takes you all the way from raw CSV files to:
# 
# - cleaned and split datasets,
# - a heterogeneous graph representation of encounters, diagnoses, drugs, specialties, etc.,
# - training and evaluating a GNN (RGCN / HGT / GraphSAGE),
# - probability calibration and decision-curve analysis,
# - systematic hyperparameter search for the GNN,
# - strong tabular baselines and stability analysis.
# 
# The notebook is designed to be runnable on **Google Colab or a local environment**, with optional GPU acceleration and PyTorch Geometric (PyG).
# 
# ---
# 
# ## High-Level Goals
# 
# 1. **Clinical task**
#    Predict whether a patient will be **readmitted within 30 days** (`readmitted_binary`) based on encounter-level features, diagnoses (ICD codes), medications, and provider / hospital information.
# 
# 2. **Representation learning**
#    Turn a relational/tabular dataset into a **heterogeneous graph** where:
#    - encounters are central nodes,
#    - diagnoses, drugs, specialties, admission types, discharge dispositions, etc. are connected as typed neighbors,
#    - this structure is used by a GNN to learn rich encounter representations.
# 
# 3. **Modeling & evaluation**
#    - Train a heterogeneous GNN (default: **RGCN**, with configuration support for **HGT** or **GraphSAGE**).
#    - Evaluate with clinically relevant metrics (AUPRC, AUROC, F1 for the positive class, calibration, decision curves).
#    - Compare against **tabular ML baselines** (e.g., XGBoost, CatBoost, Logistic Regression).
# 
# 4. **Reproducibility & robustness**
#    - Strict **leakage checks** (splits grouped by patient).
#    - Config-driven design using a Pydantic `Config` object.
#    - Random seeds, logging, and artifact saving (models, metrics, plots, CSV outputs).
#    - Hyperparameter search and **stability analysis** across multiple training runs.
# 
# ---
# 
# ## Notebook Workflow
# 
# The notebook is organized into logical sections that you can run sequentially:
# 
# ### 0. Environment & Repository Setup
# - Detects if the code is running on **Google Colab**.
# - Installs PyTorch Geometric dependencies (`torch-scatter`, `torch-sparse`, `pyg-lib`, `torch_geometric`) using wheels that match the detected PyTorch/CUDA version.
# - Optionally uses `git clone` to pull the `project273a` repository and adds it to `sys.path` so local Python modules (`utils`, `data`, `graph`, `train`, `evaluation`, etc.) can be imported.
# 
# ### 1. Configuration & System Setup
# - Defines a nested `config_dict` that controls the entire pipeline, including:
#   - **System settings**
#     Threading, dataloader workers, CUDA options (TF32, AMP, CuDNN), DDP flags, etc.
#   - **Data settings**
#     Paths to raw CSV files, target definition (`readmitted` → `readmitted_binary`), ID columns, filtering rules (e.g., exclude specific discharge IDs, keep first encounter per patient), numeric and categorical feature lists, ICD/drug columns, and preprocessing options (imputers, unknown category handling, ICD truncation).
#   - **Graph settings**
#     Enabled node/edge types, reverse edges, edge attributes (e.g., drug status), and neighbor sampling fanouts for each relation type.
#   - **Model settings**
#     Architecture (`arch`: "RGCN", "HGT", or "GraphSAGE"), hidden dimension, number of layers, number of heads (for HGT), RGCN bases, dropout, loss config (e.g. positive class weighting).
#   - **Training, inference, and evaluation**
#     Epochs, early stopping, batch size, optimizer (AdamW + LR, weight decay), logging frequency, metrics to compute, threshold-tuning strategy, and which plots to generate (ROC, PR, calibration, confusion matrix, decision curves).
#   - **Paths & artifacts**
#     Where to store checkpoints, TensorBoard logs, metrics CSVs, and prediction outputs.
# 
# - Constructs a strongly-typed `Config` object from this dictionary.
# - Applies system configuration (threads, CUDA flags) via `apply_system_config`.
# - Sets random seeds with `set_seed` for reproducibility.
# - Initializes structured logging and TensorBoard writers via `init_logging`.
# 
# ### 2. Data Loading, Splitting & Preprocessing
# - Loads the main encounter-level CSV (e.g., `diabetic_data.csv`) and auxiliary ID mapping files.
# - Applies dataset-level filters:
#   - Excluding certain discharge dispositions (e.g., hospice, expired),
#   - Keeping only the **first encounter per patient** to avoid leakage.
# - Creates **patient-grouped train/validation/test splits** using `create_splits`, optionally stratified by the binary target.
# - Runs:
#   - **Leakage checks** to ensure patients do not appear in multiple splits.
#   - **Target distribution checks** (`check_target_distribution`) to verify that class balance is reasonably stable across splits.
# - Separates the original DataFrame into `df_train`, `df_val`, and `df_test`.
# - Applies preprocessing (`preprocess.preprocess_data`):
#   - Imputes numeric and categorical values.
#   - Encodes categories (including “UNKNOWN” handling).
#   - Scales numeric features using statistics computed on the **training set only**.
# - Saves processed splits to CSV (`df_train.csv`, `df_val.csv`, `df_test.csv`) for external inspection or downstream use.
# 
# ### 3. Data Exploration & Vocabulary Construction
# - Uses a `DataExplorer` helper to:
#   - Generate a **summary table** of the dataset (means, standard deviations, missingness, etc.).
#   - Plot the **class balance** for the binary readmission target.
#   - Create **violin plots** to visualize target vs feature distributions.
# - Saves the summary table to `data_summary.csv`.
# - Builds vocabularies with `vocab.make_vocabs`:
#   - Encodes ICD codes, drugs, drug classes, specialties, admission types, etc. as integer IDs.
#   - Stores both vocabulary dictionaries and mappings.
# - Exports each vocabulary to individual CSV files (e.g., `vocab_icd.csv`, `vocab_drug.csv`) for transparency and reproducibility.
# 
# ### 4. Heterogeneous Graph Construction
# - Confirms **no patient overlap** across train/val/test.
# - Logs readmission rates for each split for sanity checking.
# - Uses `graph.builder.build_heterodata` to construct **PyTorch Geometric `HeteroData`** objects:
#   - **Node types** include encounters, ICD codes, ICD groups, drugs, drug classes, specialties, admission types, discharge dispositions, admission sources, etc.
#   - **Edge types** capture relations such as `encounter__has_icd__icd`, `icd__is_a__icd_group`, `encounter__has_drug__drug`, `drug__belongs_to__drug_class`, and their reverse counterparts.
#   - Edge attributes can encode medication status (e.g., up, down, steady).
# - Logs node counts and edge counts for each type for the training graph.
# 
# ### 5. GNN Model Definition & Training Loop
# - Extracts:
#   - Encounter input dimension (feature size),
#   - Vocabulary sizes for each non-encounter node type.
# - Creates the GNN using a model factory (`get_model_class`), driven entirely by the `config`:
#   - Default: **RGCN** with configurable bases, layers, hidden size, and dropout.
#   - Optionally **HGT** or **GraphSAGE** if configured.
# - Applies `setup_and_compile_model` for any compilation / optimization passes.
# - Builds:
#   - **NeighborLoaders** for mini-batching via multi-hop neighbor sampling (fanouts defined in `config.graph.fanouts`).
#   - Optimizer and learning-rate scheduler from `config.train.optimizer`.
#   - Loss function via `make_criterion` (with support for class weights / positive weighting).
# - Instantiates a `Trainer` object that encapsulates:
#   - One-epoch training loops (optionally with gradient clipping),
#   - Validation using AUPRC,
#   - Early stopping and best-model tracking,
#   - Logging and optional TensorBoard integration.
# - Runs the main training loop:
#   - For each epoch, trains over sampled batches,
#   - Periodically evaluates on the validation graph,
#   - Updates early-stopping logic and saves the best model parameters.
# - At the end, reloads the **best model state** for final evaluation.
# 
# ### 6. Evaluation, Threshold Tuning & Calibration
# - Uses the trained model to compute logits and probabilities on validation and test graphs.
# - Computes evaluation metrics via `compute_metrics`:
#   - **Primary**: AUPRC, AUROC, F1 for the positive class.
#   - **Secondary**: precision, recall, specificity, balanced accuracy, Brier score, log loss, etc.
# - Performs **threshold tuning** with `find_best_threshold`:
#   - Searches over thresholds (either a grid or automatically chosen),
#   - Optimizes for `f1_pos` by default,
#   - Applies the best threshold to the test set and computes F1/precision/recall.
# - Applies **probability calibration** using Platt scaling or isotonic regression:
#   - Fits a calibration model on validation probabilities,
#   - Applies it to test probabilities,
#   - Recomputes calibrated metrics.
# - Generates plots:
#   - ROC and PR curves,
#   - Calibration (reliability) plot,
#   - Confusion matrix at the tuned threshold,
#   - Decision curve showing net benefit vs threshold.
# 
# ### 7. GNN Hyperparameter Search
# - Defines a **hyperparameter grid** for the RGCN (hidden_dim, num_layers, dropout, number of bases, learning rate, weight decay, etc.).
# - Uses:
#   - `ParameterGrid` (sklearn) to enumerate configurations,
#   - `joblib.Parallel` with a custom `tqdm_joblib` wrapper for progress bars,
#   - Optional **CUDA MPS** integration for better GPU concurrency.
# - For each configuration:
#   - Creates a deep copy of the base `config` and updates nested keys like `model.hidden_dim`, `train.optimizer.lr`, etc.
#   - Trains a model for a capped number of epochs and tracks best validation AUPRC.
#   - Cleans up GPU/CPU memory aggressively after each trial.
# - Aggregates results into a DataFrame (`results_df`), sorts by `val_auprc`, and saves to `grid_search_results.csv`.
# - Performs **analysis of hyperparameters**:
#   - 1D effect plots (hyperparameter value vs mean AUPRC with error bars).
#   - 2D heatmaps (e.g., hidden_dim × num_layers, lr × weight_decay).
#   - Cluster analysis in hyperparameter space using KMeans, with cluster-level metrics and visualizations.
# - Prints and inspects the top-N configs and distribution of hyperparameters among the best trials.
# 
# ### 8. Tabular Baseline Models
# - Constructs tabular feature matrices from the preprocessed DataFrames:
#   - Numeric features,
#   - One-hot encoded low-cardinality categoricals (with `"oh__"` prefixes).
# - Trains and evaluates standard ML baselines (`train_and_eval_baselines`):
#   - Logistic regression, gradient boosting, random forests, XGBoost, CatBoost, etc. (depending on the project implementation).
#   - Reports test metrics comparable to the GNN metrics.
# - Runs a dedicated **grid search for tabular baselines** via `BaselineGridSearch`:
#   - Samples configs per model up to a specified limit (e.g., 30 per model).
#   - Produces a `tabular_report` with all results and the best configuration.
#   - Evaluates the best tabular model on the test set.
# - Visualizes baseline performance (e.g., bar charts of scores per model, score vs rank) with Plotly.
# - Saves tabular grid search results to `tabular_grid_search_results_report.csv`.
# 
# ### 9. Training Stability Study
# - Defines a simple factory for a logistic regression model.
# - Uses `RepeatedTrainingStudy` to:
#   - Train and evaluate the same model multiple times (`n_runs`) with different random seeds.
#   - Collect and summarize distributions of key metrics (AUROC, AUPRC, F1, precision, recall).
# - Displays:
#   - Per-run metrics,
#   - Summary statistics (mean, std, min, max) for each metric.
# - This helps understand **variance due to random initialization and stochastic training**, and provides a reference for how stable the models are.
# 
# ---
# 
# ## Outputs & Artifacts
# 
# Throughout the notebook, intermediate artifacts are written to disk so you can inspect or reuse them:
# 
# - **Preprocessed datasets**: `df_train.csv`, `df_val.csv`, `df_test.csv`
# - **Vocabularies**: `vocab_<type>.csv` (e.g., `vocab_icd.csv`, `vocab_drug.csv`)
# - **Exploratory summary**: `data_summary.csv`
# - **Model artifacts**:
#   - Checkpoints and metadata in `artifacts/`
#   - Best run info: `metrics_summary.csv`, `threshold_metrics.csv`, `test_predictions.csv`, etc.
# - **Hyperparameter search**:
#   - `grid_search_results.csv` for GNN
#   - `tabular_grid_search_results_report.csv` for baselines
# - **Stability analysis outputs** (if saved): per-run and summary metric tables
# 
# You can treat this notebook as both:
# - A **reproducible experiment notebook** for the diabetic readmission problem, and
# - A **template** for building heterogeneous GNN pipelines on other EHR-like tabular datasets with similar structure.
# 
# ---
#%% md
# # 0. Environment & Repository Setup
# 
# ## 0.1 Colab detection & PyG installation
# - Install torch-scatter, torch-sparse, torch-geometric, pyg-lib based on Torch/CUDA.
# - Recommended to have installed requirements.txt beforehand
# ## 0.2 Clone project repository
# - Clone `project273a` and add it to `sys.path`.
#%%
import sys

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
!pip install dcor
!pip install phik
!pip install tqdm_joblib
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
#%% md
# # 1. Configuration & System Setup
# 
# ## 1.1 Global Config (data, graph, model, train, evaluation)
# - Define nested `config_dict` for system, data, graph, model, training, evaluation, and paths.
# 
# ## 1.2 Apply system settings, logging, and seeding
# - Apply CPU/CUDA settings.
# - Initialize logging and TensorBoard writer.
# - Set random seeds.
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
            "group_by": "patient",    # group splits by patient_id to avoid leakage
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
#%% md
# # 2. Data Loading & Preprocessing
# 
# ## 2.1 Load raw CSV data and apply filters
# - Load `diabetic_data.csv` and `IDS_mapping.csv`.
# - Apply dataset filters (discharge exclusions, first encounter per patient).
# 
# ## 2.2 Train/Val/Test splits & leakage checks
# - Create patient-level splits.
# - Check for data leakage and target distribution stability.
# 
# ## 2.3 Preprocess features
# - Impute numeric/categorical features.
# - Scale / encode columns.
# - Save train/val/test splits to CSV.
#%%
from data.filters import apply_filters
from utils.io import load_csv
import data

# Load the datasets
df = load_csv(config.data.csv_path)
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

#%%
df_train.to_csv('df_train.csv', index=False)
df_val.to_csv('df_val.csv', index=False)

# Check if df_test exists before saving, as it might be None
if df_test is not None:
    df_test.to_csv('df_test.csv', index=False)
    print("df_train, df_val, and df_test saved to CSV files.")
else:
    print("df_train and df_val saved to CSV files. df_test was not available.")
#%% md
# # 3. Data Exploration
# 
# ## 3.1 Exploratory statistics & plots
# - Use `DataExplorer` for summary table, class balance, violin plots.
# - Save summary to CSV.
# 
# ## 3.2 Vocabulary construction
# - Build vocabularies for ICD codes, drugs, specialties, etc.
# - Export vocabularies to CSV.
#%%
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
data_summary.to_csv('data_summary.csv', index=False)
print("Data summary saved to 'data_summary.csv'")
#%%
from data import vocab

vocabs, mappings = vocab.make_vocabs(df_train, config)
logger.info("Vocab sizes: " + ", ".join(f"{k}: {len(v)}" for k,v in vocabs.items()))
#%%
import pandas as pd

# Assuming 'vocabs' is a dictionary of dictionaries (vocab_name: {item: id})
for vocab_name, vocab_dict in vocabs.items():
    # Convert the inner dictionary to a list of tuples (item, id)
    vocab_data = list(vocab_dict.items())

    # Create a pandas DataFrame
    df_vocab = pd.DataFrame(vocab_data, columns=['item', 'id'])

    # Define the filename and save to CSV
    filename = f'vocab_{vocab_name}.csv'
    df_vocab.to_csv(filename, index=False)
    print(f"Saved vocabulary '{vocab_name}' to '{filename}'")

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
#%% md
# # 4. Heterogeneous Graph Construction
# 
# ## 4.1 Sanity checks on patient splits
# - Verify no patient overlap between train/val/test.
# - Compare readmission rates per split.
# 
# ## 4.2 Build HeteroData graphs
# - Construct train/val/test heterogeneous graphs.
# - Log node and edge type statistics.
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
#%% md
# # 5. GNN Model Definition & Training
# 
# ## 5.1 Define model metadata & instantiate model
# - RGCN/HGT/GraphSAGE via `get_model_class`.
# - Compute input dims and vocab sizes.
# - Compile and move model to device.
# 
# ## 5.2 Train loop setup
# - Build NeighborLoaders.
# - Create optimizer, scheduler, criterion, Trainer.
# 
# ## 5.3 Training & early stopping
# - Epoch loop with validation AUPRC.
# - Save best state and artifacts.
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
best_state['val_auprc']
#%%
artifact_path = save_best_artifact(best_state, config, artifacts_dir="artifacts")
logger.info(f"Saved best artifact to: {artifact_path}")
#%%
from utils.artifacts import load_best_artifact

# Restore latest (or pass a specific run_id):
artifact = load_best_artifact(artifacts_dir="artifacts", run_id="latest", map_location="cpu")

# Access stored metadata:
print("Run:", artifact["run_id"])
print("Best epoch:", artifact["best_state"]["epoch"])
print("Best Val AUPRC:", artifact["best_state"]["val_auprc"])
restored_config = artifact["config"]  # JSON-serializable dict
#%% md
# # 6. Evaluation & Calibration
# 
# ## 6.1 Reload best artifact & evaluate metrics
# - Basic metrics on validation and test sets.
# - Threshold tuning for F1.
# 
# ## 6.2 Probability calibration
# - Platt / isotonic calibration.
# - Metrics after calibration.
# 
# ## 6.3 Plots
# - ROC, PR, calibration curve, confusion matrix.
# - Decision curve analysis.
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

#%%
from evaluation.model_evaluator import Evaluator
try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore[redefinition]
        print(obj)

test_data = [graph_test]
test_labels_np = graph_test["encounter"].y.cpu().numpy()
model.to("cpu")
model._device = torch.device("cpu")  # keep mixin in sync (optional but nice)

evaluator = Evaluator({"gnn_model": model})
evaluator.evaluate(test_data, test_labels_np)

evaluator = Evaluator({"gnn_model": model})
evaluator.evaluate(test_data, test_labels_np)

display(evaluator.metrics_summary_table())
display(evaluator.threshold_metrics_table(thresholds=(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45)))
#%%

#%%
# Save the metrics summary table to CSV
metrics_summary_df = evaluator.metrics_summary_table()
metrics_summary_df.to_csv('metrics_summary.csv', index=False)
print("Metrics summary table saved to 'metrics_summary.csv'")

# Save the threshold metrics table to CSV
threshold_metrics_df = evaluator.threshold_metrics_table()
threshold_metrics_df.to_csv('threshold_metrics.csv', index=False)
print("Threshold metrics table saved to 'threshold_metrics.csv'")
#%%

#%%
import pandas as pd

# Create a DataFrame from test_labels and test_probs
df_test_predictions = pd.DataFrame({
    'test_labels': test_labels,
    'test_probabilities': test_probs.flatten() # Flatten if test_probs is 2D
})

# Save the DataFrame to a CSV file
df_test_predictions.to_csv('test_predictions.csv', index=False)
print("Test labels and probabilities saved to 'test_predictions.csv'")
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

#%% md
# # 7. Hyperparameter Search for RGCN
# 
# ## 7.1 Grid definition & infrastructure
# - ParameterGrid over hidden_dim, num_layers, dropout, bases, lr, weight_decay.
# - CUDA MPS and joblib parallelization.
# - `tqdm_joblib` helper.
# 
# ## 7.2 Run grid search
# - Parallel train/eval for each config.
# - Collect and sort results, save to CSV.
# 
# ## 7.3 Hyperparameter analysis
# - 1D plots for each hyperparameter vs AUPRC.
# - 2D heatmaps (hidden_dim × num_layers, lr × weight_decay).
# - Clustering configs in hyperparameter space.
#%%
"""
Fast grid search for RGCNModel using joblib + tqdm_joblib + optional CUDA MPS
on the REAL diabetic heterogeneous graph.

Uses:
  - joblib.Parallel (threading backend)
  - tqdm_joblib progress bar
  - CUDA MPS env + nvidia-cuda-mps-control call

Relies on your existing:
  - graph_train, graph_val, config, rt, device
  - get_model_class, setup_and_compile_model, Trainer, make_neighbor_loader, etc.

Key RAM fixes:
  - Smaller batch size
  - DataLoader num_workers = 0 when using joblib parallelism
  - Aggressive cleanup at the end of each trial
"""

import os
import subprocess
import copy
import gc

import torch
import pandas as pd
from joblib import Parallel, delayed
import joblib.parallel as joblib_parallel
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
from contextlib import contextmanager

from train.model_factory import get_model_class, setup_and_compile_model
from train.optim import make_optimizer, make_scheduler
from train.losses import make_criterion
from train.loader import make_neighbor_loader
from data.sampling import build_num_neighbors
from train.loop import Trainer

# ---------------------------------------------------------------------
# Optional: NVIDIA CUDA MPS setup (safe to ignore on CPU)
# ---------------------------------------------------------------------
os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps"

if torch.cuda.is_available():
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("MPS control started (or already running).")
    except FileNotFoundError:
        print("MPS control binary not found; continuing without it.")

# ---------------------------------------------------------------------
# tqdm_joblib helper (progress bar for joblib.Parallel)
# ---------------------------------------------------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.

    Use as:
        with tqdm_joblib(tqdm(...)) as pbar:
            Parallel(...)(...)
    """
    OldBatchCompletionCallBack = joblib_parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallBack(OldBatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = OldBatchCompletionCallBack
        tqdm_object.close()

# ---------------------------------------------------------------------
# Reuse data + metadata from earlier cells
# ---------------------------------------------------------------------
metadata = (list(graph_train.node_types), list(graph_train.edge_types))
enc_input_dim = graph_train["encounter"].x.size(-1)
type_vocab_sizes = {
    nt: graph_train[nt].num_nodes
    for nt in graph_train.node_types
    if nt != "encounter"
}

# ---------------------------------------------------------------------
# Utility: set nested config attributes like "model.hidden_dim"
# ---------------------------------------------------------------------
def set_nested_attr(obj, dotted_key: str, value):
    """
    Example:
        set_nested_attr(cfg, "model.hidden_dim", 128)
        set_nested_attr(cfg, "train.optimizer.lr", 1e-3)
    """
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)

# ---------------------------------------------------------------------
# Hyperparameter grid (edit as you like)
# ---------------------------------------------------------------------
param_grid = {
    "model.hidden_dim": [128, 256, 300],                 # 2 valores
    "model.num_layers": [1, 2, 3],                     # 2 valores
    "model.dropout": [0.10, 0.15, 0.20, 0.25, 0.30],# 5 valores
    "model.rgcn_bases": [4, 6],                  # 3 valores
    "train.optimizer.lr": [5e-4, 7e-4, 9e-4, 1.1e-3, 1.3e-3],  # 5 valores
    "train.optimizer.weight_decay": [0.05, 0.01],    # 2 valores
    "train.epochs": [80],                           # 1 valor
}


parameter_grid = list(ParameterGrid(param_grid))

# ---------------------------------------------------------------------
# Single trial function run in each joblib worker (thread)
# ---------------------------------------------------------------------
def _run_single_trial(
    trial_index: int,
    param_dict: dict,
    base_config,
):
    """
    Train one RGCNModel with a given hyperparameter config and
    return metrics for this trial.
    """
    # ---- per-thread hygiene ----
    os.environ["OMP_NUM_THREADS"] = "2"   # a couple of CPU threads per trial
    torch.set_num_threads(2)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Local deep copy so we don't mutate the shared base_config
    cfg = copy.deepcopy(base_config)

    # Apply hyperparameters to this copy
    for key, value in param_dict.items():
        set_nested_attr(cfg, key, value)

    # Make each trial non-insane in RAM, but still use GPU decently
    if hasattr(cfg.train, "batching"):
        cfg.train.batching.batch_size_encounters = 2048  # or 1024 if needed

    # IMPORTANT: avoid extra processes inside joblib parallelism
    if hasattr(cfg, "system") and hasattr(cfg.system, "dataloader"):
        dl_cfg = cfg.system.dataloader
        dl_cfg.num_workers = 0
        # When num_workers == 0, prefetch_factor must be None
        dl_cfg.prefetch_factor = None
        dl_cfg.persistent_workers = False

    # --- Build model as in your main training code ---
    ModelClass = get_model_class(cfg)
    model = ModelClass(
        metadata,
        cfg,
        enc_input_dim=enc_input_dim,
        type_vocab_sizes=type_vocab_sizes,
    ).to(device)

    model = setup_and_compile_model(model, cfg, logger=None)

    # --- Optimizer / scheduler / loss ---
    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)
    criterion = make_criterion(graph_train, cfg, device)

    # --- Neighbor loader (depends on num_layers) ---
    num_layers = int(getattr(cfg.model, "num_layers", 2))
    num_neighbors = build_num_neighbors(graph_train, cfg, num_layers)

    train_loader = make_neighbor_loader(
        graph_train,
        input_nodes=("encounter", torch.arange(graph_train["encounter"].num_nodes)),
        num_neighbors=num_neighbors,
        config=cfg,
        train=True,
        shuffle=True,
    )

    val_data = graph_val.to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg,
        rt=rt,
        scheduler=scheduler,
        writer=None,
        logger=None,
        save_best_fn=lambda payload: None,
        early_stopping_patience=int(getattr(cfg.train, "early_stopping_patience", 5)),
        val_every=int(getattr(cfg.train, "val_every", 1)),
    )

    grad_clip = getattr(cfg.train, "gradient_clip_norm", None)
    # cap epochs for “fast” search
    max_epochs = min(int(getattr(cfg.train, "epochs", 30)), 10)

    best_val_auprc = -float("inf")

    for epoch in range(1, max_epochs + 1):
        trainer.train_epoch(train_loader, grad_clip=grad_clip)

        # validate every epoch (you could also respect cfg.train.val_every here)
        val_auprc = trainer.validate_auprc(val_data)

        # let Trainer handle best-state tracking + patience
        early_stop = trainer.update_after_validation(val_auprc, epoch=epoch)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc

        if early_stop:
            break

    # ---- cleanup to reduce RAM ----
    try:
        del train_loader
        del val_data
        del trainer
        del model
        del optimizer
        del scheduler
        del criterion
        del num_neighbors
    except NameError:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    result = dict(param_dict)
    result["val_auprc"] = float(best_val_auprc)
    result["trial"] = trial_index
    return result

# ---------------------------------------------------------------------
# Run the grid search with joblib + tqdm_joblib
# ---------------------------------------------------------------------
base_config = copy.deepcopy(config)

# With one A100 + ~12 CPU cores, 1–2 parallel trials is usually best.
n_jobs = 3  # set to 1 if you still see high RAM

print(f"Running RGCN grid search with {len(parameter_grid)} configs, n_jobs={n_jobs} ...")

with tqdm_joblib(
    tqdm(total=len(parameter_grid), desc=f"Grid search (n_jobs={n_jobs})", unit="config")
):
    results = Parallel(
        n_jobs=n_jobs,
        backend="threading",
        batch_size=1,
        verbose=0,
    )(
        delayed(_run_single_trial)(
            i,
            params,
            base_config,
        )
        for i, params in enumerate(parameter_grid)
    )

results_df = (
    pd.DataFrame(results)
    .sort_values("val_auprc", ascending=False)
    .reset_index(drop=True)
)

# Show top results
try:
    from IPython.display import display
    display(results_df.head())
except Exception:
    print(results_df.head())

best_row = results_df.iloc[0].to_dict()
best_score = best_row.pop("val_auprc")
print("\nBest hyperparameters:")
for k, v in best_row.items():
    print(f"  {k}: {v}")
print(f"\nBest validation AUPRC: {best_score:.4f}")
#%%

#%%
results_df.to_csv('grid_search_results.csv', index=False)
print("Grid search results saved to 'grid_search_results.csv'")
#%%
import numpy as np

hyperparams = [
    "model.hidden_dim",
    "model.num_layers",
    "model.dropout",
    "model.rgcn_bases",
    "train.optimizer.lr",
    "train.optimizer.weight_decay",
]

for hp in hyperparams:
    if hp not in results_df.columns:
        continue

    agg = (
        results_df
        .groupby(hp)["val_auprc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print(f"\n=== {hp} ===")
    print(agg)

    x = agg[hp].values
    y = agg["mean"].values
    yerr = agg["std"].fillna(0).values

    plt.figure()
    # Handle numeric vs categorical x-axis
    if np.issubdtype(np.array(x).dtype, np.number):
        plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-")
    else:
        positions = np.arange(len(x))
        plt.errorbar(positions, y, yerr=yerr, marker="o", linestyle="-")
        plt.xticks(positions, x, rotation=45)

    plt.xlabel(hp)
    plt.ylabel("Mean val AUPRC")
    plt.title(f"Effect of {hp} on validation AUPRC")
    plt.tight_layout()
    plt.show()

#%%
hp1 = "model.hidden_dim"
hp2 = "model.num_layers"

pivot = results_df.pivot_table(
    index=hp1,
    columns=hp2,
    values="val_auprc",
    aggfunc="mean",
)

print("\nMean AUPRC for hidden_dim × num_layers:")
display(pivot)

plt.figure()
plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.colorbar(label="Mean val AUPRC")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel(hp2)
plt.ylabel(hp1)
plt.title("Hidden dim × Num layers (mean val AUPRC)")
plt.tight_layout()
plt.show()

#%%
hp1 = "train.optimizer.lr"
hp2 = "train.optimizer.weight_decay"

pivot = results_df.pivot_table(
    index=hp1,
    columns=hp2,
    values="val_auprc",
    aggfunc="mean",
)

print("\nMean AUPRC for lr × weight_decay:")
display(pivot)

plt.figure()
plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.colorbar(label="Mean val AUPRC")
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel(hp2)
plt.ylabel(hp1)
plt.title("LR × Weight decay (mean val AUPRC)")
plt.tight_layout()
plt.show()

#%%
results_df
#%%
top_n = 20
top = results_df.nlargest(top_n, "val_auprc")

print(f"\nTop {top_n} configs:")
display(top)

print("\nHyperparameter distributions in top configs:")
for hp in hyperparams:
    if hp not in top.columns:
        continue
    print(f"\n{hp}:")
    print(top[hp].value_counts())
#%%
top_n = 20
top = results_df.nlargest(top_n, "val_auprc")

print(f"\nTop {top_n} configs:")
display(top)

print("\nHyperparameter distributions in top configs:")
for hp in hyperparams:
    if hp not in top.columns:
        continue
    print(f"\n{hp}:")
    print(top[hp].value_counts())

#%%
top_n = 20
top = results_df.nlargest(top_n, "val_auprc")

print(f"\nTop {top_n} configs:")
display(top)

print("\nHyperparameter distributions in top configs:")
for hp in hyperparams:
    if hp not in top.columns:
        continue
    print(f"\n{hp}:")
    print(top[hp].value_counts())

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Columns to use for clustering (all numeric already)
hp_cols = [
    "model.hidden_dim",
    "model.num_layers",
    "model.dropout",
    "model.rgcn_bases",
    "train.optimizer.lr",
    "train.optimizer.weight_decay",
]

# Just a safety check
for col in hp_cols:
    if col not in results_df.columns:
        raise ValueError(f"Column {col} is missing from results_df")

X = results_df[hp_cols].values

# Standardize hyperparameters so different scales don't dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: elbow plot to pick K
inertias = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia")
plt.title("KMeans elbow plot (hyperparameter space)")
plt.show()

#%%
K = 4  # adjust based on the elbow plot

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
results_df["cluster"] = kmeans.fit_predict(X_scaled)

print("Cluster sizes:")
print(results_df["cluster"].value_counts())

#%%
cluster_stats = (
    results_df
    .groupby("cluster")["val_auprc"]
    .agg(["mean", "std", "min", "max", "count"])
    .sort_values("mean", ascending=False)
)

print("Cluster-wise AUPRC stats:")
display(cluster_stats)

plt.figure()
plt.bar(cluster_stats.index, cluster_stats["mean"], yerr=cluster_stats["std"])
plt.xlabel("Cluster")
plt.ylabel("Mean val AUPRC")
plt.title("Mean validation AUPRC by cluster")
plt.show()

#%%
print("Cluster-wise mean hyperparameters:")
display(
    results_df
    .groupby("cluster")[hp_cols + ["val_auprc"]]
    .mean()
    .sort_values("val_auprc", ascending=False)
)

print("Cluster-wise median hyperparameters:")
display(
    results_df
    .groupby("cluster")[hp_cols + ["val_auprc"]]
    .median()
    .sort_values("val_auprc", ascending=False)
)

#%%
pairs = [
    ("model.hidden_dim", "model.num_layers"),
    ("train.optimizer.lr", "train.optimizer.weight_decay"),
    ("model.dropout", "model.rgcn_bases"),
]

for x_col, y_col in pairs:
    plt.figure()
    for c in sorted(results_df["cluster"].unique()):
        subset = results_df[results_df["cluster"] == c]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            alpha=0.7,
            label=f"cluster {c}",
        )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if "lr" in x_col or "weight_decay" in x_col:
        plt.xscale("log")
    if "lr" in y_col or "weight_decay" in y_col:
        plt.yscale("log")
    plt.title(f"{x_col} vs {y_col} colored by cluster")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% md
# # 8. Tabular Baseline Models
# 
# ## 8.1 Feature construction for tabular models
# - Construct numeric + one-hot features from preprocessed dfs.
# 
# ## 8.2 Train/evaluate baselines
# - Logistic regression, tree ensembles, etc.
# - Compare test metrics.
# 
# ## 8.3 Baseline grid search
# - Run `BaselineGridSearch` on tabular models.
# - Plot scores and ranks; save results.
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
print("Starting")
# Train and evaluate baseline models
baseline_results = train_and_eval_baselines(X_train_tab, y_train_tab, X_val_tab, y_val_tab, X_test_tab, y_test_tab, config=config)
print("Finishing")
# Display baseline evaluation results
for model_name, metrics_dict in baseline_results.items():
    test_met = metrics_dict["test"]
#%%
from grid_search.tabular import BaselineGridSearch

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore[redefinition]
        print(obj)

tabular_grid = BaselineGridSearch()
tabular_report = tabular_grid.run(
    X_train_tab,
    y_train_tab,
    X_val=X_val_tab,
    y_val=y_val_tab,
    sample_size=45000,
    max_configs_per_model=30
)

#%%

display(tabular_report.results)
print(
    "Best tabular baseline:",
    tabular_report.best_params,
    f"score={tabular_report.best_score:.4f}",
)

best_tab_metrics = tabular_grid._evaluate(  # pylint: disable=protected-access
    tabular_report.best_model,
    X_test_tab,
    y_test_tab,
)
print("Best model test metrics:")
for metric_name, metric_value in best_tab_metrics.items():
    print(f"  {metric_name}: {metric_value:.4f}")

import plotly.express as px

score_fig = px.bar(
    tabular_report.results,
    x="model",
    y="score",
    color="model",
    title="Tabular Baseline Grid Search Scores",
    hover_data=[
        column
        for column in tabular_report.results.columns
        if column not in {"model", "score"}
    ],
)
score_fig.show()

if {"score", "model"}.issubset(tabular_report.results.columns):
    ranked_results = tabular_report.results.copy()
    ranked_results["rank"] = ranked_results.index + 1
    rank_fig = px.scatter(
        ranked_results,
        x="rank",
        y="score",
        color="model",
        title="Score by Rank for Tabular Baselines",
        hover_data=[
            column
            for column in ranked_results.columns
            if column not in {"rank", "score"}
        ],
    )
    rank_fig.show()
#%%
tabular_report.results
#%%
import pandas as pd

tabular_report.results.to_csv('tabular_grid_search_results_report.csv', index=False)
print("Tabular grid search results report saved to 'tabular_grid_search_results_report.csv'")
#%% md
# # 9. Training Stability Study
# 
# ## 9.1 Logistic regression repeated runs
# - RepeatedTrainingStudy on tabular model.
# - Summarize distribution of AUROC, AUPRC, F1, precision, recall.
# 
# ## 9.2 Save summaries
# - Export runs and summary tables to CSV for reporting.
# 
#%%
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
#%%
