# Diabetes Readmission Prediction with Heterogeneous Graph Neural Network

This repository contains a full pipeline for predicting 30-day readmission of diabetic patients using heterogeneous graph neural networks (HGT, R-GCN, GraphSAGE) implemented with PyTorch Geometric. The notebook `main.ipynb` orchestrates the end-to-end workflow while the supporting Python packages under `data/`, `graph/`, `models/`, `train/`, and `evaluation/` let you script and reuse each stage independently.

Final reports and figures saved in a dedicated reports/ folder. (Figures, csv, configs, final report)
## Requirements
Python 3.10+ is recommended. Install dependencies with:

```
python -m pip install -r requirements.txt
```

Key packages include:

- PyTorch + PyTorch Geometric (and optional DGL neighbor loaders). Uncomment the wheel index in `requirements.txt` that matches your CUDA/Torch combo before installing.
- Scientific Python stack: numpy, pandas, scipy, scikit-learn, matplotlib, tensorboard, numexpr, joblib, tqdm, tqdm-joblib.
- Configuration & utilities: pydantic (v1), omegaconf.
- Visualisation & data exploration: plotly, dcor, phik.
- Baseline/tabular models: xgboost, catboost.

**Note**: PyTorch Geometric requires CUDA-specific wheels. Follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html if you need GPU support.

## Repository Structure
* `main.ipynb` / `main.py` – orchestrate the full experiment (data prep → graph building → model training → evaluation).
* `data/` – loaders, filters, preprocessing, vocabularies, and split helpers.
* `graph/` – hetero graph builders plus inductive inference helpers.
* `models/` – HGT, R-GCN, and GraphSAGE backbones plus classification heads.
* `train/` – data loaders, optimizer utilities, and the AMP-aware training loop.
* `evaluation/` – calibration, metrics, reporting utilities, and rich data exploration plots.
* `benchmarks/` – fast tabular baselines (logistic regression, random forest, XGBoost, CatBoost, KNN).
* `grid_search/` – reusable grid-search runners for graph models and tabular models.
* `infer/` – helpers for batch inference on new encounters.
* `utils/` – config schema, logging, artifact helpers, IO utilities, and system tuning helpers.
* `tests/` – lightweight unit tests for data vocabularies, graph builders, sampling, and evaluation helpers.
* `raw/` & `data/` – expected locations for the CSV inputs and cached/intermediate artifacts.

## How to Run
1. Download the public diabetes readmission dataset plus `IDS_mapping.csv` and place them under `raw/` (see the defaults in `config_dict` inside `main.ipynb`).
2. Adjust the configuration JSON in Section 2 of the notebook (or inside `main.py`) to point to your data files, tweak graph settings, and pick a backbone architecture.
3. Run `main.ipynb` in Jupyter/Colab (or mirror the sequence inside your own script). The numbered cells:
   - Load the CSVs, filter encounters, and build train/validation/test splits that respect patient grouping.
   - Preprocess features (imputation, rare-category handling, one-hot encoding) and persist scalers/encoders.
   - Construct heterogeneous graphs via `graph.builder.build_heterodata` for each split.
   - Instantiate a GNN from `models/` and train it with the AMP-aware `train.loop.Trainer`, logging metrics to TensorBoard.
   - Evaluate predictions with `evaluation.evaluation` (metrics, threshold tuning, calibration, subgroup reports) and generate interactive figures from `evaluation.model_evaluator` / `evaluation.data_exploration`.
   - Optionally train tabular baselines (`benchmarks/train_and_eval_baselines`) for comparison.
4. Artifacts (model checkpoints, configs, scalers, calibration models, and plots) are stored under `artifacts/`. `utils.artifacts.save_best_artifact` keeps a `latest` pointer for quick reuse.
5. To drive the pipeline from a pure Python script, import `Config` from `utils.config`, call the functions in `data/` and `graph/`, and instantiate `train.loop.Trainer` with loaders from `train/loader.py`.

## Inductive Inference
`graph.inductive.build_star_graph_for_row` reuses the same vocabularies to build a star graph around a single encounter. This lets you score unseen patients by loading the saved vocabs, creating the star graph on-the-fly, and running the trained GNN head. The `infer/` package contains helpers for batching predictions and exporting CSVs.

## Hyper-parameter search & baselines
- `grid_search/tabular.py` and `grid_search/rgcn_fast.py` provide joblib+tqdm powered sweeps with optional CUDA MPS helpers. Configure them by passing the `Config` object and a data factory callable.
- `benchmarks/baselines.py` trains Logistic Regression, Random Forest, XGBoost, CatBoost, and KNN references to contextualise GNN performance.

## Running Tests
Basic unit tests are provided in the `tests/` directory. You can run these tests to verify that:
- Unknown token handling in vocab works (`test_vocabs.py`).
- Graph builder creates expected nodes/edges and adds reverse edges correctly (`test_graph_builder.py`).
- Group splits have no patient overlap (`test_splits.py`).

Run the full suite with:

```
pytest tests
```

## Notes
This notebook and codebase are designed for clarity and completeness. For actual production use, some optimization and tuning might be necessary. The model variants (HGT, R-GCN, GraphSAGE) are all implemented; you can switch the `model.arch` in the config to try different GNN types. Calibration and threshold selection are performed on validation data to optimize final performance.
