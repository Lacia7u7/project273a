# Diabetes Readmission Prediction with Heterogeneous Graph Neural Network

This repository contains a full pipeline for predicting 30-day readmission of diabetic patients using a heterogeneous graph neural network (GNN) with PyTorch Geometric. The pipeline is orchestrated in a single Jupyter Notebook (`main.ipynb`) and uses a modular `src/` codebase for data processing, model definition, training, and evaluation.

## Requirements
Install the required packages (see `requirements.txt`). Key libraries include:
- Python 3.x
- PyTorch (with CUDA support if available)
- PyTorch Geometric (matching your PyTorch/CUDA version)
- scikit-learn, pandas, numpy, scipy
- torchmetrics, tensorboard, tqdm, pyarrow, rich

**Note**: Ensure PyG is installed properly (refer to PyG installation instructions for the correct CUDA wheels).

## Repository Structure
* main.ipynb # Notebook orchestrating the entire pipeline
* README.md
* requirements.txt
* src/
* data/ ... # Data loading, preprocessing, vocab, mapping, splits
* graph/ ... # Graph construction, inductive subgraph, sampling
* models/ ... # Heterogeneous GNN model definitions (HGT, RGCN, GraphSAGE) and heads
* train/ ... # Training loop, metrics, calibration, threshold tuning
* infer/ ... # Batch prediction for new data
* utils/ ... # Config schema, logging, seeding, I/O utilities
* tests/ ... # Basic tests for vocabs, graph builder, splits


## How to Run
1. Open `main.ipynb` in Jupyter. It contains numbered sections with explanations and code.
2. **Configuration**: The notebook includes a JSON config cell (`Section 2`) where you can adjust paths and parameters. By default it expects `diabetic_data.csv` and `IDS_mapping.csv` in appropriate paths.
3. **Run Pipeline**: Execute the notebook cells in order. The pipeline will:
   - Load and preprocess data (filter out certain encounters, handle missing values, encode features).
   - Split data into train/validation/test ensuring no leakage (grouped by patient).
   - Construct a heterogeneous graph for each split with nodes for encounters, diagnoses (ICDs and groups), medications (drugs and classes), hospital, specialty, admission/discharge types, etc., and edges linking them.
   - Train a heterogeneous GNN (HGT by default) with neighbor sampling and early stopping.
   - Evaluate on validation/test sets with metrics: AUROC, AUPRC, F1, precision/recall, balanced accuracy, Brier score, ECE.
   - Perform probability calibration (Platt or isotonic) and threshold tuning to optimize F1.
   - Plot ROC, PR, calibration curve, confusion matrix, and decision curves. Compute subgroup metrics by age, race, gender, hospital.
   - Train baseline tabular model (MLP) for comparison and evaluate similarly.
   - Demonstrate inductive inference: predicting readmission for new encounters by building a star subgraph and using the trained model.
4. **TensorBoard Logs**: Training and evaluation metrics are logged to TensorBoard (log directory configurable, default `./tb_logs`). Launch TensorBoard to monitor training progress and view plots.
5. **Artifacts**: All important artifacts (model checkpoint, scalers, vocabularies, thresholds, calibration model, metrics and plots) are saved under `artifacts/` directory for reuse.

## Inductive Inference
The pipeline supports inductive inference. Given a new patient encounter (not seen during training), the code will construct a star graph connecting the encounter to relevant entity nodes (ICDs, drugs, etc., using `"UNKNOWN"` nodes for unseen categories) and then apply the GNN to generate a prediction. Results for batch inference can be saved to a CSV.

## Running Tests
Basic unit tests are provided in the `tests/` directory. You can run these tests to verify that:
- Unknown token handling in vocab works (`test_vocabs.py`).
- Graph builder creates expected nodes/edges and adds reverse edges correctly (`test_graph_builder.py`).
- Group splits have no patient overlap (`test_splits.py`).

## Notes
This notebook and codebase are designed for clarity and completeness. For actual production use, some optimization and tuning might be necessary. The model variants (HGT, R-GCN, GraphSAGE) are all implemented; you can switch the `model.arch` in the config to try different GNN types. Calibration and threshold selection are performed on validation data to optimize final performance.
