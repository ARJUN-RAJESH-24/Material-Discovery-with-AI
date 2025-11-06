# Material Discovery with AI (Deep Learning + Graph ML)

A research-ready pipeline to learn structure–property relationships from crystalline materials using classical ML and Graph Neural Networks (CGCNN). The system enables property prediction (formation energy, band gap, mechanical moduli) and screening of thousands of candidates.

## Directory Structure

```
data/               # raw and processed datasets
graphs/             # graph-structured data for GNNs
models/             # trained model weights and checkpoints
notebooks/          # step-by-step Jupyter workflows
results/            # figures, plots, predictions, metrics
scripts/            # core python code (preprocessing, models, training)
utils/              # helper functions and configuration
```

## Quickstart

1) Create environment and install dependencies

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell on Windows
pip install -U pip
pip install -r requirements.txt
```

2) Install torch-geometric (PyG) per your CUDA/CPU setup

- Find the right command at `https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html`.
- Example (CPU-only):

```bash
pip install torch-geometric
```

3) (Optional) Setup Materials Project API for data download

```powershell
setx MAPI_KEY "<your_api_key_here>"
```

## Data Assumptions

- Tabular file (CSV/Parquet) with at least columns:
  - `material_id`, `cif` or `structure` (CIF text preferred)
  - Targets such as `formation_energy_per_atom`, `band_gap`, `bulk_modulus`, `shear_modulus`
- Place your files under `data/`.

## Baseline ML

Train a classical ML baseline with composition features:

```bash
python scripts/baseline_ml.py \
  --dataset data/materials.csv \
  --target formation_energy_per_atom \
  --model xgboost \
  --output results/baseline_formation_energy
```

Outputs:
- Fitted model (`.joblib`)
- Metrics JSON
- Pred vs Actual plot

## Graph Preprocessing

Convert CIF structures into PyTorch Geometric graphs:

```bash
python scripts/graph_preprocessing.py \
  --dataset data/materials.csv \
  --structure-col cif \
  --radius 8.0 \
  --max-neighbors 12 \
  --out-dir graphs/
```

## CGCNN Training

Train CGCNN on the prepared graphs:

```bash
python scripts/cgcnn_train.py \
  --graphs-dir graphs/ \
  --target formation_energy_per_atom \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --model-dir models/cgcnn_fe \
  --results-dir results/
```

## Notebooks

- `notebooks/baseline_ml.ipynb` – End-to-end baseline workflow
- `notebooks/evaluation.ipynb` – Metrics, parity plots, error analysis
- `notebooks/screening.ipynb` – Predict and rank new/hypothetical materials

## Notes

- Torch Geometric installs differ by CUDA version; consult their docs for the exact wheel.
- For large datasets, prefer Parquet over CSV and enable `--num-workers` where applicable.
- Extend with alternative models (MEGNet, SchNet, ALIGNN) as needed.

## License

MIT

