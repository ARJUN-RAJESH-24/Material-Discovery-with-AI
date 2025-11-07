#!/usr/bin/env bash
set -e
echo "▶️ Training CGCNN"
python scripts/prepare_dataset.py --infile data/sample_mini.pkl --outfile data/processed.pkl --dropna
python scripts/graph_preprocessing.py --infile data/processed.pkl --outdir graphs --cutoff 8.0 --targets formation_energy_per_atom band_gap
python scripts/cgcnn_train.py --config configs/config.json
echo "✅ Training complete"
