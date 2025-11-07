#!/usr/bin/env bash
set -e
echo "▶️ Evaluating model"
python scripts/evaluate_model.py --model models/cgcnn_model.pt --graphs graphs
echo "✅ Evaluation complete"
