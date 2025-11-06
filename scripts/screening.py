from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rich import print
from torch_geometric.data import Data

from scripts.models.cgcnn import CGCNN


def load_graph(path: Path) -> Data:
    return torch.load(path)


def main():
    parser = argparse.ArgumentParser(description="Screen graphs with a trained CGCNN model")
    parser.add_argument("--graphs-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/top_candidates.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    graphs_dir = Path(args.graphs_dir)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = CGCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    rows = []
    for gpath in sorted(graphs_dir.glob("*.pt")):
        data = load_graph(gpath).to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_attr, torch.zeros(data.x.size(0), dtype=torch.long, device=device))
        rows.append({"id": gpath.stem, "prediction": float(pred.item())})

    # Sort by predicted value ascending (e.g., lower formation energy is better); users can invert if needed.
    rows.sort(key=lambda r: r["prediction"]) 

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[green]Wrote rankings to[/green] {str(out_path)} | n={len(rows)}")


if __name__ == "__main__":
    main()


