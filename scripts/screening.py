#!/usr/bin/env python3
"""
Screen all graphs with a trained model; supports MC Dropout for uncertainty.
Outputs results/top_candidates.csv sorted by target (first column) asc.
"""
import argparse, os, csv
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from scripts.models.cgcnn import CGCNN
from utils.data_loading import GraphFolderDataset

def mc_dropout_predictions(model, batch, n_samples=20):
    model.train()  # enable dropout
    preds = []
    for _ in range(n_samples):
        preds.append(model(batch).detach().cpu().numpy())
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cgcnn_model.pt")
    ap.add_argument("--graphs", default="graphs")
    ap.add_argument("--samples", type=int, default=0, help="MC dropout samples (0 disables)")
    ap.add_argument("--out", default="results/top_candidates.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = GraphFolderDataset(args.graphs)
    dl = DataLoader(ds, batch_size=64)

    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt.get("cfg", {})
    n_outputs = len(ds.targets)
    model = CGCNN(n_outputs=n_outputs,
                  hidden=int(cfg.get("hidden_dim",128)),
                  n_conv=int(cfg.get("num_conv_layers",3)),
                  dropout=float(cfg.get("dropout",0.1))).to(device)
    model.load_state_dict(ckpt["model"])

    rows = []
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            if args.samples > 0:
                preds = mc_dropout_predictions(model, b, n_samples=args.samples)
                import numpy as np
                mean = np.stack(preds, axis=0).mean(axis=0)
                std  = np.stack(preds, axis=0).std(axis=0)
                pred = torch.tensor(mean)
                unc  = torch.tensor(std)
            else:
                model.eval()
                pred = model(b).cpu()
                unc = torch.zeros_like(pred)

            # collect ids & formulas
            mids = getattr(b, "material_id", None)
            if mids is None:
                mids = [f"idx_{i}" for i in range(b.num_graphs)]
            # PyG batches don't carry arbitrary lists by default; build from dataset
            # We'll pull from indices stored in batch.ptr if needed
            # Simpler: rely on ds.meta via slicing order
            pass

    # Since collecting IDs from batched object is non-trivial, simpler approach:
    # Re-iterate sample-wise using DataLoader with batch_size=1 to capture attributes properly.
    model.eval()
    per_rows = []
    single_loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data in single_loader:
            data = data.to(device)
            if args.samples > 0:
                preds = mc_dropout_predictions(model, data, n_samples=args.samples)
                import numpy as np
                mean = float(np.mean(preds))
                std  = float(np.std(preds))
            else:
                mean = float(model(data).cpu().item())
                std  = 0.0
            mid = getattr(data, "material_id", ["unknown"])[0]
            formula = getattr(data, "formula", ["unknown"])[0]
            row = {"material_id": mid, "formula": formula, "pred": mean, "uncertainty": std}
            per_rows.append(row)

    df = pd.DataFrame(per_rows)
    df = df.sort_values("pred", ascending=True).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(df)} entries to {args.out}")

if __name__ == "__main__":
    main()
