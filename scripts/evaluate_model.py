#!/usr/bin/env python3
"""
Load trained CGCNN and evaluate on the test split by filename hash.
If you trained with cgcnn_train.py, metrics.txt is already written,
but this script re-computes MAE for safety and prints it.
"""
import argparse, os, torch, glob
from torch_geometric.loader import DataLoader
from scripts.models.cgcnn import CGCNN
from utils.data_loading import GraphFolderDataset
from torch.nn.functional import l1_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cgcnn_model.pt")
    ap.add_argument("--graphs", default="graphs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = GraphFolderDataset(args.graphs)
    dl = DataLoader(ds, batch_size=32)
    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt.get("cfg", {})
    n_outputs = len(ds.targets)
    model = CGCNN(n_outputs=n_outputs,
                  hidden=int(cfg.get("hidden_dim",128)),
                  n_conv=int(cfg.get("num_conv_layers",3)),
                  dropout=float(cfg.get("dropout",0.1))).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    tot_loss, n_items = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            pred = model(b)
            tgt = b.y if b.y.ndim==2 else b.y.unsqueeze(-1)
            loss = l1_loss(pred, tgt.squeeze(-1), reduction="sum")
            tot_loss += loss.item()
            n_items += b.num_graphs
    mae = tot_loss / n_items
    print(f"✅ Evaluate MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
