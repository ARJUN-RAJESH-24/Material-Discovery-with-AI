#!/usr/bin/env python3
"""
Train CGCNN using graphs/ as dataset cache and configs/config.json for hyperparams.
Supports single-task or multi-task (len(targets) > 1).
"""
import argparse, json, os, glob, random
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from scripts.models.cgcnn import CGCNN
from utils.data_loading import GraphFolderDataset
from utils.config import load_config
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)
        target = batch.y if batch.y.ndim == 2 else batch.y.unsqueeze(-1)
        loss = loss_fn(pred, target.squeeze(-1))
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * batch.num_graphs
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y if batch.y.ndim == 2 else batch.y.unsqueeze(-1)
        loss = loss_fn(pred, target.squeeze(-1))
        epoch_loss += loss.item() * batch.num_graphs
    return epoch_loss / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--graphs", default="graphs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    graphs_path = cfg.get("graph_path", args.graphs)
    targets = cfg.get("targets", ["formation_energy_per_atom"])
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("epochs", 50))
    hidden = int(cfg.get("hidden_dim", 128))
    n_conv = int(cfg.get("num_conv_layers", 3))
    dropout = float(cfg.get("dropout", 0.1))
    model_save = cfg.get("model_save_path", "models/cgcnn_model.pt")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = GraphFolderDataset(graphs_path, targets=targets)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=batch_size)

    n_outputs = len(targets)
    model = CGCNN(n_outputs=n_outputs, hidden=hidden, n_conv=n_conv, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # MAE

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d} | train MAE: {tr:.4f} | val MAE: {va:.4f}")
        if va < best_val:
            best_val = va
            os.makedirs(os.path.dirname(model_save), exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, model_save)

    # Final test
    ckpt = torch.load(model_save, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = eval_epoch(model, test_loader, loss_fn, device)
    print(f"✅ Test MAE: {te:.4f}")
    with open("results/metrics.txt", "w") as f:
        f.write(f"val_mae={best_val:.6f}\n")
        f.write(f"test_mae={te:.6f}\n")
    print("Saved results/metrics.txt")

if __name__ == "__main__":
    main()
