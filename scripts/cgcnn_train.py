from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich import print
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from scripts.models.cgcnn import CGCNN


class GraphFolderDataset(Dataset):
    def __init__(self, folder: str | Path):
        super().__init__()
        self.folder = Path(folder)
        self.paths = sorted(list(self.folder.glob("*.pt")))

    def len(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def get(self, idx: int) -> Data:  # type: ignore[override]
        return torch.load(self.paths[idx])


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if getattr(batch, "y", None) is None:
            raise RuntimeError("Training graphs must include 'y'")
        y = batch.y.view(-1).to(device)
        loss = torch.nn.functional.l1_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(-1).to(device)
        loss = torch.nn.functional.l1_loss(pred, y)
        total += float(loss.item()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train CGCNN on PyG graphs")
    parser.add_argument("--graphs-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = GraphFolderDataset(args.graphs_dir)
    if len(dataset) == 0:
        raise RuntimeError("No .pt graphs found. Run scripts/graph_preprocessing.py first.")

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_test = max(1, int(0.1 * n_total))
    n_train = max(1, n_total - n_val - n_test)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CGCNN()
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
        print(f"Epoch {epoch:03d} | train MAE: {train_mae:.4f} | val MAE: {val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save({"model": model.state_dict(), "epoch": epoch}, model_dir / "best.pt")

    test_mae = evaluate(model, test_loader, device)
    print(f"[bold green]Test MAE[/bold green]: {test_mae:.4f}")

    with open(results_dir / "cgcnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_mae": best_val, "test_mae": test_mae, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()


