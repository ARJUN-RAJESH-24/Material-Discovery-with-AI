#!/usr/bin/env python3
"""
Tiny demo: load first 64 graphs and run a few epochs.
Useful to sanity-check the pipeline quickly.
"""
import torch
from torch_geometric.loader import DataLoader
from scripts.models.cgcnn import CGCNN
from utils.data_loading import GraphFolderDataset

def main():
    ds = GraphFolderDataset("graphs", max_items=64)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = CGCNN(n_outputs=len(ds.targets)).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()
    for epoch in range(3):
        tot = 0
        for b in dl:
            opt.zero_grad()
            pred = model(b)
            loss = loss_fn(pred, b.y.squeeze(-1))
            loss.backward(); opt.step()
            tot += loss.item()*b.num_graphs
        print(f"Epoch {epoch+1}: MAE={tot/len(ds):.4f}")
    print("✅ Demo finished.")
if __name__ == "__main__":
    main()
