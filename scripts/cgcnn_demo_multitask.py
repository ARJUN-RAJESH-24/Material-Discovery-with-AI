#!/usr/bin/env python3
"""
Multitask demo (e.g., formation_energy_per_atom + band_gap).
Requires graphs built with y having 2 columns.
"""
import torch
from torch_geometric.loader import DataLoader
from scripts.models.cgcnn import CGCNN
from utils.data_loading import GraphFolderDataset

def main():
    ds = GraphFolderDataset("graphs", targets=["formation_energy_per_atom","band_gap"], max_items=128)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = CGCNN(n_outputs=2).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()
    for epoch in range(5):
        tot = 0
        for b in dl:
            opt.zero_grad()
            pred = model(b)
            loss = loss_fn(pred, b.y)
            loss.backward(); opt.step()
            tot += loss.item()*b.num_graphs
        print(f"Epoch {epoch+1}: MAE={tot/len(ds):.4f}")
    print("✅ Multitask demo finished.")

if __name__ == "__main__":
    main()
