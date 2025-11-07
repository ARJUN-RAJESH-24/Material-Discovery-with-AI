#!/usr/bin/env python3
"""
Convert crystal structures (pymatgen.Structure) into PyG graphs and cache to graphs/*.pt
Node features: atomic number (Z)
Edges: neighbor pairs within cutoff; edge_attr = [distance]
Targets: y = formation_energy_per_atom, band_gap (multi-task optional)
"""
import argparse, os, math
import torch
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm

def structure_to_graph(struct, cutoff=8.0):
    # Nodes: atomic numbers, positions in Cartesian Angstrom
    Z = torch.tensor([s.specie.Z for s in struct.sites], dtype=torch.long)
    pos = torch.tensor(struct.cart_coords, dtype=torch.float32)

    # Edges via neighbor search
    edge_i, edge_j, edge_attr = [], [], []
    for i, site in enumerate(struct.sites):
        neighs = struct.get_neighbors(site, r=cutoff)
        for nn in neighs:
            # neighbor.index can be None; map by position match if needed
            j = nn.index
            if j is None:
                # fallback: skip unknown indices (rare)
                continue
            dij = nn.nn_distance  # distance in Angstrom
            edge_i.append(i); edge_j.append(j)
            edge_attr.append([dij])

    if len(edge_i) == 0:
        # Single atom or pathological; create self loop
        edge_i, edge_j, edge_attr = [0], [0], [[0.0]]

    edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=Z.view(-1, 1),       # simple embedding index
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/processed.pkl")
    ap.add_argument("--outdir", default="graphs")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--targets", nargs="+", default=["formation_energy_per_atom"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_pickle(args.infile)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        struct = row["structure"]
        data = structure_to_graph(struct, cutoff=args.cutoff)

        # attach targets
        ys = []
        for t in args.targets:
            val = float(row[t]) if t in row and not pd.isna(row[t]) else math.nan
            ys.append(val)
        data.y = torch.tensor(ys, dtype=torch.float32)

        # bookkeeping
        data.material_id = row.get("material_id", f"mid_{idx}")
        data.formula = row.get("pretty_formula", str(data.material_id))
        outpath = os.path.join(args.outdir, f"{data.material_id}.pt")
        torch.save(data, outpath)

    print(f"✅ Saved {len(df)} graphs to {args.outdir}")

if __name__ == "__main__":
    main()
