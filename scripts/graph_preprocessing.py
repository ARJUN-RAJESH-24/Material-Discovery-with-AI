from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from rich import print
from torch_geometric.data import Data

from utils.data_loading import ensure_structures, read_table


def structure_to_graph(
    structure: Structure,
    radius: float = 8.0,
    max_neighbors: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use CrystalNN to infer neighbors; fallback to cutoff if needed
    try:
        cnn = CrystalNN(distance_cutoffs=None)
        neighbors = {}
        for i in range(len(structure)):
            neigh = cnn.get_nn_info(structure, i)
            # Sort by distance and clip
            neigh = sorted(neigh, key=lambda x: x["weight"], reverse=True)[:max_neighbors]
            neighbors[i] = [(n["site_index"], structure.get_distance(i, n["site_index"])) for n in neigh]
    except Exception:
        # Fallback simple radius neighbor search
        neighbors = {}
        for i in range(len(structure)):
            row = []
            for j in range(len(structure)):
                if i == j:
                    continue
                d = structure.get_distance(i, j)
                if d <= radius:
                    row.append((j, d))
            row = sorted(row, key=lambda x: x[1])[:max_neighbors]
            neighbors[i] = row

    # Node features: atomic numbers
    z = np.array([site.specie.Z for site in structure.sites], dtype=np.int64)

    # Edges and edge attributes: distance
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_attr: List[float] = []
    for i, neighs in neighbors.items():
        for j, dist in neighs:
            edge_src.append(i)
            edge_dst.append(j)
            edge_attr.append(dist)

    edge_index = np.vstack([edge_src, edge_dst]).astype(np.int64)
    edge_attr = np.array(edge_attr, dtype=np.float32)[:, None]  # (E, 1)
    return z, edge_index, edge_attr


def main():
    parser = argparse.ArgumentParser(description="Convert structures to PyG graphs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--structure-col", type=str, default="cif")
    parser.add_argument("--id-col", type=str, default="material_id")
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.dataset)
    df = ensure_structures(df, structure_col=args.structure_col, out_col="structure")

    graphs_written = 0
    for idx, row in df.iterrows():
        sid = row.get(args.id_col, f"row_{idx}")
        structure: Structure = row["structure"]
        z, edge_index, edge_attr = structure_to_graph(
            structure, radius=args.radius, max_neighbors=args.max_neighbors
        )

        x = torch.tensor(z[:, None], dtype=torch.long)  # (N, 1) atomic number
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
        if args.target and args.target in row and pd.notna(row[args.target]):
            y = torch.tensor([float(row[args.target])], dtype=torch.float32)
            data.y = y

        torch.save(data, out_dir / f"{sid}.pt")
        graphs_written += 1

    print(f"[green]Wrote {graphs_written} graphs to[/green] {str(out_dir)}")


if __name__ == "__main__":
    main()


