#!/usr/bin/env python3
import pandas as pd, os, sys

path = sys.argv[1] if len(sys.argv)>1 else "data/processed.pkl"
assert os.path.exists(path), f"File not found: {path}"

df = pd.read_pickle(path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df[["structure","formation_energy_per_atom","band_gap"]].isna().sum())
assert "structure" in df.columns, "Missing 'structure'"
assert "formation_energy_per_atom" in df.columns, "Missing 'formation_energy_per_atom'"
print("✅ Data check passed.")
