#!/usr/bin/env python3
"""
Prepare dataset from existing pickles or (optionally) Materials Project.
- Validates columns
- (Optional) filters rows
- Writes a cleaned pickle to data/processed.pkl
"""
import argparse
import os
import pandas as pd

REQUIRED_COLS = ["structure", "formation_energy_per_atom", "band_gap"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/sample_mini.pkl", help="Input .pkl (DataFrame with 'structure' column)")
    ap.add_argument("--outfile", default="data/processed.pkl", help="Output cleaned .pkl")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with missing targets")
    args = ap.parse_args()

    assert os.path.exists(args.infile), f"Input not found: {args.infile}"
    df = pd.read_pickle(args.infile)

    # Basic schema check
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.dropna:
        df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

    # Ensure 'material_id' & 'pretty_formula' exist for bookkeeping
    if "material_id" not in df.columns:
        df["material_id"] = [f"mid_{i}" for i in range(len(df))]
    if "pretty_formula" not in df.columns and "formula" in df.columns:
        df["pretty_formula"] = df["formula"].astype(str)
    elif "pretty_formula" not in df.columns:
        df["pretty_formula"] = df["material_id"].astype(str)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df.to_pickle(args.outfile)
    print(f"✅ Saved cleaned dataset to {args.outfile} with {len(df)} rows.")

if __name__ == "__main__":
    main()
