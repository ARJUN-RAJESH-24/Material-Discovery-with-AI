#!/usr/bin/env python3
"""
Baseline classical ML using matminer composition features + XGBoost.
Outputs metrics and a scatter plot to results/prediction_plot.png
"""
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from matminer.utils.conversions import str_to_composition
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/processed.pkl")
    ap.add_argument("--target", default="formation_energy_per_atom")
    ap.add_argument("--plot", default="results/prediction_plot.png")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    df = pd.read_pickle(args.infile).dropna(subset=[args.target]).copy()
    # ensure composition string
    if "pretty_formula" in df.columns:
        comp_series = df["pretty_formula"].astype(str)
    elif "formula" in df.columns:
        comp_series = df["formula"].astype(str)
    else:
        raise ValueError("Need pretty_formula or formula for composition features.")

    df["composition"] = comp_series.apply(str_to_composition)

    ep = ElementProperty.from_preset("magpie")
    st = Stoichiometry()
    X = ep.featurize_dataframe(df.copy(), "composition", pbar=False)
    X = st.featurize_dataframe(X, "composition", pbar=False)

    drop_cols = ["composition","structure","material_id","pretty_formula","formula"]
    X_feat = X.drop(columns=[c for c in drop_cols if c in X.columns])
    y = df[args.target].values

    Xtr, Xte, ytr, yte = train_test_split(X_feat, y, test_size=0.15, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-ish
        ("xgb", XGBRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42))
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    print(f"✅ Baseline {args.target} | MAE={mae:.4f} | R2={r2:.4f}")

    plt.figure()
    plt.scatter(yte, pred, s=8, alpha=0.7)
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(f"Baseline {args.target}")
    lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    plt.plot(lims, lims, "--")
    plt.savefig(args.plot, dpi=180, bbox_inches="tight")

    with open("results/baseline_metrics.json","w") as f:
        json.dump({"MAE": float(mae), "R2": float(r2), "target": args.target}, f, indent=2)

if __name__ == "__main__":
    main()
