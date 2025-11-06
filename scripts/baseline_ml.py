from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core.composition import Composition
from rich import print
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.data_loading import read_table


def build_pipeline(model_name: str):
    # Composition features via Magpie set
    featurizer = ElementProperty.from_preset("magpie")

    if model_name.lower() == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=0,
            tree_method="hist",
            objective="reg:squarederror",
        )
    elif model_name.lower() == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=2000,
            num_leaves=63,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
        )

    pipe = Pipeline(
        steps=[
            ("to_comp", StrToComposition(target_col_id="composition", overwrite_data=True)),
            ("magpie", featurizer),
            ("model", model),
        ]
    )
    return pipe


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(5, 5), dpi=140)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline ML for materials property prediction")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm", "rf"])
    parser.add_argument("--output", type=str, required=True, help="Output prefix (no extension)")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(args.dataset)

    # Expect a column named 'composition' or 'formula'. If 'formula' is present, copy to 'composition'.
    if "composition" not in df.columns:
        if "formula" in df.columns:
            df = df.rename(columns={"formula": "composition"})
        else:
            raise ValueError("Dataset must include a 'composition' or 'formula' column.")

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available: {list(df.columns)}")

    df = df[["composition", args.target]].dropna().reset_index(drop=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df[["composition"]], df[args.target], test_size=args.test_size + args.val_size, random_state=args.random_state
    )
    rel_val = args.val_size / (args.test_size + args.val_size) if (args.test_size + args.val_size) > 0 else 0.0
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - rel_val, random_state=args.random_state)

    pipe = build_pipeline(args.model)
    pipe.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    y_pred = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"mae": mae, "r2": r2, "n_test": int(len(y_test)), "model": args.model, "target": args.target}
    print("[bold green]Baseline Results[/bold green]", metrics)

    with open(out_prefix.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_parity(
        y_true=np.asarray(y_test, dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
        out_path=out_prefix.with_suffix(".parity.png"),
        title=f"Baseline {args.model.upper()} - {args.target}",
    )

    joblib.dump(pipe, out_prefix.with_suffix(".joblib"))


if __name__ == "__main__":
    main()


