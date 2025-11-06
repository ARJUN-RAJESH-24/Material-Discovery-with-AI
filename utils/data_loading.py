from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from pymatgen.core import Structure


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def structure_from_cif_text(cif_text: str) -> Structure:
    return Structure.from_str(cif_text, fmt="cif")


def coerce_structure(row: pd.Series, structure_col: str = "cif") -> Optional[Structure]:
    value = row.get(structure_col)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, Structure):
        return value
    return structure_from_cif_text(str(value))


def ensure_structures(df: pd.DataFrame, structure_col: str = "cif", out_col: str = "structure") -> pd.DataFrame:
    if out_col in df.columns and df[out_col].map(lambda x: isinstance(x, Structure)).all():
        return df
    df = df.copy()
    df[out_col] = df.apply(lambda r: coerce_structure(r, structure_col=structure_col), axis=1)
    df = df[df[out_col].notna()].reset_index(drop=True)
    return df


def split_train_val_test(
    df: pd.DataFrame,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df)
    n_test = int(n * frac_test)
    n_val = int(n * frac_val)
    test_df = df.iloc[:n_test]
    val_df = df.iloc[n_test : n_test + n_val]
    train_df = df.iloc[n_test + n_val :]
    return train_df, val_df, test_df


