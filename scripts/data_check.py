import pandas as pd

df = pd.read_pickle("data/sample_mini.pkl")

print("Rows:", len(df))
print("Columns:", df.columns)
print(df.isna().sum())

# Basic checks
assert "structure" in df.columns, "Missing 'structure' column!"
assert "formation_energy_per_atom" in df.columns, "Missing formation energy!"
assert len(df) > 0, "Dataset empty!"

print("✅ Data check passed!")
