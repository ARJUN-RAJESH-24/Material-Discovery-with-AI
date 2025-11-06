# scripts/prepare_dataset.py
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Structure, Composition  # FIX 1: Import Composition
from matminer.featurizers.composition import ElementProperty

# Load the dataset
print("Loading matbench_mp_e_form dataset...")
data = load_dataset("matbench_mp_e_form")
print(f"Dataset loaded with {len(data)} entries")

# Sample ~1000 entries for demo
data_sample = data.sample(n=200, random_state=42) # Reduced sample size for quicker demo

# Add Magpie features for each atom
ep_feat = ElementProperty.from_preset("magpie")

def get_node_features(structure: Structure):
    feats = []
    for site in structure.sites:
        # FIX 2: Create a Composition object from the element symbol string
        comp = Composition(site.specie.symbol)
        feats.append(ep_feat.featurize(comp)[0]) # Featurize and get the first (and only) row
    return feats

# Convert structure column to Magpie features (as a list of node features)
data_sample['node_feats'] = data_sample['structure'].apply(get_node_features)

# Save for quick loading in CGCNN
data_sample.to_pickle("data/sample_magpie.pkl")
print("Sample dataset with Magpie features saved as data/sample_magpie.pkl")
print(data_sample.head())
