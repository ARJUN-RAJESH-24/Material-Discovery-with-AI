# Data Format Documentation

## 1. Dataset Files (.pkl)

Each dataset pickle contains a pandas DataFrame with:
- structure (pymatgen Structure object)
- formation_energy_per_atom
- band_gap
- material_id (if available)
- formula / pretty_formula

## 2. Graph Files (PyTorch Geometric)

Each graph saved in /graphs is a torch_geometric.data.Data object:

- data.x → node features (atomic numbers or embeddings)
- data.edge_index → graph connectivity
- data.edge_attr → edge distances or features
- data.y → target property tensor
- data.pos → atomic coordinates (optional)

## 3. Expected Columns for Dataset Preprocessing

- structure  
- composition / formula  
- formation_energy_per_atom  
- band_gap  
- stability (optional)  
