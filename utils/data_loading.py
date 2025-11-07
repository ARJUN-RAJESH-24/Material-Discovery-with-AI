import os, glob, torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

class GraphFolderDataset(Dataset):
    """
    Loads *.pt files from a folder. Each file must be a torch_geometric.data.Data
    with fields: x, edge_index, edge_attr, y, and optional material_id/formula.
    """
    def __init__(self, root, targets=None, max_items=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        if max_items is not None:
            self.files = self.files[:max_items]
        self.targets = targets or ["formation_energy_per_atom"]  # for metadata

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(self.files[idx])
        # Ensure batch attributes are lists for screening convenience
        if not hasattr(data, "material_id"):
            data.material_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
        if not hasattr(data, "formula"):
            data.formula = data.material_id
        return data
