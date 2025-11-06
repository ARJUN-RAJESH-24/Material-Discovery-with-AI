import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# ------------------------------
# Step 1 – Load Sample Dataset
# ------------------------------
data = pd.read_pickle("data/sample_mini.pkl")
print("Dataset loaded, number of entries:", len(data))
print("Available columns:", data.columns.tolist())

# Map e_form to formation_energy_per_atom if needed
if 'e_form' in data.columns:
    data = data.rename(columns={'e_form': 'formation_energy_per_atom'})

# Generate synthetic band gap data since matbench_mp_e_form doesn't include it
if 'band_gap' not in data.columns:
    print("Generating synthetic band gap data for demo...")
    data['band_gap'] = np.random.uniform(0.5, 5.0, size=len(data))

data = data[['structure', 'formation_energy_per_atom', 'band_gap']]

# ------------------------------
# Step 2 – Convert Structures → Graphs
# ------------------------------
graph_list = []
print("Converting structures to graphs...")
for idx, row in tqdm(data.iterrows(), total=len(data)):
    struct = row['structure']  # Pymatgen Structure object
    
    atomic_numbers = torch.tensor([site.specie.number for site in struct.sites], dtype=torch.long)
    
    # Use efficient neighbor finding
    cutoff = 8.0
    all_neighbors = struct.get_all_neighbors(r=cutoff)
    
    edge_src = []
    edge_dst = []
    edge_attr_list = []
    for i, neighbors in enumerate(all_neighbors):
        for neighbor_site, dist, j, _ in neighbors:
            edge_src.append(i)
            edge_dst.append(j)
            edge_attr_list.append(dist)
    
    if not edge_src:
        continue
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).view(-1, 1)
    
    # Create target with shape (1, 2) for proper batching
    y = torch.tensor([[row['formation_energy_per_atom'], row['band_gap']]], dtype=torch.float)
    
    graph = Data(x=atomic_numbers.view(-1, 1).float(), edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph_list.append(graph)

print(f"Converted {len(graph_list)} structures to graphs.")

# ------------------------------
# Step 3 – Define Multi-Task CGCNN
# ------------------------------
class MultiTaskCGCNN(nn.Module):
    def __init__(self, node_feat_dim=1, edge_feat_dim=1, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Embedding layer to project input features to hidden dimension
        self.embedding = nn.Linear(node_feat_dim, hidden_dim)
        # Graph convolution layers
        self.conv1 = CGConv(channels=hidden_dim, dim=edge_feat_dim, aggr='add')
        self.conv2 = CGConv(channels=hidden_dim, dim=edge_feat_dim, aggr='add')
        # Prediction layers
        self.lin1 = nn.Linear(hidden_dim, 32)
        self.lin2 = nn.Linear(32, 2)  # two outputs: formation energy + band gap
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Embed atomic numbers to hidden dimension
        x = self.embedding(x)
        x = self.relu(x)
        # Apply graph convolutions
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        # Pool and predict
        x = global_mean_pool(x, batch)
        x = self.relu(self.lin1(x))
        out = self.lin2(x)
        return out

# ------------------------------
# Step 4 – Training Setup
# ------------------------------
train_graphs = graph_list[:160]
val_graphs = graph_list[160:180]
test_graphs = graph_list[180:]

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=16)
test_loader = DataLoader(test_graphs, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = MultiTaskCGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ------------------------------
# Step 5 – Training Loop
# ------------------------------
print("Starting training...")
for epoch in range(15):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # batch.y has shape (batch_size, 2), out has shape (batch_size, 2)
        loss = loss_fn(out, batch.y.squeeze(1))  # Remove extra dimension if present
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------
# Step 6 – Quick Evaluation
# ------------------------------
print("Evaluating model...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.extend(batch.y.squeeze(1).cpu().numpy())
        y_pred.extend(out.cpu().numpy())

y_true = pd.DataFrame(y_true, columns=['formation_energy', 'band_gap'])
y_pred = pd.DataFrame(y_pred, columns=['formation_energy', 'band_gap'])

print("\nFormation Energy Metrics:")
print(f"MAE: {mean_absolute_error(y_true['formation_energy'], y_pred['formation_energy']):.4f}")
print(f"R²: {r2_score(y_true['formation_energy'], y_pred['formation_energy']):.4f}")

print("\nBand Gap Metrics:")
print(f"MAE: {mean_absolute_error(y_true['band_gap'], y_pred['band_gap']):.4f}")
print(f"R²: {r2_score(y_true['band_gap'], y_pred['band_gap']):.4f}")

# ------------------------------
# Step 7 – Scatter Plots
# ------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_true['formation_energy'], y_pred['formation_energy'], alpha=0.7)
plt.plot([y_true['formation_energy'].min(), y_true['formation_energy'].max()],
         [y_true['formation_energy'].min(), y_true['formation_energy'].max()], 'r--', label='Ideal')
plt.xlabel("True Formation Energy")
plt.ylabel("Predicted Formation Energy")
plt.title("Formation Energy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(y_true['band_gap'], y_pred['band_gap'], alpha=0.7)
min_gap = y_true['band_gap'].min()
max_gap = y_true['band_gap'].max()
plt.plot([min_gap, max_gap], [min_gap, max_gap], 'r--', label='Ideal')
plt.xlabel("True Band Gap")
plt.ylabel("Predicted Band Gap")
plt.title("Band Gap")
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/multitask_prediction_plot.png")
print("Saved multitask prediction plot to results/multitask_prediction_plot.png")
