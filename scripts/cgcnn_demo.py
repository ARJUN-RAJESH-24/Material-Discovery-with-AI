import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------
# Load Pre-processed Dataset
# ------------------------------
try:
    data = pd.read_pickle("data/sample_magpie.pkl")
    print(f"Loaded {len(data)} samples with Magpie features")
except FileNotFoundError:
    print("Error: data/sample_magpie.pkl not found. Please run the prepare_dataset.py script first.")
    exit()

# ------------------------------
# Convert Structures -> Graphs
# ------------------------------
graph_list = []
print("Converting structures to graphs...")
for _, row in tqdm(data.iterrows(), total=len(data)):
    # Node features are the pre-computed Magpie features
    x = torch.tensor(row['node_feats'], dtype=torch.float)
    struct = row['structure']
    
    # Use Pymatgen's efficient neighbor finding
    cutoff = 8.0
    all_neighbors = struct.get_all_neighbors(r=cutoff)
    edge_src, edge_dst, edge_attr_list = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        for neighbor_site, dist, j, _ in neighbors:
            edge_src.append(i)
            edge_dst.append(j)
            edge_attr_list.append(dist)

    if not edge_src:
        continue

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    # Reshape edge_attr to be [num_edges, 1]
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).view(-1, 1)
    
    y = torch.tensor([row['e_form']], dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph_list.append(graph)

print(f"Converted {len(graph_list)} structures to graphs")

# ------------------------------
# Data Splitting
# ------------------------------
# Adjust split based on your sample size (200 in this case)
train_graphs = graph_list[:160]
val_graphs = graph_list[160:180]
test_graphs = graph_list[180:]

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=16)
test_loader = DataLoader(test_graphs, batch_size=16)

# ------------------------------
# Correct CGCNN Model Definition
# ------------------------------
class SimpleCGCNN(nn.Module):
    # THIS IS THE CRITICAL FIX: node_feat_dim=133 and edge_feat_dim=1 are specified
    def __init__(self, node_feat_dim=133, edge_feat_dim=1, hidden_dim=64):
        super().__init__()
        # The 'dim' argument for edge features is REQUIRED
        self.conv1 = CGConv(channels=node_feat_dim, dim=edge_feat_dim, aggr='add')
        self.conv2 = CGConv(channels=hidden_dim, dim=edge_feat_dim, aggr='add')
        self.lin1 = nn.Linear(hidden_dim, 32)
        self.lin2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.relu(self.lin1(x))
        out = self.lin2(x)
        return out

# ------------------------------
# Training Setup
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SimpleCGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ------------------------------
# Training Loop
# ------------------------------
print("Starting training...")
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------
# Evaluation
# ------------------------------
print("Evaluating model...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.extend(batch.y.cpu().numpy().flatten())
        y_pred.extend(out.cpu().numpy().flatten())

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# ------------------------------
# Plotting
# ------------------------------
plt.figure()
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
plt.xlabel("True Formation Energy (eV/atom)")
plt.ylabel("Predicted Formation Energy (eV/atom)")
plt.title("CGCNN with Magpie Features")
plt.legend()
plt.grid(True)
plt.savefig("results/prediction_plot_magpie.png")
print("Saved prediction plot to results/prediction_plot_magpie.png")
