"""
Minimal CGCNN-style message passing network.
- Node input: atomic number IDs (x as [N,1])
- Learn an embedding for elements, pass messages using edge distances
- Predict 1 or multi-task outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class EdgeMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )
    def forward(self, e):
        return self.net(e)

class MessageBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2*hidden + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, x, edge_index, edge_attr):
        # aggregate messages from neighbors j -> i
        row, col = edge_index  # i <- j
        m = torch.cat([x[row], x[col], edge_attr], dim=-1)
        m = self.msg(m)
        # sum aggregation
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m)
        # update
        x_new = self.gru(agg, x)
        return x_new

class CGCNN(nn.Module):
    def __init__(self, n_outputs=1, hidden=128, n_conv=3, elem_emb_dim=128, max_Z=100, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(max_Z+1, elem_emb_dim)
        self.proj = nn.Linear(elem_emb_dim, hidden)
        self.convs = nn.ModuleList([MessageBlock(hidden) for _ in range(n_conv)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_conv)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.squeeze(-1), data.edge_index, data.edge_attr, data.batch
        x = self.emb(x)                  # [N, emb]
        x = self.proj(x)                 # [N, hidden]
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, edge_index, edge_attr)  # residual
            x = norm(x)
        g = global_mean_pool(x, batch)   # [B, hidden]
        out = self.head(g)
        return out.squeeze(-1)
