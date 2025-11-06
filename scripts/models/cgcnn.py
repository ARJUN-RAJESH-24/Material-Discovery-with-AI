from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import CGConv, global_mean_pool


class CGCNN(nn.Module):
    def __init__(
        self,
        node_in_channels: int = 1,
        edge_in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        fc_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Node encoder: atomic number -> embedding
        self.node_embed = nn.Embedding(num_embeddings=120, embedding_dim=hidden_channels)

        self.convs = nn.ModuleList([
            CGConv(channels=hidden_channels, dim=edge_in_channels, normalize=True, batch_norm=True)
            for _ in range(num_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # x: (N, 1) atomic number (long)
        z = x.squeeze(-1).clamp(min=0, max=119)
        h = self.node_embed(z)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = torch.relu(h)
        g = global_mean_pool(h, batch)
        out = self.mlp(g)
        return out.view(-1)


