import torch
import torch.nn as nn
from torch_geometric.nn import GINConv,global_mean_pool

class MoleculeGraphEncoder(nn.Module):

    def __init__(
        self,
        node_dim=46,
        hidden_dim=128,
        embedding_dim=128,
        num_layers=4
    ):
        super().__init__()

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()

        for i in range(num_layers):

            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            self.convs.append(GINConv(mlp))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        graph_embedding = global_mean_pool(
            x,
            batch
        )

        graph_embedding = self.readout(
            graph_embedding
        )

        return graph_embedding