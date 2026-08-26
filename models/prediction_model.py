import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

N_CELL_LINES = 7


class ResNet(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1, n_layers=6, layernorm=True):
        super().__init__()
        self.mlps = nn.ModuleList()

        for _ in range(n_layers):
            norm = (
                nn.BatchNorm1d(hidden_dim) if layernorm == "batchnorm"
                else nn.LayerNorm(hidden_dim) if layernorm
                else nn.Identity()
            )

            self.mlps.append(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    norm,
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim),
                )
            )

    def forward(self, x):
        for l in self.mlps:
            x = x + l(x)
        return x


class MoleculeGraphEncoder(nn.Module):
    def __init__(self, node_dim=47, hidden_dim=128, embedding_dim=128, num_layers=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, data):
        x = self.node_proj(data.x)

        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index))

        x = global_mean_pool(x, data.batch)
        return self.readout(x)


class DrugCombinationModel(nn.Module):
    def __init__(self, mol_encoder, embedding_dim=128, hidden_dim=256,
                 n_cell_lines=7, dropout=0.1, n_layers=6, layernorm=True):

        super().__init__()
        self.N_CELL_LINES = n_cell_lines
        self.mol_encoder = mol_encoder

        self.rnet = ResNet(embedding_dim + n_cell_lines, hidden_dim, dropout, n_layers, layernorm)

        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim + n_cell_lines, 1),
        )

    def forward(self, batch):

        device = batch.z.device

        B, D, T = batch.z.shape

        # Encode all unique molecular graphs exactly once.
        mol_batch = batch.mol_batch.to(device)
        mol_embeddings = self.mol_encoder(mol_batch)

        # Map each drug position to its molecular embedding.
        #
        # batch.drug_index: (B, D)
        # mol_embeddings:  (N_unique_drugs, E)
        # drug_stack:      (B, D, E)
        drug_stack = mol_embeddings[
            batch.drug_index.to(device)
        ]

        # Weighted combination of drug embeddings.
        #
        # drug_stack: (B, D, E)
        # z:          (B, D, T)
        # result:     (B, T, E)
        aggregated = torch.einsum(
            "bde,bdt->bte",
            drug_stack,
            batch.z,
        )

        # Cell-line encoding.
        cell_oh = torch.zeros(
            B,
            self.N_CELL_LINES,
            device=device,
            dtype=aggregated.dtype,
        )

        cell_oh.scatter_(
            1,
            batch.cell_line.long().view(B, 1),
            1.0,
        )

        cell_oh = cell_oh.unsqueeze(1).expand(
            B,
            T,
            self.N_CELL_LINES,
        )

        # Prediction network.
        x = torch.cat(
            [
                aggregated,
                cell_oh,
            ],
            dim=-1,
        )

        x = x.reshape(B * T, -1)

        x = self.rnet(x)

        pred = self.predictor(x).squeeze(-1)

        pred = pred.reshape(B, T)

        return pred, batch.mask.squeeze(1)


class DrugCombinationModelWithPrecomputedEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256,
                 n_cell_lines=7, dropout=0.1, n_layers=6, layernorm=True):

        super().__init__()
        self.N_CELL_LINES = n_cell_lines

        self.rnet = ResNet(embedding_dim + n_cell_lines, hidden_dim, dropout, n_layers, layernorm)

        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim + n_cell_lines, 1),
        )

    def forward(self, batch):
        device = batch.z.device
        B, D, T = batch.z.shape

        drug_embeddings = []

        for slot in batch.mol_batches:
            emb = slot["emb"].to(device)   # (B, E)
            mask = slot["mask"].to(device)  # (B,)

            full = torch.zeros(B, emb.size(-1), device=device)
            full[mask] = emb
            drug_embeddings.append(full)

        drug_stack = torch.stack(drug_embeddings, dim=1)  # (B, D, E)

        z = batch.z  # (B, D, T)

        weighted = drug_stack.unsqueeze(2) * z.unsqueeze(-1)  # (B, D, T, E)
        aggregated = weighted.sum(dim=1)  # (B, T, E)

        cell_oh = torch.zeros(B, self.N_CELL_LINES, device=device)
        cell_oh.scatter_(1, batch.cell_line.long().view(B, 1), 1.0)

        cell_oh = cell_oh.unsqueeze(1).expand(B, T, self.N_CELL_LINES)

        x = torch.cat([aggregated, cell_oh], dim=-1)  # (B, T, E+C)

        x = x.view(B * T, -1)

        out = self.rnet(x)
        pred = self.predictor(out).squeeze(-1)

        pred = pred.view(B, T)

        return pred, batch.mask.squeeze(1)