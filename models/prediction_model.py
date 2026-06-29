import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

N_CELL_LINES = 7

class ResNet(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1, n_layers = 6, layernorm = True):
        super().__init__()
        self.mlps = nn.ModuleList()
        for l in range(n_layers):
            norm = (
                nn.BatchNorm1d(hidden_dim) if layernorm == "batchnorm"
                else nn.LayerNorm(hidden_dim) if layernorm
                else nn.Identity()
            )
            self.mlps.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                           norm,
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, embed_dim)))
    def forward(self, x):
        for l in self.mlps:
            x = torch.add(l(x), x)
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
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, data):
        x = self.node_proj(data.x)
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index))
        return self.readout(global_mean_pool(x, data.batch))


class DrugCombinationModel(nn.Module):
    def __init__(self,mol_encoder, embedding_dim=128, hidden_dim=256, n_cell_lines=7,dropout=0.1,n_layers=6,layernorm=True):
        super().__init__()
        N_CELL_LINES = n_cell_lines 
        self.mol_encoder = mol_encoder
        self.rnet = ResNet(embedding_dim+N_CELL_LINES,hidden_dim,dropout,n_layers,layernorm)
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim+N_CELL_LINES, 1)
        )

    def forward(self, batch):
        device = batch.z.device
        B = batch.mask.shape[0]
        max_exp = batch.z.shape[1]

        drug_embeddings = []
        for slot in batch.mol_batches:
            mask = slot['mask'].to(device)
            emb = self.mol_encoder(slot['batch'].to(device)) 
            full = torch.zeros(B, emb.shape[-1], device=device)
            full[mask] = emb
            drug_embeddings.append(full)

        n_drugs = len(drug_embeddings)

        drug_stack = torch.stack(drug_embeddings, dim=1)             
        z = batch.z.view(B, n_drugs, max_exp)                         
        weighted = drug_stack.unsqueeze(2) * z.unsqueeze(-1)          
        aggregated = weighted.sum(dim=1)                             

        cell_oh = torch.zeros(B, N_CELL_LINES, device=device)
        cell_oh.scatter_(1, batch.cell_line.long().view(B, 1), 1.0)
        cell_oh = cell_oh.unsqueeze(1).expand(-1, max_exp, -1)

        pred = self.predictor(
            torch.cat([aggregated, cell_oh], dim=-1)
        ).squeeze(-1)                                                  

        return pred, batch.mask.squeeze(1)                            
    

class DrugCombinationModelWithPrecomputedEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, n_cell_lines=7,dropout=0.1,n_layers=6,layernorm=True):
        super().__init__()
        N_CELL_LINES = n_cell_lines 
        
        self.N_CELL_LINES = N_CELL_LINES
        self.rnet = ResNet(embedding_dim+N_CELL_LINES,hidden_dim,dropout,n_layers,layernorm)
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim+N_CELL_LINES, 1)
        )

    def forward(self, batch):
        device = batch.z.device
        B = batch.mask.shape[0]
        max_exp = batch.z.shape[1]

        drug_embeddings = []
        for slot in batch.mol_batches:
            mask = slot['mask'].to(device)
            emb = slot['emb'].to(device) 
            full = torch.zeros(B, emb.shape[-1], device=device)
            full[mask] = emb
            drug_embeddings.append(full)

        n_drugs = len(drug_embeddings)
        drug_stack = torch.stack(drug_embeddings, dim=1)             
        z = batch.z.view(B, n_drugs, max_exp)                        
        weighted = drug_stack.unsqueeze(2) * z.unsqueeze(-1)         
        aggregated = weighted.sum(dim=1)                             

        cell_oh = torch.zeros(B, self.N_CELL_LINES, device=device)
        cell_oh.scatter_(1, batch.cell_line.long().view(B, 1), 1.0)
        cell_oh = cell_oh.unsqueeze(1).expand(-1, max_exp, -1)

        out = self.rnet(
            torch.cat([aggregated, cell_oh], dim=-1)
        ).squeeze(-1)
        print(out.shape)
        pred = self.predictor(out).squeeze(-1)
        return pred, batch.mask.squeeze(1) 