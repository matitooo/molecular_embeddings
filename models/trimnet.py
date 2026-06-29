
import argparse
import dgl
import math
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, Parameter
from torch.nn.functional import leaky_relu
from torch.nn.init import kaiming_uniform_, zeros_
from torch.utils.data import DataLoader
from torch_geometric.data import Data,Batch
from torch_geometric.nn import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from tqdm import tqdm
from types import SimpleNamespace
import yaml

def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"{x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]

def onehot_encoding_unk(x, allowable_set):
    """Maps unknowns to the last element ('other')."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for atom in mol.GetAtoms():
        results = (
            onehot_encoding_unk(atom.GetSymbol(),
                ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other'])
            + onehot_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10])
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                'other'])
            + [atom.GetIsAromatic()]
        )
        if not explicit_H:
            results += onehot_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4])
        if use_chirality:
            try:
                results += (onehot_encoding_unk(atom.GetProp('_CIPCode'), ['R','S'])
                            + [atom.HasProp('_ChiralityPossible')])
            except Exception:
                results += [0, 0] + [atom.HasProp('_ChiralityPossible')]
        feat.append(results)
    return np.array(feat)



def bond_attr(mol, use_chirality=True):
    feat, index = [], []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE,
                        bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE,
                        bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing(),
                    ]
                    if use_chirality:
                        bond_feats += onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ['STEREONONE','STEREOANY','STEREOZ','STEREOE'])
                    feat.append(bond_feats)
                    index.append([i, j])
    return np.array(index), np.array(feat)

def smiles_to_data(smiles: str) -> Data:
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    node_attr = atom_attr(mol)
    edge_index, edge_attr = bond_attr(mol)
    return Data(
        x=torch.FloatTensor(node_attr),
        edge_index=torch.LongTensor(edge_index).t().contiguous(),
        edge_attr=torch.FloatTensor(edge_attr),
    )

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class MultiHeadTripletAttention(MessagePassing):
    def __init__(
        self, node_channels, edge_channels, heads=3, negative_slope=0.2,
        **kwargs
    ):
        super(MultiHeadTripletAttention, self).__init__(
            aggr="add", node_dim=0, **kwargs
        )  # aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels,
                                                  heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels,
                                                  heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads,
                                                         3 * node_channels))
        self.weight_scale = Parameter(
            torch.Tensor(heads * node_channels, node_channels)
        )
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr  # noqa: E501
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        # return x_j * alpha
        # return self.prelu(alpha * e_ij * x_j)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return "{node_channels}, {node_channels},heads={heads}".format(**self.__dict__)  # noqa: E501


class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(
            dim, edge_dim, heads
        )  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            m = F.celu(self.conv.forward(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.ln(x.squeeze(0))
        return x


class TrimNet(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        edge_in_dim,
        hidden_dim=32,
        depth=3,
        heads=4,
        dropout=0.1,
        outdim=2,
    ):
        super(TrimNet, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.lin0 = Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [Block(hidden_dim, edge_in_dim, heads) for i in range(depth)]
        )
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        # self.lin1 = torch.nn.Linear(2 * hidden_dim, 2)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, outdim),
        )

    def forward(self, data):
        x = F.celu(self.lin0(data.x))
        for conv in self.convs:
            x = x + F.dropout(
                conv(x, data.edge_index, data.edge_attr),
                p=self.dropout,
                training=self.training,
            )
        x = self.set2set(x, data.batch)
        x = self.out(F.dropout(x, p=self.dropout, training=self.training))
        return x
    def embed(self, data):
        x = F.celu(self.lin0(data.x))
        for conv in self.convs:
            x = x + F.dropout(
                conv(x, data.edge_index, data.edge_attr),
                p=self.dropout,
                training=False,
            )
        return self.set2set(x, data.batch)  # (N, 2*hidden_dim)



class Inference_trimnet:

    def __init__(self):
        self.ckpt_path = 'checkpoints/trimnet_checkpoint.ckpt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ckpt = torch.load(self.ckpt_path, weights_only=False)
        self.opt  = self.ckpt['option']
    def load_model(self):
        self.model = TrimNet(
            in_dim      = self.opt['in_dim'],
            edge_in_dim = self.opt['edge_in_dim'],
            hidden_dim  = self.opt['hid'],
            depth       = self.opt['depth'],
            heads       = self.opt['heads'],
            dropout     = self.opt['dropout'],
            outdim      = self.opt['out_dim'],
        )
        self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.model.eval().to(self.device)
        return None
    
    def embed_smiles(self,smiles_list):
        all_embeddings = []
        self.model.eval() 
        for smi in tqdm(smiles_list):
            emb = self.predict(smi) 
            all_embeddings.append(emb)
        return all_embeddings


    def predict(self, smi):
        graph = smiles_to_data(smi)
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            embedding = self.model.embed(batch).squeeze()
        return embedding.detach().cpu()
    
    def pipe(self,smiles_list):
        self.load_model()
        embedded_smiles =self.embed_smiles(smiles_list)
        return embedded_smiles