import argparse
import torch 
import torch.nn.functional as F
import os
import yaml
import dgl 
import random
import numpy as np 
import os
import pandas as pd
from tqdm import tqdm 
from rdkit import Chem
from models.infomax import PNA
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch.utils.data import Dataset,DataLoader
from torch_geometric.data import Batch
from types import SimpleNamespace
from math import sqrt

EPS = 1e-5
SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'SiLU', 'None'}


def seed_all(seed):
    if not seed:
        seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    random.seed(seed)



def graph_only_collate(batch):
        return dgl.batch(batch)


def aggregate_mean(h, **kwargs):
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))

def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()

class InferenceDataset(Dataset):

    def __init__(self, smiles_list, device=torch.device('cuda:0'), transform=None, **kwargs):
        atom_slices = [0]
        edge_slices = [0]
        all_atom_features = []
        all_edge_features = []
        edge_indices = []  # edges of each molecule in coo format
        total_atoms = 0
        total_edges = 0
        n_atoms_list = []
        for mol_idx, smiles in tqdm(enumerate(smiles_list)):
            # get the molecule using the smiles representation from the csv file
            mol = Chem.MolFromSmiles(smiles)
            # add hydrogen bonds to molecule because they are not in the smiles representation
            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            # Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(edges_list, dtype=torch.long).T
            edge_features = torch.tensor(edge_features_list, dtype=torch.long)

            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)
            total_edges += len(edges_list)
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)
            n_atoms_list.append(n_atoms)

        self.n_atoms = torch.tensor(n_atoms_list)
        self.atom_slices = torch.tensor(atom_slices, dtype=torch.long)
        self.edge_slices = torch.tensor(edge_slices, dtype=torch.long)
        self.edge_indices = torch.cat(edge_indices, dim=1)
        self.all_atom_features = torch.cat(all_atom_features, dim=0)
        self.all_edge_features = torch.cat(all_edge_features, dim=0)

    def __len__(self):
        return len(self.atom_slices) - 1

    def __getitem__(self, idx):

        e_start = self.edge_slices[idx]
        e_end = self.edge_slices[idx + 1]
        start = self.atom_slices[idx]
        n_atoms = self.n_atoms[idx]
        edge_indices = self.edge_indices[:, e_start: e_end]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms)
        g.ndata['feat'] = self.all_atom_features[start: start + n_atoms]
        g.edata['feat'] = self.all_edge_features[e_start: e_end]
        return g

class Inference_3d_infomax:
    def __init__(self):
        with open("config/3d_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.args = args = SimpleNamespace(**config)

    def pipe(self,smiles):
        seed_all(self.args.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.device == 'cuda' else "cpu")
        self.test_data = InferenceDataset(device=self.device, smiles_list=smiles)
        print('num_smiles: ', len(self.test_data))
        model, _,_ = self.load_model()
        print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad), '\n')
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device,weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test_loader = DataLoader(self.test_data, batch_size=256, collate_fn=graph_only_collate)
        embed_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                out = model(batch).squeeze()
                embed_list += out.unbind(0)
        return embed_list

    def load_model(self):
        model = PNA(avg_d=self.test_data.avg_degree if hasattr(self.test_data, 'avg_degree') else 1, device=self.device,
                                        **self.args.model_parameters)
        pretrained_gnn_dict = {}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict) 
        model.load_state_dict(model_state_dict)
        return model, None, False