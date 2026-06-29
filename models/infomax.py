from typing import Dict, List, Union, Callable
import dgl
import torch
import numpy as np
from functools import partial
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader


import argparse
import os
import yaml
import dgl 
import random
import numpy as np 
import pandas as pd
from tqdm import tqdm 
from rdkit import Chem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
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

class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_dim: int
            Input dimension of the layer (the torch.nn.Linear)
        out_dim: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        batch_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_dim}}`
            (Default value = None)
    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        batch_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_dim: int
            Input dimension of the linear layer
        out_dim: int
            Output dimension of the linear layer
    """

    def __init__(self, in_dim, out_dim, activation='relu', dropout=0., batch_norm=False, batch_norm_momentum=0.1,
                 bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()
        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias).to(device)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim, momentum=batch_norm_momentum).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_dim)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            if h.shape[1] != self.out_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        return h

class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_dim, out_dim, layers, hidden_size=None, mid_activation='relu', last_activation='none',
                 dropout=0., mid_batch_norm=False, last_batch_norm=False, batch_norm_momentum=0.1, device='cpu'):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_dim, out_dim, activation=last_activation, batch_norm=last_batch_norm,
                                                device=device, dropout=dropout,
                                                batch_norm_momentum=batch_norm_momentum))
        else:
            self.fully_connected.append(
                FCLayer(in_dim, hidden_size, activation=mid_activation, batch_norm=mid_batch_norm,
                        device=device, dropout=dropout, batch_norm_momentum=batch_norm_momentum))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    batch_norm=mid_batch_norm, device=device, dropout=dropout,
                                                    batch_norm_momentum=batch_norm_momentum))
            self.fully_connected.append(
                FCLayer(hidden_size, out_dim, activation=last_activation, batch_norm=last_batch_norm,
                        device=device, dropout=dropout, batch_norm_momentum=batch_norm_momentum))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding
        self.full_atom_feature_dims = get_atom_feature_dims()
        

        for i, dim in enumerate(self.full_atom_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for i, embedder in enumerate(self.atom_embedding_list):
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i] + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding
        self.full_bond_feature_dims = get_bond_feature_dims()

        for i, dim in enumerate(self.full_bond_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i] + 1)
            else:
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding




PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNA(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """

    def __init__(self,
                 hidden_dim,
                 target_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 readout_aggregators: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 readout_layers: int = 2,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(PNA, self).__init__()
        self.node_gnn = PNAGNN(hidden_dim=hidden_dim, aggregators=aggregators,
                               scalers=scalers, residual=residual, pairwise_distances=pairwise_distances,
                               activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, propagation_depth=propagation_depth, dropout=dropout,
                               posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum
                               )
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph):
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class PNAGNN(nn.Module):
    def __init__(self, hidden_dim, aggregators: List[str], scalers: List[str],
                 residual: bool = True, pairwise_distances: bool = False, activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none", mid_batch_norm: bool = False,
                 last_batch_norm: bool = False, batch_norm_momentum=0.1, propagation_depth: int = 5,
                 dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, **kwargs):
        super(PNAGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(in_dim=hidden_dim, out_dim=int(hidden_dim), in_dim_edges=hidden_dim, aggregators=aggregators,
                         scalers=scalers, pairwise_distances=pairwise_distances, residual=residual, dropout=dropout,
                         activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                         last_batch_norm=last_batch_norm, avg_d={"log": 1.0}, posttrans_layers=posttrans_layers,
                         pretrans_layers=pretrans_layers, batch_norm_momentum=batch_norm_momentum
                         ),

            )
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, graph: dgl.DGLGraph):
        graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        graph.edata['feat'] = self.bond_encoder(graph.edata['feat'])

        for mp_layer in self.mp_layers:
            mp_layer(graph)


class PNALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_dim_edges: int, aggregators: List[str], scalers: List[str],
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0, residual: bool = True, pairwise_distances: bool = False,
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, batch_norm_momentum=0.1,
                 avg_d: Dict[str, float] = {"log": 1.0}, posttrans_layers: int = 2, pretrans_layers: int = 1, ):
        super(PNALayer, self).__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges + 1) if self.pairwise_distances else (2 * in_dim + in_dim_edges),
            hidden_size=in_dim, out_dim=in_dim, mid_batch_norm=mid_batch_norm, last_batch_norm=last_batch_norm,
            layers=pretrans_layers, mid_activation=activation, dropout=dropout, last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum

        )
        self.posttrans = MLP(in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_dim=out_dim, layers=posttrans_layers, mid_activation=activation,
                             last_activation=last_activation, dropout=dropout, mid_batch_norm=mid_batch_norm,
                             last_batch_norm=last_batch_norm, batch_norm_momentum=batch_norm_momentum
                             )

    def forward(self, g):
        h = g.ndata['feat']
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['feat']], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        g.ndata['feat'] = h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data['feat']
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=-1)

        return {'feat': h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        if self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], squared_distance], dim=-1)
        elif not self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x']) ** 2, dim=-1)[:, None]
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], squared_distance], dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=-1)
        else:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1)
        return {"e": self.pretrans(z2)}





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