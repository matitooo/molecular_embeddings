import argparse
import torch
import torch.nn.functional as F
import os
import yaml
import dgl
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.trimnet import TrimNet
from torch_geometric.data import Data,Batch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from types import SimpleNamespace

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
