import torch
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles



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
