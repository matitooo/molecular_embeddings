from rdkit import Chem 
from rdkit.Chem import MolFromSmiles
import torch
from torch_geometric.data import Data
import numpy as np

def return_dicts():
    return {
        'symbol_dict': {
            'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
            'Si': 5, 'P': 6, 'S': 7, 'Cl': 8, 'As': 9,
            'Se': 10, 'Br': 11, 'Te': 12, 'I': 13, 'At': 14
        },
        'hybridization_dict': {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4
        },
        'chirality_dict': {
            'R': 0,
            'S': 1
        },
        'stereo_dict' :  {
        'STEREONONE': 0,
        'STEREOANY': 1,
        'STEREOZ': 2,
        'STEREOE': 3
}
    }

def read_data():
  with open ('data/smiles_full.txt','r') as f:
            smiles_list = [s for s in f.read().splitlines() if s.strip()]
  return smiles_list

def smiles_to_data(smiles: str,dicts,config):
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    node_attr = mole_encoding(mol,dicts,config)
    edge_index, edge_attr = bond_encoding(mol,dicts)
    return Data(
        x=torch.FloatTensor(node_attr),
        edge_index=torch.LongTensor(edge_index).t().contiguous(),
        edge_attr=torch.FloatTensor(edge_attr),
    )


def encode(value,dict,other = True):
     if other:
      out = torch.zeros(size= (len(dict)+1,))
      out[dict[value] if value in dict.keys() else -1] = 1
      return out
     else:
      out = torch.zeros(size= (len(dict),))
      out[dict[value]] = 1
      return out

def encode_num(value,list):
     out = torch.zeros(size=(len(list)+1,))
     out[value if value in list else -1] = 1
     return out

def mole_encoding(mol,dicts,config):
    encoded_atoms = []
    for atom in mol.GetAtoms():
        encoded_atoms.append(atom_encoding(atom,dicts,config))
    return np.array(encoded_atoms)

def atom_encoding(atom, dicts,config):
    vector_collection = {}
    vector_collection['atom_symbol'] = encode(atom.GetSymbol(), dicts['symbol_dict'])
    vector_collection['degree'] = encode_num(atom.GetDegree(), list(range(10)))
    vector_collection['charge'] = torch.tensor([atom.GetFormalCharge()])

    vector_collection['radical_electrons'] = torch.tensor([atom.GetNumRadicalElectrons()])
    vector_collection['hybridization'] = encode(
        atom.GetHybridization(),
        dicts['hybridization_dict']
    )
    vector_collection['aromatic'] = torch.tensor([int(atom.GetIsAromatic())])
    vector_collection['total_h'] = encode_num(atom.GetTotalNumHs(), list(range(5)))
    chirality = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else 'other'
    vector_collection['chirality'] = encode(chirality, dicts['chirality_dict'])
    vector_collection['chirality_center'] = torch.tensor([int(atom.HasProp('_ChiralityPossible'))])
    tensors = [vector_collection[key] for key in config if key in vector_collection.keys()]
    return tensors

def bond_encoding(mol, dicts):
    edge_attr = []
    edge_index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                continue
            bt = bond.GetBondType()
            basic_features = torch.tensor([
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing(),
            ], dtype=torch.float)
            stereo_features = encode(
                bond.GetStereo(),
                dicts['stereo_dict']
            ).float()
            bond_features = torch.cat([
                basic_features,
                stereo_features
            ])
            edge_attr.append(bond_features)
            edge_index.append([i, j])

    if len(edge_attr) == 0:
        edge_attr = torch.empty(
            (0, 6 + len(dicts['stereo_dict']) + 1),
            dtype=torch.float
        )
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )
    else:
        edge_attr = torch.stack(edge_attr)
        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

    return edge_index, edge_attr
    




