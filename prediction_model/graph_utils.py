from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import torch
from torch_geometric.data import Data
import yaml


with open('config/graph_config.yaml','r') as f:
        graph_config = yaml.safe_load(f)


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
        'stereo_dict': {
            Chem.rdchem.BondStereo.STEREONONE: 0,
            Chem.rdchem.BondStereo.STEREOANY: 1,
            Chem.rdchem.BondStereo.STEREOZ: 2,
            Chem.rdchem.BondStereo.STEREOE: 3
        },
        'size_dict': {
  'atom_symbol':16,
  'degree':11,
  'charge':1,
  'radical_electrons':1,
  'hybridization':6,
  'aromatic':1,
  'total_h':6,
  'chirality':3,
  'chirality_center':1
}
    }


def read_data(data_path):
    with open(data_path, 'r') as f:
        return [s for s in f.read().splitlines() if s.strip()]


def encode(value, mapping):
    out = torch.zeros(len(mapping) + 1)
    out[mapping.get(value, len(mapping))] = 1
    return out


def encode_num(value, values):
    out = torch.zeros(len(values) + 1)
    out[value if value in values else len(values)] = 1
    return out


def atom_encoding(atom, dicts):
    
    vc = {}

    vc['atom_symbol'] = encode(atom.GetSymbol(), dicts['symbol_dict'])
    vc['degree'] = encode_num(atom.GetDegree(), list(range(10)))
    vc['charge'] = torch.tensor([atom.GetFormalCharge()], dtype=torch.float)
    vc['radical_electrons'] = torch.tensor([atom.GetNumRadicalElectrons()], dtype=torch.float)

    vc['hybridization'] = encode(atom.GetHybridization(), dicts['hybridization_dict'])
    vc['aromatic'] = torch.tensor([int(atom.GetIsAromatic())], dtype=torch.float)
    vc['total_h'] = encode_num(atom.GetTotalNumHs(), list(range(5)))

    chirality = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else 'OTHER'
    if chirality not in dicts['chirality_dict']:
        chirality = 'OTHER'

    vc['chirality'] = encode(chirality, dicts['chirality_dict'])
    vc['chirality_center'] = torch.tensor([int(atom.HasProp('_ChiralityPossible'))], dtype=torch.float)

    order = [
        'atom_symbol',
        'degree',
        'charge',
        'radical_electrons',
        'hybridization',
        'aromatic',
        'total_h',
        'chirality',
        'chirality_center'
    ]

    return torch.cat([vc[k] for k in order if graph_config[k] ], dim=0)


def mole_encoding(mol, dicts):
    return torch.stack([atom_encoding(a, dicts) for a in mol.GetAtoms()])


def bond_encoding(mol, dicts):
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bt = bond.GetBondType()

        basic = torch.tensor([
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ], dtype=torch.float)

        stereo = encode(bond.GetStereo(), dicts['stereo_dict']).float()

        feat = torch.cat([basic, stereo])

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(feat)
        edge_attr.append(feat)

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 11), dtype=torch.float)
        return edge_index, edge_attr

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return edge_index, edge_attr


def smiles_to_data(smiles, dicts):
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)

    x = mole_encoding(mol, dicts)
    edge_index, edge_attr = bond_encoding(mol, dicts)

    return Data(
        x=x.float(),
        edge_index=edge_index,
        edge_attr=edge_attr.float()
    )