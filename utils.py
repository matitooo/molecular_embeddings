import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from graph_utils import *

def collate_fn(data_list):
    B = len(data_list)
    n_drugs_list = [d.n_drugs for d in data_list]
    max_drugs    = max(n_drugs_list)
    max_exp      = max(d.z.shape[1] for d in data_list)
    z_pad  = torch.zeros(B, max_drugs, max_exp)
    z_mask = torch.zeros(B, max_exp, dtype=torch.bool)
    y_pad  = torch.zeros(B, max_exp)

    for i, d in enumerate(data_list):
        nd = d.n_drugs
        ne = d.z.shape[1]
        z_pad[i, :nd, :ne]  = d.z
        z_mask[i, :ne]      = True
        y_pad[i, :ne]       = d.y.squeeze(-1)

    mol_batches = []
    for pos in range(max_drugs):
        graphs = [d.mol_graphs[pos] for d in data_list if d.n_drugs > pos]
        mask   = torch.tensor([d.n_drugs > pos for d in data_list])
        mol_batches.append({'batch': Batch.from_data_list(graphs), 'mask': mask})

    return {
        'y':           y_pad,            
        'z':           z_pad,            
        'z_mask':      z_mask,           
        'cell_line':   torch.stack([d.cell_line for d in data_list]),
        'n_drugs':     torch.tensor(n_drugs_list),
        'mol_batches': mol_batches
    }


def masked_mse(pred, target, mask):
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum()


def generate_dataloader():
    dicts = return_dicts()
    train_path = 'data/train.pt'
    test_path = 'data/test.pt'
    dataset_path = 'data/droparray_small.pt'

    print('Loading dataset')
    dataset = torch.load(dataset_path,weights_only=False)
    print('Loading train data')
    train = torch.load(train_path,weights_only=False)
    print('Loading test data')      
    test = torch.load(test_path,weights_only=False)

    smiles_dict = dict(zip(dataset['smiles']['drug_id'],dataset['smiles']['SMILES']))

    for instance in tqdm(train,desc='Vectorizing train molecules'):
      embed = []
      for drug in instance.x:
        embed.append(smiles_to_data(smiles_dict[drug.item()],dicts))
      instance.vectors = embed

    train_loader = torch.utils.data.DataLoader(train, batch_size=128, num_workers=16, collate_fn=batch_instances_graph, shuffle=True, drop_last=False)

    for instance in tqdm(test,desc='Vectorizing test molecules'):
        embed = []
        for drug in instance.x:
          embed.append(smiles_to_data(smiles_dict[drug.item()],dicts))
        instance.vectors = embed

    test_loader = torch.utils.data.DataLoader(test, batch_size=128, num_workers=16, collate_fn=batch_instances_graph, shuffle=True, drop_last=False)

    return train_loader,test_loader

def batch_instances_graph(instances, drug_graph_dict):
    max_l = max(inst.z.shape[1] for inst in instances)
    
    instances_ = []
    for inst in instances:
        inst_ = inst.clone()
        length = inst.z.shape[1]
        delta_length = max_l - length
        
        mask = torch.ones([1, max_l])
        mask[:, length:] = 0
        inst_.mask = mask.bool()
        inst_.z = torch.cat([inst.z, inst.z.new_zeros([inst.z.shape[0], delta_length])], 1)
        inst_.y = torch.cat([inst.y, inst.y.new_zeros([delta_length, 1])], 0)
        
        try:
            del inst_.z_single
            del inst_.y_single
        except:
            pass
        instances_.append(inst_)

    max_drugs = max(inst.x.shape[0] for inst in instances)
    mol_batches = []
    for pos in range(max_drugs):
        graphs, mask_list = [], []
        for inst in instances:
            drug_indices = inst.x.squeeze(-1).long().tolist()
            if isinstance(drug_indices, int):
                drug_indices = [drug_indices]
            if pos < len(drug_indices):
                graphs.append(drug_graph_dict[drug_indices[pos]])
                mask_list.append(True)
            else:
                mask_list.append(False)
        mol_batches.append({
            'batch': Batch.from_data_list(graphs),
            'mask':  torch.tensor(mask_list)
        })

    batch = Batch.from_data_list(instances_)
    batch.mol_batches = mol_batches
    return batch

def batch_instances_embedding(instances, drug_embedding_dict):
    """
    same logig of batch_instance_graph but for tensor-based embedding
    """
    max_l = max(inst.z.shape[1] for inst in instances)
    instances_ = []
    for inst in instances:
        inst_ = inst.clone()
        length = inst.z.shape[1]
        delta_length = max_l - length
        mask = torch.ones([1, max_l])
        mask[:, length:] = 0
        inst_.mask = mask.bool()
        inst_.z = torch.cat([inst.z, inst.z.new_zeros([inst.z.shape[0], delta_length])], 1)
        inst_.y = torch.cat([inst.y, inst.y.new_zeros([delta_length, 1])], 0)
        try:
            del inst_.z_single
            del inst_.y_single
        except Exception:
            pass
        instances_.append(inst_)

    max_drugs = max(inst.x.shape[0] for inst in instances)
    mol_batches = []
    for pos in range(max_drugs):
        embeddings, mask_list = [], []
        for inst in instances:
            drug_indices = inst.x.squeeze(-1).long().tolist()
            if isinstance(drug_indices, int):
                drug_indices = [drug_indices]
            if pos < len(drug_indices):
                embeddings.append(drug_embedding_dict[drug_indices[pos]])
                mask_list.append(True)
            else:
                mask_list.append(False)

        if len(embeddings) > 0:
            emb_tensor = torch.stack(embeddings, dim=0)
        else:
            # nessuna istanza ha un farmaco in questa posizione: tensore vuoto
            emb_dim = next(iter(drug_embedding_dict.values())).shape[-1]
            emb_tensor = torch.zeros(0, emb_dim)

        mol_batches.append({
            'emb': emb_tensor,        # (n_istanze_con_farmaco_in_pos, embedding_dim)
            'mask': torch.tensor(mask_list),
        })

    batch = Batch.from_data_list(instances_)
    batch.mol_batches = mol_batches
    return batch

def split_over_drugs(instances, fold, n_folds=10, seed=3558, min_test_idx = 0):
    all_drugs = set()
    for i in range(len(instances)):
        drugs = instances[i].x.squeeze().tolist()
        if type(drugs) != list:
            all_drugs.add(drugs)
        else:
            all_drugs = all_drugs.union(set(drugs))
    np.random.seed(seed)
    all_drugs = np.sort(np.array(list(all_drugs)))
    all_drugs = all_drugs[all_drugs >= min_test_idx]
    np.random.shuffle(all_drugs)
    folds = np.array_split(all_drugs, n_folds)
    test_fold = folds[fold]
    mask = np.ones(len(instances)).astype(bool)
    for i in range(len(instances)):
        drugs = instances[i].x.squeeze().tolist()
        if type(drugs) != list:
            drugs = [drugs]
        train = True
        for d in drugs:
            if d in test_fold:
                train = False
        if not train:
            mask[i] = False
    train = [instances[i] for i in range(len(instances)) if mask[i]]
    test = [instances[i] for i in range(len(instances)) if not mask[i]]
    return train, test

def split_over_cells(instances, fold, n_folds=10, seed=3558, min_test_idx = 0):
    all_cells = set()
    for i in range(len(instances)):
        cells = instances[i].cell_line.squeeze().tolist()
        if type(cells) != list:
            all_cells.add(cells)
        else:
            all_cells = all_cells.union(set(cells))
            
    np.random.seed(seed)
    all_cells = np.sort(np.array(list(all_cells)))
    all_cells = all_cells[all_cells >= min_test_idx]
    np.random.shuffle(all_cells)
    folds = np.array_split(all_cells, n_folds)
    test_fold = folds[fold]
    mask = np.ones(len(instances)).astype(bool)
    for i in range(len(instances)):
        cells = instances[i].cell_line.squeeze().tolist()
        if type(cells) != list:
            cells = [cells]
        train = True
        for d in cells:
            if d in test_fold:
                train = False
        if not train:
            mask[i] = False
    train = [instances[i] for i in range(len(instances)) if mask[i]]
    test = [instances[i] for i in range(len(instances)) if not mask[i]]
    return train, test

def split_over_combinations(instances, fold, n_folds=10, seed=3558, exclude_single_drugs_from_test = True):
    all_combinations = set()
    combinations = []
    s = []
    for i in range(len(instances)):
        drugs = instances[i].x.squeeze().numpy().astype(str).tolist()
        if type(drugs) == list:
            drugs.sort()
            combination = ",".join(drugs)
            all_combinations.add(combination)
            combinations += [combination]
        else:
            drugs = [drugs]
            combination = ",".join(drugs)
            if not exclude_single_drugs_from_test:
                all_combinations.add(combination)
            combinations += [combination]
    np.random.seed(seed)
    all_combinations = np.sort(np.array(list(all_combinations)))
    np.random.shuffle(all_combinations)
    folds = np.array_split(all_combinations, n_folds)
    test_fold = folds[fold]
    mask = np.ones(len(instances)).astype(bool)
    for i, c in enumerate(combinations):
        if c in test_fold:
            mask[i] = False
    train = [instances[i] for i in range(len(instances)) if mask[i]]
    test = [instances[i] for i in range(len(instances)) if not mask[i]]
    return train, test