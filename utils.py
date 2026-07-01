import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from graph_utils import *


def collate_fn(data_list):
    """
    Collate Fn function for data loader, takes the list of data objects as input and returns the dictionary to build the loader with
    """
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
    """
    Filters MSE with mask
    """
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum()


def batch_instances_graph(instances, drug_graph_dict):
    """
    Creates batched instances when a graph model is selected
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
    Same logic of batch_instance_graph but for tensor-based embedding
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
            emb_dim = next(iter(drug_embedding_dict.values())).shape[-1]
            emb_tensor = torch.zeros(0, emb_dim)

        mol_batches.append({
            'emb': emb_tensor,
            'mask': torch.tensor(mask_list),
        })

    batch = Batch.from_data_list(instances_)
    batch.mol_batches = mol_batches
    return batch