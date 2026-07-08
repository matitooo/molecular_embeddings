import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from graph_utils import *




def masked_mse(pred, target, mask):
    """
    Filters MSE with mask
    """
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum()


def batch_instances_graph(instances, drug_graph_dict):
    """
    Creates batched instances when a graph model is selected.
    Produces:
        batch.z          -> (B, max_drugs, max_exp)
        batch.mask       -> (B, max_exp)
        batch.drug_mask  -> (B, max_drugs)
        batch.mol_batches
    """

    B = len(instances)

    max_exp = max(inst.z.shape[1] for inst in instances)
    max_drugs = max(inst.z.shape[0] for inst in instances)

    instances_ = []

    z_tensor = []
    drug_mask = []

    for inst in instances:
        inst_ = inst.clone()

        n_drugs = inst.z.shape[0]
        L = inst.z.shape[1]

        # -------------------------
        # prediction mask
        # -------------------------
        mask = torch.zeros(max_exp, dtype=torch.bool)
        mask[:L] = True
        inst_.mask = mask.unsqueeze(0)

        # -------------------------
        # pad y
        # -------------------------
        if L < max_exp:
            inst_.y = torch.cat([
                inst.y,
                inst.y.new_zeros(max_exp - L, 1)
            ], dim=0)

        # -------------------------
        # build padded z separately
        # -------------------------
        z_pad = inst.z.new_zeros(max_drugs, max_exp)
        z_pad[:n_drugs, :L] = inst.z
        z_tensor.append(z_pad)

        # -------------------------
        # drug mask
        # -------------------------
        dmask = torch.zeros(max_drugs, dtype=torch.bool)
        dmask[:n_drugs] = True
        drug_mask.append(dmask)

        # remove fields we rebuild
        try:
            del inst_.z
        except AttributeError:
            pass

        try:
            del inst_.z_single
            del inst_.y_single
        except AttributeError:
            pass

        instances_.append(inst_)

    # -------------------------
    # molecule batches
    # -------------------------
    mol_batches = []

    for pos in range(max_drugs):
        graphs = []
        mask_list = []

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
            "batch": Batch.from_data_list(graphs),
            "mask": torch.tensor(mask_list, dtype=torch.bool)
        })

    batch = Batch.from_data_list(instances_)

    # overwrite with our tensors
    batch.z = torch.stack(z_tensor, dim=0)          # (B,max_drugs,max_exp)
    batch.drug_mask = torch.stack(drug_mask, dim=0) # (B,max_drugs)
    batch.mol_batches = mol_batches

    return batch

def batch_instances_embedding(instances, drug_embedding_dict):
    """
    Batched version for precomputed drug embeddings.
    Produces:
        batch.z -> (B, max_drugs, max_exp)
        batch.mask -> (B, max_exp)
        batch.mol_batches -> list of dicts with:
            emb: (B, emb_dim) per slot
            mask: (B,)
    """

    B = len(instances)

    max_l = max(inst.z.shape[1] for inst in instances)
    max_drugs = max(inst.x.shape[0] for inst in instances)

    instances_ = []
    z_tensor = []

    for inst in instances:
        inst_ = inst.clone()

        n_drugs = inst.z.shape[0]
        L = inst.z.shape[1]

        mask = torch.zeros(max_l, dtype=torch.bool)
        mask[:L] = True
        inst_.mask = mask.unsqueeze(0)

        if L < max_l:
            inst_.y = torch.cat(
                [inst.y, inst.y.new_zeros(max_l - L, 1)],
                dim=0
            )

        z_pad = inst.z.new_zeros(max_drugs, max_l)
        z_pad[:n_drugs, :L] = inst.z
        z_tensor.append(z_pad)

        try:
            del inst_.z
            del inst_.z_single
            del inst_.y_single
        except Exception:
            pass

        instances_.append(inst_)

    mol_batches = []

    for pos in range(max_drugs):
        emb_list = []
        mask_list = []

        for inst in instances:
            drug_indices = inst.x.squeeze(-1).long().tolist()

            if isinstance(drug_indices, int):
                drug_indices = [drug_indices]

            if pos < len(drug_indices):
                emb_list.append(drug_embedding_dict[drug_indices[pos]])
                mask_list.append(True)
            else:
                emb_dim = next(iter(drug_embedding_dict.values())).shape[-1]
                emb_list.append(torch.zeros(emb_dim))
                mask_list.append(False)

        mol_batches.append({
            "emb": torch.stack(emb_list, dim=0),
            "mask": torch.tensor(mask_list, dtype=torch.bool),
        })

    batch = Batch.from_data_list(instances_)
    batch.z = torch.stack(z_tensor, dim=0)
    batch.mol_batches = mol_batches

    return batch