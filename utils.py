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
    Creates a batched batch for the graph model.

    The molecular graphs are batched ONCE per DataLoader batch.
    Repeated molecules are represented by indices into the unique
    molecular graph batch.

    Produces:
        batch.z           -> (B, max_drugs, max_exp)
        batch.mask        -> (B, max_exp)
        batch.drug_mask   -> (B, max_drugs)

        batch.mol_batch   -> PyG Batch containing UNIQUE molecules
        batch.drug_index  -> (B, max_drugs), indices into mol_batch

    For padded drug positions, drug_index is 0 and drug_mask is False.
    """

    B = len(instances)

    max_exp = max(inst.z.shape[1] for inst in instances)
    max_drugs = max(inst.z.shape[0] for inst in instances)

    instances_ = []

    z_tensor = []
    drug_mask = []

    # ---------------------------------------------------------
    # Build a single list of unique drug IDs for this batch
    # ---------------------------------------------------------

    unique_drug_ids = []
    seen_drugs = set()

    instance_drug_ids = []

    for inst in instances:
        drug_indices = (
            inst.x.squeeze(-1)
            .long()
            .tolist()
        )

        if isinstance(drug_indices, int):
            drug_indices = [drug_indices]

        instance_drug_ids.append(drug_indices)

        for drug_id in drug_indices:
            if drug_id not in seen_drugs:
                seen_drugs.add(drug_id)
                unique_drug_ids.append(drug_id)

    # Map global drug ID -> local index in the unique graph batch
    drug_id_to_local = {
        drug_id: i
        for i, drug_id in enumerate(unique_drug_ids)
    }

    # ---------------------------------------------------------
    # Build one index tensor per experiment
    # ---------------------------------------------------------

    drug_index_tensor = []

    for drug_indices in instance_drug_ids:

        n_drugs = len(drug_indices)

        # Padded positions point to 0, but are ignored by drug_mask.
        index_row = torch.zeros(
            max_drugs,
            dtype=torch.long,
        )

        for pos, drug_id in enumerate(drug_indices):
            index_row[pos] = drug_id_to_local[drug_id]

        drug_index_tensor.append(index_row)

    # ---------------------------------------------------------
    # Build padded experiment tensors
    # ---------------------------------------------------------

    for inst, drug_indices in zip(instances, instance_drug_ids):

        inst_ = inst.clone()

        n_drugs = len(drug_indices)
        L = inst.z.shape[1]

        # -----------------------------------------------------
        # Prediction mask
        # -----------------------------------------------------

        mask = torch.zeros(
            max_exp,
            dtype=torch.bool,
        )

        mask[:L] = True

        inst_.mask = mask.unsqueeze(0)

        # -----------------------------------------------------
        # Pad y
        # -----------------------------------------------------

        if L < max_exp:
            inst_.y = torch.cat(
                [
                    inst.y,
                    inst.y.new_zeros(max_exp - L, 1),
                ],
                dim=0,
            )

        # -----------------------------------------------------
        # Pad z
        # -----------------------------------------------------

        z_pad = inst.z.new_zeros(
            max_drugs,
            max_exp,
        )

        z_pad[:n_drugs, :L] = inst.z

        z_tensor.append(z_pad)

        # -----------------------------------------------------
        # Drug mask
        # -----------------------------------------------------

        dmask = torch.zeros(
            max_drugs,
            dtype=torch.bool,
        )

        dmask[:n_drugs] = True

        drug_mask.append(dmask)

        # -----------------------------------------------------
        # Remove fields rebuilt below
        # -----------------------------------------------------

        try:
            del inst_.z
        except AttributeError:
            pass

        try:
            del inst_.z_single
        except AttributeError:
            pass

        try:
            del inst_.y_single
        except AttributeError:
            pass

        instances_.append(inst_)

    # ---------------------------------------------------------
    # ONE PyG graph batch for all UNIQUE molecules
    # ---------------------------------------------------------

    graphs = [
        drug_graph_dict[drug_id]
        for drug_id in unique_drug_ids
    ]

    mol_batch = Batch.from_data_list(graphs)

    # ---------------------------------------------------------
    # Build final experiment batch
    # ---------------------------------------------------------

    batch = Batch.from_data_list(instances_)

    batch.z = torch.stack(
        z_tensor,
        dim=0,
    )  # (B, max_drugs, max_exp)

    batch.drug_mask = torch.stack(
        drug_mask,
        dim=0,
    )  # (B, max_drugs)

    batch.mol_batch = mol_batch

    batch.drug_index = torch.stack(
        drug_index_tensor,
        dim=0,
    )  # (B, max_drugs)

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