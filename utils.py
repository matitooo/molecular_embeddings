import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from graph_utils import *
import json
import os
import yaml
import optuna
from datetime import datetime




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


def compute_base_path(model_type,config,fold = None):
    pth = 'results/predictions/'
    pth += model_type
    pth += "/"
    for value in config.values():
        if type(value) == int or type(value) == float:
            pth += "_"
            pth += str(value)
    if fold:
        pth+="/fold_"+str(fold)
    return pth



def compute_base_path_sweep(model_type,params):
    pth = 'results/sweep_predictions/'
    pth += model_type
    pth += "/"
    for value in params:
        if type(value) == int or type(value) == float:
            pth += "_"
            pth += str(value)
    return pth



def create_yaml_from_params(params_path):
    with open(params_path, 'r') as f:
        data = json.load(f)

    config = data["best_params"]

    name = f"results/best_configurations/{data['model_type']}_config.yaml"
    with open(name, "w") as f:
        yaml.dump(config, f, sort_keys=False)



def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def dump_study_statistics(study, model_type, output_dir="results/sweep_statistics"):
    os.makedirs(output_dir, exist_ok=True)

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
    ]

    try:
        param_importances = optuna.importance.get_param_importances(study)
    except Exception:
        param_importances = {}

    trials_data = []

    best_so_far = None

    for trial in study.trials:

        if trial.value is not None:
            if best_so_far is None:
                best_so_far = trial.value
            else:
                best_so_far = min(best_so_far, trial.value)

        trial_data = {
            "number": trial.number,
            "state": trial.state.name,
            "score": _json_safe(trial.value),
            "params": {
                k: _json_safe(v)
                for k, v in trial.params.items()
            },
            "distributions": {
                name: str(distribution)
                for name, distribution in trial.distributions.items()
            },
            "datetime_start": (
                trial.datetime_start.isoformat()
                if trial.datetime_start else None
            ),
            "datetime_complete": (
                trial.datetime_complete.isoformat()
                if trial.datetime_complete else None
            ),
            "duration_seconds": (
                trial.duration.total_seconds()
                if trial.duration else None
            ),
            "best_score_so_far": _json_safe(best_so_far),
            "user_attrs": {
                k: _json_safe(v)
                for k, v in trial.user_attrs.items()
            },
            "system_attrs": {
                k: _json_safe(v)
                for k, v in trial.system_attrs.items()
            }
        }

        trials_data.append(trial_data)

    scores = np.array(
        [t.value for t in completed_trials],
        dtype=float
    )

    if len(scores) > 0:
        score_statistics = {
            "count": int(len(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "variance": float(np.var(scores)),
            "q25": float(np.quantile(scores, 0.25)),
            "q75": float(np.quantile(scores, 0.75)),
        }
    else:
        score_statistics = {}

    result = {
        "model_type": model_type,

        "study": {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "sampler": type(study.sampler).__name__,
            "pruner": type(study.pruner).__name__,
            "n_trials": len(study.trials),
            "n_completed_trials": len(completed_trials),
        },

        "best_trial": {
            "number": study.best_trial.number,
            "score": _json_safe(study.best_value),
            "params": {
                k: _json_safe(v)
                for k, v in study.best_params.items()
            }
        },

        "score_statistics": score_statistics,

        "parameter_importance": {
            k: float(v)
            for k, v in param_importances.items()
        },

        "trials": trials_data
    }

    output_path = os.path.join(
        output_dir,
        f"{model_type}_sweep_statistics.json"
    )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved sweep statistics to: {output_path}")

    return output_path