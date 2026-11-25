import torch
import os
from graph_utils import smiles_to_data,return_dicts
from tqdm import tqdm 
import numpy as np
import logging
import time

logger = logging.getLogger('pipeline.debug.dataset')
from utils import *
from models.trimnet import Inference_trimnet
from models.infomax import Inference_3d_infomax
   
class TreatmentDataset():
    def __init__(self):
        self.M = None
        self.C = None
    def get_split(self, fold = 0, how = "new_combinations", n_folds=10):
        if how == "new_drugs":
            train, test = split_over_drugs(self.dataset["x"], fold, min_test_idx = self.min_test_idx)
        return train, test
    def _cache_drugs(self, cache_path, get_drug_fn):
        if os.path.exists(cache_path):
            self.M = torch.load(cache_path,weights_only = False)
        else:
            self.M = get_drug_fn(self.dataset)
            torch.save(self.M, cache_path)

            
class DropArray(TreatmentDataset):
    def __init__(self, dataset_path, model='graph'):
        started = time.monotonic()
        logger.info('Dataset load start path=%s model=%s', os.path.abspath(dataset_path), model)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f'Dataset file not found: {os.path.abspath(dataset_path)}')
        logger.info('Dataset file size=%.2f MiB', os.path.getsize(dataset_path) / (1024 * 1024))
        self.dataset = torch.load(dataset_path, weights_only=False)
        logger.info('torch.load complete after %.2fs; top-level type=%s', time.monotonic() - started, type(self.dataset).__name__)
        if isinstance(self.dataset, dict):
            logger.info('Dataset keys=%s', list(self.dataset.keys()))
            for key, value in self.dataset.items():
                try:
                    logger.info('Dataset field %s type=%s length=%d', key, type(value).__name__, len(value))
                except (TypeError, AttributeError):
                    logger.info('Dataset field %s type=%s', key, type(value).__name__)

        # Global number of experiments (T) used by every DataLoader batch.
        # Keeping this value at dataset level prevents each batch from
        # choosing a different padded T.
        try:
            self.max_exp = max(
                int(instance.z.shape[1])
                for instance in self.dataset["x"]
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Unable to determine the global number of experiments "
                "from dataset['x']."
            ) from exc

        self.C = None
        self.num_c_embeddings = len(self.dataset["cell_map"])
        self.target_dim = 1
        self.min_test_idx = 25
        self.other_features = False
        self.dicts = return_dicts()
        self.smiles_dict = dict(zip(self.dataset['smiles']['drug_id'], self.dataset['smiles']['SMILES']))
        logger.info('Dataset summary: instances=%d cells=%d unique_molecules=%d max_exp=%d', len(self.dataset['x']), self.num_c_embeddings, len(self.smiles_dict), self.max_exp)
        vectorize_started = time.monotonic()
        if model == 'graph':
            logger.info('Graph vectorization start for %d molecules', len(self.smiles_dict))
            self.drug_graph_dict = {
                idx: smiles_to_data(smi, self.dicts)
                for idx, smi in tqdm(self.smiles_dict.items(), desc='Graph vectorization')
            }
            logger.info('Graph vectorization complete: %d molecules in %.2fs', len(self.drug_graph_dict), time.monotonic() - vectorize_started)
        elif model =='3d_infomax':
            logger.info('3D-Infomax embedding start for %d molecules', len(self.smiles_dict))
            inf_model = Inference_3d_infomax()
            embedded_molecules = inf_model.pipe(self.smiles_dict.values())
            logger.info('3D-Infomax embedding returned %d molecules in %.2fs', len(embedded_molecules), time.monotonic() - vectorize_started)
            self.drug_embedding_dict = {
                list(self.smiles_dict.keys())[i]: embedded_molecules[i] for i in range(len(embedded_molecules)) 
            }
        elif model =='trimnet':
            logger.info('TrimNet embedding start for %d molecules', len(self.smiles_dict))
            inf_model = Inference_trimnet()
            embedded_molecules = inf_model.pipe(self.smiles_dict.values())
            logger.info('TrimNet embedding returned %d molecules in %.2fs', len(embedded_molecules), time.monotonic() - vectorize_started)
            self.drug_embedding_dict = {
                 list(self.smiles_dict.keys())[i]: embedded_molecules[i] for i in range(len(embedded_molecules)) 
            }
        logger.info('Dataset preprocessing complete in %.2fs', time.monotonic() - started)


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