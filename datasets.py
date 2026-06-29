import torch
import os
from graph_utils import smiles_to_data,return_dicts
from tqdm import tqdm 
import numpy as np
from utils import *
from trimnet_utils import Inference_trimnet
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
    def __init__(self,dataset_path,model='graph'):
        self.dataset = torch.load(dataset_path,weights_only = False)
        self.C = None
        self.num_c_embeddings = len(self.dataset["cell_map"])
        self.target_dim = 1
        self.min_test_idx = 25
        self.other_features = False
        self.dicts = return_dicts()
        self.smiles_dict = dict(zip(self.dataset['smiles']['drug_id'],self.dataset['smiles']['SMILES']))
        if model == 'graph':
            self.drug_graph_dict = {
                idx: smiles_to_data(smi, self.dicts)
                for idx, smi in tqdm(self.smiles_dict.items())
            }
        elif model =='3d_infomax':
            inf_model = Inference_3d_infomax()
            embedded_molecules = inf_model.pipe(self.smiles_dict.values())
            self.drug_embedding_dict = {
                list(self.smiles_dict.keys())[i]: embedded_molecules[i] for i in range(len(embedded_molecules)) 
            }
        elif model =='trimnet':
            inf_model = Inference_trimnet()
            embedded_molecules = inf_model.pipe(self.smiles_dict.values())
            self.drug_embedding_dict = {
                 list(self.smiles_dict.keys())[i]: embedded_molecules[i] for i in range(len(embedded_molecules)) 
            }

        
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