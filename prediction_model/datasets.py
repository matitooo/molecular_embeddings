import torch
from utils import *
import os
from graph_utils import smiles_to_data,return_dicts
from tqdm import tqdm 

   
class TreatmentDataset():
    def __init__(self):
        self.M = None
        self.C = None
    def get_split(self, fold = 0, how = "new_combinations", n_folds=10):
        if how == "new_drugs":
            train, test = split_over_drugs(self.dataset["x"], fold, min_test_idx = self.min_test_idx)
        elif how == "new_combinations":
            train, test = split_over_combinations(self.dataset["x"], fold)
        elif how == "new_cell_lines":
            train, test = split_over_cells(self.dataset["x"], fold) 
        return train, test
    def _cache_drugs(self, cache_path, get_drug_fn):
        if os.path.exists(cache_path):
            self.M = torch.load(cache_path,weights_only = False)
        else:
            self.M = get_drug_fn(self.dataset)
            torch.save(self.M, cache_path)

            
class DropArray(TreatmentDataset):
    def __init__(self,dataset_path):
        self.dataset = torch.load(dataset_path,weights_only = False)
        self.C = None
        self.num_c_embeddings = len(self.dataset["cell_map"])
        self.target_dim = 1
        self.min_test_idx = 25
        self.other_features = False
        self.dicts = return_dicts()
        self.smiles_dict = dict(zip(self.dataset['smiles']['drug_id'],self.dataset['smiles']['SMILES']))
        self.drug_graph_dict = {
              idx: smiles_to_data(smi, self.dicts)
              for idx, smi in tqdm(self.smiles_dict.items())
          }