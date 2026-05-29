import torch 
from torch.utils.data import DataLoader
from infomax_utils import InferenceDataset
import argparse
import os
import yaml
from infomax_utils import seed_all
import dgl
import yaml
from types import SimpleNamespace
from infomax_utils import graph_only_collate
from models.infomax import PNA
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from models.trimnet import TrimNet
from trimnet_utils import smiles_to_data
from tqdm import tqdm
import pandas as pd
import os



class Inference_3d_infomax:
    def __init__(self):
        with open("3d_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.args = args = SimpleNamespace(**config)

    def pipe(self):
        seed_all(self.args.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.device == 'cuda' else "cpu")
        self.test_data = InferenceDataset(device=self.device, smiles_txt_path=self.args.smiles_txt_path)
        print('num_smiles: ', len(self.test_data))
        model, _,_ = self.load_model()
        print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad), '\n')
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device,weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test_loader = DataLoader(self.test_data, batch_size=1, collate_fn=graph_only_collate)
        fingerprints_list = []
        for i, batch in enumerate(test_loader):
            fingerprints_list.append(model(batch))

        os.makedirs(self.args.out_path, exist_ok=True)
        path = os.path.join(self.args.out_path, f'infomax_embeddings.pt')
        print(f'Embeddings saved in {path}')
        torch.save({'fingerprints': torch.cat(fingerprints_list, dim=0)}, path)
        return None

    def load_model(self):
        model = PNA(avg_d=self.test_data.avg_degree if hasattr(self.test_data, 'avg_degree') else 1, device=self.device,
                                        **self.args.model_parameters)
        pretrained_gnn_dict = {}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict) 
        model.load_state_dict(model_state_dict)
        return model, None, False

class Inference_trimnet:

    def __init__(self):
        self.ckpt_path = 'checkpoints/trimnet_checkpoint.ckpt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ckpt = torch.load(self.ckpt_path, weights_only=False)
        self.opt  = self.ckpt['option']
        self.smiles_path = 'data/smiles_200.txt'
        self.save_path_parent_folder = 'out_smiles'
        self.save_path = 'out_smiles/trimnet_embeddings.pt'
        with open ('data/smiles_200.txt','r') as f:
            self.smiles_list = [s for s in f.read().splitlines() if s.strip()]

    def load_model(self):
        self.model = TrimNet(
            in_dim      = self.opt['in_dim'],
            edge_in_dim = self.opt['edge_in_dim'],
            hidden_dim  = self.opt['hid'],
            depth       = self.opt['depth'],
            heads       = self.opt['heads'],
            dropout     = self.opt['dropout'],
            outdim      = self.opt['out_dim'],
        )
        self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.model.eval().to(self.device)
        return None
    
    def embed_smiles(self):
        all_embeddings = []
        self.model.eval() 
        for smi in tqdm(self.smiles_list):
            emb = self.predict(smi)  # tensor
            all_embeddings.append(emb)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        os.makedirs(self.save_path_parent_folder,exist_ok=True)
        torch.save({'fingerprints': all_embeddings}, self.save_path)
        print(f'Embeddings saved in: {self.save_path}')
        return None


    def predict(self, smi):
        graph = smiles_to_data(smi)
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            embedding = self.model.embed(batch)
        return embedding.detach().cpu()
    
    def pipe(self):
        self.load_model()
        self.embed_smiles()
