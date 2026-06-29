import yaml
from functools import partial
from datasets import DropArray
from utils import batch_instances_graph,batch_instances_embedding
from models.prediction_model import MoleculeGraphEncoder,DrugCombinationModel,DrugCombinationModelWithPrecomputedEmbedding
import torch
from model_utils import train_loop
from graph_utils import return_dicts


def run_train(model_type):
  #load config
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  config_path = 'config/train.yaml'
  with open(config_path,'r') as f:
    config = yaml.safe_load(f)

  #load and preprocess data
  print('Loading Dataset and Vectorizing Molecules')
  try:
    dataset = torch.load(config['vectorized_dataset_path'],weights_only= False)
  except:
    dataset = DropArray(config['dataset_path'],model=model_type)
  
  if model_type =='trimnet' or model_type=='3d_infomax':
    collate_fn = partial(batch_instances_embedding, drug_embedding_dict=dataset.drug_embedding_dict)
  elif model_type=='graph':
    collate_fn = partial(batch_instances_graph, drug_graph_dict=dataset.drug_graph_dict)
  
  train, test = dataset.get_split(how="new_drugs", fold=0)

  train_loader = torch.utils.data.DataLoader(
      train, batch_size=128, num_workers=0,
      collate_fn=collate_fn, shuffle=True
  )
  test_loader = torch.utils.data.DataLoader(
      test, batch_size=128, num_workers=0,
      collate_fn=collate_fn, shuffle=True
)
  
  #compute node_dim_size
  if model_type=='graph':
    with open('config/graph_config.yaml','r') as f:
      graph_config = yaml.safe_load(f)
    
    size_dict = return_dicts()['size_dict']
    node_dim = sum(size_dict[k] for k in graph_config.keys() if graph_config[k])
    #create and configure model and optimizer
    mol_encoder = MoleculeGraphEncoder(
      node_dim=node_dim,
      hidden_dim=config['hidden_dim'],
      embedding_dim=config['embedding_dim'],
      num_layers=4
  )

    model = DrugCombinationModel(
        mol_encoder=mol_encoder,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
  
  else:
    if model_type =='trimnet':
      embedding_dim = 64
    elif model_type=='3d_infomax':
      embedding_dim=256

    model = DrugCombinationModelWithPrecomputedEmbedding(embedding_dim=embedding_dim,hidden_dim=config['hidden_dim'])

 

  print(f'Models created with the following parameters')
  print(model.parameters())

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])


  #execute train
  print('Training Model')
  train_loop(model,optimizer,device,train_loader,test_loader,config['n_epochs'])

  print('Training Completed')

