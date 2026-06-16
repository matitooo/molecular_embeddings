import optuna
import yaml
from model import DrugCombinationModel,MoleculeGraphEncoder
from datasets import DropArray
from functools import partial
from utils import batch_instances_graph
import torch
from train_utils import train_loop
from eval import eval
from graph_utils import return_dicts

with open('config/sweep.yaml','r') as f:
   sweep_config = yaml.safe_load(f)

def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = trial.suggest_categorical("batch_size",sweep_config['batch_size'])
    lr = trial.suggest_categorical("lr",sweep_config['lr'])
    n_epochs = trial.suggest_categorical("n_epochs",sweep_config['n_epochs'])
    embedding_dim = trial.suggest_categorical("embedding_dim",sweep_config['embedding_dim'])
    hidden_dim = trial.suggest_categorical("hidden_dim",sweep_config['hidden_dim'])

    #load and preprocess data
    print('Loading Dataset and Vectorizing Molecules')
    try:
      dataset = torch.load(sweep_config['vectorized_dataset_path'],weights_only= False)
    except:
      dataset = DropArray(sweep_config['dataset_path'])
    
    collate_fn = partial(batch_instances_graph, drug_graph_dict=dataset.drug_graph_dict)

    train, test = dataset.get_split(how="new_drugs", fold=0)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, num_workers=0,
        collate_fn=collate_fn, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, num_workers=0,
        collate_fn=collate_fn, shuffle=True
  )
    

    #compute node_dim_size
    with open('config/graph_config.yaml','r') as f:
      graph_config = yaml.safe_load(f)
    
    size_dict = return_dicts()['size_dict']
    node_dim = sum(size_dict[k] for k in graph_config.keys() if graph_config[k])

    #create and configure model and optimizer
    mol_encoder = MoleculeGraphEncoder(
      node_dim=node_dim,
      hidden_dim=hidden_dim,
      embedding_dim=embedding_dim,
      num_layers=4
  )

    model = DrugCombinationModel(
        mol_encoder=mol_encoder,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    trained_model = train_loop(model,optimizer,device,train_loader,test_loader,n_epochs)

    score = eval(trained_model,test_loader)

    return score



def run_sweep():
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=10)

  print("Best parameters:", study.best_params)
  print("Best score:", study.best_value)