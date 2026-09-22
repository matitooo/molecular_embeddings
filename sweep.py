import optuna
import yaml
from models.prediction_model import DrugCombinationModel, MoleculeGraphEncoder, DrugCombinationModelWithPrecomputedEmbedding
from datasets import DropArray
from functools import partial
from utils import batch_instances_graph, batch_instances_embedding, compute_base_path_sweep, create_yaml_from_params, dump_study_statistics
import torch
from model_utils import train_loop
from model_utils import eval
from graph_utils import return_dicts
import os
import json


def objective_graph(trial, debug_flag=False):
    with open('config/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)['graph']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = trial.suggest_int(
        "batch_size",
        sweep_config['batch_size']['min'],
        sweep_config['batch_size']['max']
    )

    lr = trial.suggest_float(
        "lr",
        sweep_config['lr']['min'],
        sweep_config['lr']['max'],
        log=True
    )

    n_epochs = trial.suggest_int(
        "n_epochs",
        sweep_config['n_epochs']['min'],
        sweep_config['n_epochs']['max']
    )

    embedding_dim = trial.suggest_int(
        "embedding_dim",
        sweep_config['embedding_dim']['min'],
        sweep_config['embedding_dim']['max']
    )

    hidden_dim = trial.suggest_int(
        "hidden_dim",
        sweep_config['hidden_dim']['min'],
        sweep_config['hidden_dim']['max']
    )

    print('Loading Dataset and Vectorizing Molecules')

    if debug_flag:
        dataset = DropArray('data/debug_dataset.pt')
    else:
        dataset = DropArray(sweep_config['dataset_path'])

    collate_fn = partial(
        batch_instances_graph,
        drug_graph_dict=dataset.drug_graph_dict
    )

    train, test = dataset.get_split(
        how="new_drugs",
        fold=0,
        n_folds=10
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True
    )

    with open('config/graph_config.yaml', 'r') as f:
        graph_config = yaml.safe_load(f)

    size_dict = return_dicts()['size_dict']
    node_dim = sum(
        size_dict[k]
        for k in graph_config.keys()
        if graph_config[k]
    )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_params = [
        batch_size,
        lr,
        n_epochs,
        embedding_dim,
        hidden_dim
    ]

    base_path = compute_base_path_sweep(
        model_type='graph',
        params=run_params
    )

    trained_model, score = train_loop(
        model,
        optimizer,
        device,
        train_loader,
        test_loader,
        n_epochs,
        base_path=base_path
    )

    return score


def objective_trimnet(trial, debug_flag=False):
    with open('config/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)['trimnet']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = trial.suggest_int(
        "batch_size",
        sweep_config['batch_size']['min'],
        sweep_config['batch_size']['max']
    )

    lr = trial.suggest_float(
        "lr",
        sweep_config['lr']['min'],
        sweep_config['lr']['max'],
        log=True
    )

    n_epochs = trial.suggest_int(
        "n_epochs",
        sweep_config['n_epochs']['min'],
        sweep_config['n_epochs']['max']
    )

    hidden_dim = trial.suggest_int(
        "hidden_dim",
        sweep_config['hidden_dim']['min'],
        sweep_config['hidden_dim']['max']
    )

    print('Loading Dataset and Vectorizing Molecules')

    if debug_flag:
        dataset = DropArray(
            'data/debug_dataset.pt',
            model='trimnet'
        )
    else:
        dataset = DropArray(
            sweep_config['dataset_path'],
            model='trimnet'
        )

    collate_fn = partial(
        batch_instances_embedding,
        drug_embedding_dict=dataset.drug_embedding_dict
    )

    train, test = dataset.get_split(
        how="new_drugs",
        fold=0,
        n_folds=10
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True
    )

    model = DrugCombinationModelWithPrecomputedEmbedding(
        embedding_dim=64,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_params = [
        batch_size,
        lr,
        n_epochs,
        hidden_dim
    ]

    base_path = compute_base_path_sweep(
        model_type='trimnet',
        params=run_params
    )

    trained_model, score = train_loop(
        model,
        optimizer,
        device,
        train_loader,
        test_loader,
        n_epochs,
        base_path=base_path
    )

    return score


def objective_3d_infomax(trial, debug_flag=False):
    with open('config/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)['3d_infomax']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = trial.suggest_int(
        "batch_size",
        sweep_config['batch_size']['min'],
        sweep_config['batch_size']['max']
    )

    lr = trial.suggest_float(
        "lr",
        sweep_config['lr']['min'],
        sweep_config['lr']['max'],
        log=True
    )

    n_epochs = trial.suggest_int(
        "n_epochs",
        sweep_config['n_epochs']['min'],
        sweep_config['n_epochs']['max']
    )

    hidden_dim = trial.suggest_int(
        "hidden_dim",
        sweep_config['hidden_dim']['min'],
        sweep_config['hidden_dim']['max']
    )

    print('Loading Dataset and Vectorizing Molecules')

    if debug_flag:
        dataset = DropArray(
            'data/debug_dataset.pt',
            model='3d_infomax'
        )
    else:
        dataset = DropArray(
            sweep_config['dataset_path'],
            model='3d_infomax'
        )

    collate_fn = partial(
        batch_instances_embedding,
        drug_embedding_dict=dataset.drug_embedding_dict
    )

    train, test = dataset.get_split(
        how="new_drugs",
        fold=0,
        n_folds=10
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True
    )

    model = DrugCombinationModelWithPrecomputedEmbedding(
        embedding_dim=256,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_params = [
        batch_size,
        lr,
        n_epochs,
        hidden_dim
    ]

    base_path = compute_base_path_sweep(
        model_type='3d_infomax',
        params=run_params
    )

    trained_model, score = train_loop(
        model,
        optimizer,
        device,
        train_loader,
        test_loader,
        n_epochs,
        base_path=base_path
    )

    return score


def run_sweep(model_type, debug_flag=False):
    with open("config/sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)[model_type]

    study = optuna.create_study(direction="minimize")

    objectives = {
        "graph": objective_graph,
        "trimnet": objective_trimnet,
        "3d_infomax": objective_3d_infomax,
    }

    if model_type not in objectives:
        raise ValueError(f"Unknown model_type: {model_type}")

    study.optimize(
        lambda trial: objectives[model_type](
            trial,
            debug_flag=debug_flag
        ),
        n_trials=sweep_config["n_trials"]
    )

    output_path = f"best_configurations/{model_type}_best_params.json"

    os.makedirs("best_configurations", exist_ok=True)

    result = {
        "model_type": model_type,
        "best_score": study.best_value,
        "best_params": study.best_params,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    create_yaml_from_params(output_path)
    dump_study_statistics(
        study,
        model_type,
        "sweep_statistics"
    )

    print(f"Saved best configuration to: {output_path}")