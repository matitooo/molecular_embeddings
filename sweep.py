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

    batch_size = trial.suggest_categorical('batch_size',[sweep_config['batch_size']])

    lr = trial.suggest_float(
        "lr",
        sweep_config['lr']['min'],
        sweep_config['lr']['max'],
        log=True
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
        dataset_path = trial.suggest_categorical('dataset_path',['data/debug_dataset.pt'])
        dataset = DropArray(
            'data/debug_dataset.pt'
        )
        n_epochs = trial.suggest_categorical('n_epochs',[1])
    else:
        dataset_path = trial.suggest_categorical('dataset_path',[sweep_config['dataset_path']])
        dataset = DropArray(
            sweep_config['dataset_path']
        )
        n_epochs = trial.suggest_categorical('n_epochs',[sweep_config['n_epochs']])

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

    batch_size = trial.suggest_categorical('batch_size',[sweep_config['batch_size']])

    lr = trial.suggest_float(
        "lr",
        sweep_config['lr']['min'],
        sweep_config['lr']['max'],
        log=True
    )


    hidden_dim = trial.suggest_int(
        "hidden_dim",
        sweep_config['hidden_dim']['min'],
        sweep_config['hidden_dim']['max']
    )

    print('Loading Dataset and Vectorizing Molecules')

    if debug_flag:
        dataset_path = trial.suggest_categorical('dataset_path',['data/debug_dataset.pt'])
        dataset = DropArray(
            'data/debug_dataset.pt',
            model='trimnet'
        )
        n_epochs = trial.suggest_categorical('n_epochs',[1])
    else:
        dataset_path = trial.suggest_categorical('dataset_path',[sweep_config['dataset_path']])
        dataset = DropArray(
            sweep_config['dataset_path'],
            model='trimnet'
        )
        n_epochs = trial.suggest_categorical('n_epochs',[sweep_config['n_epochs']])

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

    batch_size = trial.suggest_categorical('batch_size',[sweep_config['batch_size']])

    lr = trial.suggest_float(
        "lr",
        sweep_config['lr']['min'],
        sweep_config['lr']['max'],
        log=True
    )

    hidden_dim = trial.suggest_int(
        "hidden_dim",
        sweep_config['hidden_dim']['min'],
        sweep_config['hidden_dim']['max']
    )

    print('Loading Dataset and Vectorizing Molecules')

    if debug_flag:
        dataset_path = trial.suggest_categorical('dataset_path',['data/debug_dataset.pt'])
        dataset = DropArray(
            'data/debug_dataset.pt',
            model='3d_infomax'
        )
        n_epochs = trial.suggest_categorical('n_epochs',[1])
    else:
        dataset_path = trial.suggest_categorical('dataset_path',[sweep_config['dataset_path']])
        dataset = DropArray(
            sweep_config['dataset_path'],
            model='3d_infomax'
        )
        n_epochs = trial.suggest_categorical('n_epochs',[sweep_config['n_epochs']])

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
    with open('config/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)[model_type]

    objectives = {
        'graph': objective_graph,
        'trimnet': objective_trimnet,
        '3d_infomax': objective_3d_infomax,
    }

    if model_type not in objectives:
        raise ValueError(f'Unknown model_type: {model_type}')

    os.makedirs('results/optuna', exist_ok=True)
    os.makedirs('results/best_configurations', exist_ok=True)

    study_name = (
        f'{model_type}_new_drugs_fold0_debug'
        if debug_flag
        else f'{model_type}_new_drugs_fold0'
    )

    study = optuna.create_study(
        study_name=study_name,
        storage='sqlite:///results/optuna/optuna.db',
        load_if_exists=True,
        direction='minimize'
    )

    target_trials = 1 if debug_flag else sweep_config['n_trials']

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )

    remaining_trials = max(
        0,
        target_trials - completed_trials
    )

    print(f'Study: {study_name}')
    print(f'Completed trials: {completed_trials}/{target_trials}')
    print(f'Remaining trials: {remaining_trials}')

    if remaining_trials > 0:
        study.optimize(
            lambda trial: objectives[model_type](
                trial,
                debug_flag=debug_flag
            ),
            n_trials=remaining_trials,
            catch=(torch.cuda.OutOfMemoryError,)
        )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]

    if not completed_trials:
        raise RuntimeError(
            f'No completed trials available for study {study_name}.'
        )

    output_path = (
        f'results/best_configurations/'
        f'{model_type}_best_params.json'
    )

    result = {
        'model_type': model_type,
        'study_name': study_name,
        'best_trial': study.best_trial.number,
        'best_score': study.best_value,
        'best_params': study.best_params,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    create_yaml_from_params(output_path)

    dump_study_statistics(
        study,
        model_type,
        'results/sweep_statistics'
    )

    print(f'Best trial: {study.best_trial.number}')
    print(f'Best validation loss: {study.best_value}')
    print(f'Saved best configuration to: {output_path}')