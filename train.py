import yaml
from functools import partial
from datasets import DropArray
from utils import batch_instances_graph,batch_instances_embedding,compute_base_path
from models.prediction_model import MoleculeGraphEncoder,DrugCombinationModel,DrugCombinationModelWithPrecomputedEmbedding
import torch
from model_utils import train_loop
from graph_utils import return_dicts
import os
import numpy as np
import json

def run_train(model_type, k_fold=False,custom_config=None,debug_flag=False):
    # load config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if custom_config: 
        config_path = custom_config
    else:
        config_path = 'config/train.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    # load and preprocess data
    print('Loading Dataset and Vectorizing Molecules')
    if debug_flag:
        dataset = DropArray(
            'data/debug_dataset.pt',
            model=model_type
        )
    else:
        dataset = DropArray(
            config['dataset_path'],
            model=model_type
        )

    # define collate function
    if model_type == 'trimnet' or model_type == '3d_infomax':
        collate_fn = partial(
            batch_instances_embedding,
            drug_embedding_dict=dataset.drug_embedding_dict
        )

    elif model_type == 'graph':
        collate_fn = partial(
            batch_instances_graph,
            drug_graph_dict=dataset.drug_graph_dict
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ============================================================
    # K-FOLD TRAINING
    # ============================================================
    if k_fold:

        scores = {}
        os.makedirs(f'results/trained_model_weights/{model_type}', exist_ok=True)

        for fold in range(1,10):
            
            base_path = compute_base_path(model_type,config,int(fold))
            train, test = dataset.get_split(
                how="new_drugs",
                fold=fold,
                n_folds=10
            )

            train_loader = torch.utils.data.DataLoader(
                train,
                batch_size=config['batch_size'],
                num_workers=0,
                collate_fn=collate_fn,
                shuffle=True
            )

            test_loader = torch.utils.data.DataLoader(
                test,
                batch_size=config['batch_size'],
                num_workers=0,
                collate_fn=collate_fn,
                shuffle=False
            )

            # ----------------------------------------------------
            # Create model
            # ----------------------------------------------------
            if model_type == 'graph':

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

                if model_type == 'trimnet':
                    embedding_dim = 64

                elif model_type == '3d_infomax':
                    embedding_dim = 256

                model = DrugCombinationModelWithPrecomputedEmbedding(
                    embedding_dim=embedding_dim,
                    hidden_dim=config['hidden_dim']
                ).to(device)

            # ----------------------------------------------------
            # Optimizer
            # ----------------------------------------------------
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['lr']
            )

            print(f"\nNow training fold: {fold} of 9")
            print('Training Model')

            # ----------------------------------------------------
            # Train
            # ----------------------------------------------------
            trained_model, val_loss = train_loop(
                model,
                optimizer,
                device,
                train_loader,
                test_loader,
                config['n_epochs'],
                base_path=base_path
            )

            # ----------------------------------------------------
            # Save model
            # ----------------------------------------------------
            w_path = f'results/trained_model_weights/{model_type}/{fold}.pt'

            torch.save(
                trained_model.state_dict(),
                w_path
            )

            scores[fold] = val_loss
            print(f'Fold {fold} completed')
            print(f'Validation loss: {val_loss}')

        os.makedirs("results/training_statistics", exist_ok=True)

        scores_values = list(scores.values())

        training_stats = {
            "model_type": model_type,
            "n_folds": len(scores),
            "scores": scores,
            "mean": float(np.mean(scores_values)),
            "std": float(np.std(scores_values, ddof=1)),
            "config": config
        }

        stats_path = f"results/training_statistics/{model_type}.json"

        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=4)

        print(f"\nK-fold training completed")
        print(f"Scores: {scores}")
        print(f"Training statistics saved to: {stats_path}")

    # ============================================================
    # TRAINING WITHOUT K-FOLD
    # ============================================================
    else:

        print('\nTraining without k-fold')

        
        train, test = dataset.get_split(
            how="new_drugs"
        )

        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=config['batch_size'],
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test,
            batch_size=config['batch_size'],
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=False
        )


        if model_type == 'graph':

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

            if model_type == 'trimnet':
                embedding_dim = 64

            elif model_type == '3d_infomax':
                embedding_dim = 256

            model = DrugCombinationModelWithPrecomputedEmbedding(
                embedding_dim=embedding_dim,
                hidden_dim=config['hidden_dim']
            ).to(device)


        base_path = compute_base_path(model_type,config)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr']
        )


        print('Training Model')


        trained_model, val_loss = train_loop(
            model,
            optimizer,
            device,
            train_loader,
            test_loader,
            config['n_epochs'],
            base_path=base_path
        )

        os.makedirs('results/trained_model_weights', exist_ok=True)

        w_path = f'results/trained_model_weights/{model_type}.pt'

        torch.save(
            trained_model.state_dict(),
            w_path
        )

        with open(
            f'results/trained_model_weights/{model_type}_score.txt',
            'w'
        ) as f:
            print(val_loss, file=f)

        print('\nTraining completed')
        print(f'Validation loss: {val_loss}')
        print(f'Model saved to: {w_path}')

