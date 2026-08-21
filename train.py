import yaml
from functools import partial
from datasets import DropArray
from utils import batch_instances_graph,batch_instances_embedding
from models.prediction_model import MoleculeGraphEncoder,DrugCombinationModel,DrugCombinationModelWithPrecomputedEmbedding
import torch
from model_utils import train_loop
from graph_utils import return_dicts
import os

def run_train(model_type, k_fold=False):
    # load config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = 'config/train.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # load and preprocess data
    print('Loading Dataset and Vectorizing Molecules')

    try:
        dataset = torch.load(
            config['vectorized_dataset_path'],
            weights_only=False
        )
    except:
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
        os.makedirs('trained_model_weights', exist_ok=True)

        for fold in range(10):

            train, test = dataset.get_split(
                how="new_drugs",
                fold=fold
            )

            train_loader = torch.utils.data.DataLoader(
                train,
                batch_size=128,
                num_workers=0,
                collate_fn=collate_fn,
                shuffle=True
            )

            test_loader = torch.utils.data.DataLoader(
                test,
                batch_size=128,
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

            print(f"\nNow training fold: {fold + 1} of 10")
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
                config['n_epochs']
            )

            # ----------------------------------------------------
            # Save model
            # ----------------------------------------------------
            w_path = f'trained_model_weights/{fold}.pt'

            torch.save(
                trained_model.state_dict(),
                w_path
            )

            scores[fold] = val_loss

            print(f'Fold {fold + 1} completed')
            print(f'Validation loss: {val_loss}')

        # --------------------------------------------------------
        # Save scores
        # --------------------------------------------------------
        with open('scores.txt', 'w') as f:
            print(scores, file=f)

        print('\nK-fold training completed')
        print('Scores:', scores)

    # ============================================================
    # TRAINING WITHOUT K-FOLD
    # ============================================================
    else:

        print('\nTraining without k-fold')

        # --------------------------------------------------------
        # Train/test split
        # --------------------------------------------------------
        train, test = dataset.get_split(
            how="new_drugs"
        )

        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=128,
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test,
            batch_size=128,
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=False
        )

        # --------------------------------------------------------
        # Create model
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr']
        )

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------
        print('Training Model')

        trained_model, val_loss = train_loop(
            model,
            optimizer,
            device,
            train_loader,
            test_loader,
            config['n_epochs']
        )

        # --------------------------------------------------------
        # Save model
        # --------------------------------------------------------
        os.makedirs('trained_model_weights', exist_ok=True)

        w_path = f'trained_model_weights/{model_type}.pt'

        torch.save(
            trained_model.state_dict(),
            w_path
        )

        # --------------------------------------------------------
        # Save validation loss
        # --------------------------------------------------------
        with open(
            f'trained_model_weights/{model_type}_score.txt',
            'w'
        ) as f:
            print(val_loss, file=f)

        print('\nTraining completed')
        print(f'Validation loss: {val_loss}')
        print(f'Model saved to: {w_path}')

