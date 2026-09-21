import argparse
from sweep import run_sweep
from train import run_train


def train_mode(model_type,k_fold=False,custom_config= None):
    run_train(model_type,k_fold,custom_config)

def sweep_mode(model_type):
    run_sweep(model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose mode")
    parser.add_argument('--train', action='store_true',
                        help="Train and compare models")
    parser.add_argument('--sweep', action='store_true',
                        help="Find the best Hyperparameters configuration")
    parser.add_argument(
    "--model",
    type=str,
    choices=["3d_infomax", "trimnet",'graph'],
    required=True,
    help="Choose model type"
    )
    parser.add_argument(
    "--config",
    type=str,
    required=False,
    help="Choose model type")

    parser.add_argument("--kfold",action='store_true',help="Perform K-fold validation")
    args = parser.parse_args()
    if not args.config:
        args.config = None
    k_fold = True if args.kfold else False
    if args.train:
        train_mode(args.model,k_fold,args.config)
    elif args.sweep:
        sweep_mode(args.model)