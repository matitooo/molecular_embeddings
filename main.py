import argparse
from sweep import run_sweep
from train import run_train


def train_mode(model_type):
    run_train(model_type)

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
    args = parser.parse_args()
    if args.train:
        train_mode(args.model)
    elif args.sweep:
        sweep_mode(args.model)