import argparse
from sweep import run_sweep
from train import run_train
def train_mode():
    run_train()

def sweep_mode():
    run_sweep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose mode")
    parser.add_argument('--train', action='store_true',
                        help="Train and compare models")
    parser.add_argument('--sweep', action='store_true',
                        help="Find the best Hyperparameters configuration")
    args = parser.parse_args()
    if args.train:
        train_mode()
    elif args.sweep:
        sweep_mode()