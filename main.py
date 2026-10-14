import argparse

from debug_utils import configure_debug_logging, stop_debug_logging
from sweep import run_sweep
from train import run_train


def train_mode(model_type, k_fold=False, custom_config=None, debug_flag=False, smoke_test=False):
    try:
        run_train(model_type, k_fold, custom_config, debug_flag, smoke_test)
    finally:
        if debug_flag:
            stop_debug_logging()


def sweep_mode(model_type, debug_flag=False, smoke_test=False):
    try:
        run_sweep(model_type, debug_flag, smoke_test)
    finally:
        if debug_flag:
            stop_debug_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models or run hyperparameter sweeps")
    parser.add_argument("--train", action="store_true", help="Train and compare models")
    parser.add_argument("--sweep", action="store_true", help="Find the best hyperparameter configuration")
    parser.add_argument("--model", type=str, choices=["3d_infomax", "trimnet", "graph"],
                        required=True, help="Choose model type")
    parser.add_argument("--config", type=str, required=False, help="Optional training config path")
    parser.add_argument("--kfold", action="store_true", help="Perform K-fold validation")
    parser.add_argument("--debug", action="store_true",
                        help="Run the full configured workflow with verbose diagnostics and periodic stack dumps")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a reduced one-epoch test using data/debug_dataset.pt")
    args = parser.parse_args()

    if args.debug and args.smoke_test:
        parser.error("--debug and --smoke-test are separate modes; choose one")
    if not args.train and not args.sweep:
        parser.error("choose --train or --sweep")
    if args.train and args.sweep:
        parser.error("choose only one of --train or --sweep")

    if args.debug:
        configure_debug_logging()
    if args.train:
        train_mode(args.model, args.kfold, args.config, args.debug, args.smoke_test)
    else:
        sweep_mode(args.model, args.debug, args.smoke_test)
