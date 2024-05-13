import argparse

from experiments.experiment1 import run_experiment1
from experiments.experiment2 import run_experiment2
from experiments.experiment3 import run_experiment3


def main():
    # parse command line arguments with "--expt #"
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt", type=int, required=True)
    args = parser.parse_args()

    # Experiment 1
    if args.expt == 1:
        run_experiment1(out_dir="out/expt1", max_iter=5000)
    elif args.expt == 2:
        run_experiment2(out_dir="out/expt2", max_iter=5000)
    elif args.expt == 3:
        run_experiment3(out_dir="out/expt3", max_iter=5000)
    else:
        raise ValueError(f"Invalid experiment number: {args.expt}")


if __name__ == '__main__':
    main()
