import os
import argparse

DATA_ROOT = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/data"
EXPERIMENT_ROOT = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/experiments/kronfit"
PROJECT_ROOT = os.path.expanduser("~/kronfit")
SBATCH_DIR = os.path.join(PROJECT_ROOT, "sbatches")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Run times (seconds) for kronfit per k values
RUN_TIMES = {1:41.67, 2:44.36, 3:45, 4:45, 5:45, 6:45, 6:46, 7:47, 8:48, 9:49, 10:50.34, 11:54.32, 12:62.08, 12:73.1, 13:94.89, 15:110.68, 16:133.99, 17:172.53, 18:221.04, 19:301.07, 20:463.76, 21:792.02, 22:1401.45, 23:2767.17}

def create_sbatch(experiments_dir, data_dir, fname):
    pass

def specific(k_start, k_end, samples, dataset):
    data_dir = os.path.join(DATA_ROOT, dataset)
    if not os.path.isdir(data_dir):
        return
    experiments_dir = os.path.join(EXPERIMENT_ROOT, dataset)
    os.makedirs(experiments_dir, exist_ok=True)

    existing_data = set(os.listdir(data_dir))
    existing_experiments = set(os.listdir(experiments_dir))

    for k in range(k_start, k_end+1):
        for s in range(1, samples+1):
            fname = f"k{k}_s{s}.txt"
            if fname in existing_data:
                if fname not in existing_experiments:
                    create_sbatch(k, s, experiments_dir, data_dir, fname)

def parent(k_start, k_end, samples):
    for dataset in sorted(os.listdir(DATA_ROOT)):
        specific(k_start, k_end, samples, dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k_start", type=int, help="Start value of K")
    parser.add_argument("k_end", type=int, help="End value of K")
    parser.add_argument("samples", type=int, help="Number of samples per k value")
    parser.add_argument("--dataset", type=str, help="Specific dataset to run Kronfit on")
    args = parser.parse_args()

    if args.dataset:
        specific(args.k_start, args.k_end, args.samples, args.dataset)
    else:
        parent(args.k_start, args.k_end, args.samples)
        # do this


if __name__ == "__main__":
    main()
