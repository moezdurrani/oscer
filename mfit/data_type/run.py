import os
import json
import time
import sys
import argparse

sys.path.append(os.path.abspath(".."))
from mfit import mfit

# -------- PATHS -------- #
DATA_DIR = "../../data"
EXP_DIR = "./experiments"
# ----------------------- #

DATASETS = ["Blog-Nat06all", "CA-GR-QC"]

INIT_MATRIX = [0.9, 0.7, 0.5, 0.2]
ITERATIONS = 100
LR = 0.05
WARMUP = 10000
GRAD_SAMPLES = 100000

OPTIMIZERS = ["adam", "adamw", "rmsprop"]


def run_single(file_path, optimizer, output_path, mode):
    print(f"\nRunning {mode} experiment on {file_path}")

    start_time = time.time()

    model = mfit(
        graph_file_path=file_path,
        init_matrix=INIT_MATRIX,
        iterations=ITERATIONS,
        warmup_mcmc=WARMUP,
        mcmc_per_iter=GRAD_SAMPLES,
        learning_rate=LR,
        optimizer_name=optimizer,
        mode=mode,
        gpu_synchronize=False,
    )

    best_P, best_ll, profiling_data = model.fit()

    total_time = time.time() - start_time

    output_data = {
        "best_parameters": {
            "P00": best_P[0],
            "P01": best_P[1],
            "P10": best_P[2],
            "P11": best_P[3]
        },
        "best_log_likelihood": best_ll,
        "peak_vram_mb": profiling_data.get("peak_vram_mb", 0.0),
        "total_time_seconds": total_time,
        "profiling_timings_seconds": profiling_data
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k_start",
        type=int,
        required=True,
        help="Starting value of k"
    )
    parser.add_argument(
        "--k_end",
        type=int,
        required=True,
        help="Ending value of k (inclusive)"
    )
    parser.add_argument(
        "--s_start",
        type=int,
        required=True,
        help="Starting value of s"
    )
    parser.add_argument(
        "--s_end",
        type=int,
        required=True,
        help="Ending value of s (inclusive)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="adam",
        choices=["adam", "adamw", "rmsprop"],
        help="Optimizer to use"
    )
    parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    choices=DATASETS,
    help="Dataset to use"
    )
    parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["baseline", "economy", "stress"],
    help="Experiment datatype mode"
    )
    args = parser.parse_args()

    optimizer = args.optimizer
    k_start = args.k_start
    k_end = args.k_end
    s_start = args.s_start
    s_end = args.s_end
    d = args.dataset
    mode = args.mode

    for dataset in DATASETS:
        if d and dataset != d:
            continue
        print(f"\n========== DATASET: {dataset} ==========")

        dataset_data_path = os.path.join(DATA_DIR, dataset)
        dataset_exp_path = os.path.join(EXP_DIR, dataset)

        os.makedirs(dataset_exp_path, exist_ok=True)

        opt_path = os.path.join(dataset_exp_path, mode)
        os.makedirs(opt_path, exist_ok=True)

        # loop over k and s
        for k in range(k_start, k_end+1):
            for s in range(s_start, s_end+1):

                filename = f"k{k}_s{s}.txt"
                file_path = os.path.join(dataset_data_path, filename)

                if not os.path.exists(file_path):
                    print(f"Missing: {file_path}")
                    continue

                json_name = f"k{k}_s{s}.json"
                json_path = os.path.join(opt_path, json_name)

                # skip if already exists
                if os.path.exists(json_path):
                    print(f"Skipping (exists): {json_path}")
                    continue

                try:
                    run_single(file_path, optimizer, json_path, mode)
                except Exception as e:
                    print(f"Error on {filename} ({mode}): {e}")


if __name__ == "__main__":
    main()
