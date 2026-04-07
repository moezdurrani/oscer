import os
import json
import time
import sys

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


def run_single(file_path, optimizer, output_path):
    print(f"\nRunning {optimizer} on {file_path}")

    start_time = time.time()

    model = mfit(
        graph_file_path=file_path,
        init_matrix=INIT_MATRIX,
        iterations=ITERATIONS,
        warmup_mcmc=WARMUP,
        mcmc_per_iter=GRAD_SAMPLES,
        learning_rate=LR,
        optimizer_name=optimizer,
        gpu_synchronize=False
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
        "total_time_seconds": total_time,
        "profiling_timings_seconds": profiling_data
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved → {output_path}")


def main():
    for dataset in DATASETS:
        print(f"\n========== DATASET: {dataset} ==========")

        dataset_data_path = os.path.join(DATA_DIR, dataset)
        dataset_exp_path = os.path.join(EXP_DIR, dataset)

        os.makedirs(dataset_exp_path, exist_ok=True)

        # create optimizer subfolders
        optimizer_paths = {}
        for opt in OPTIMIZERS:
            opt_path = os.path.join(dataset_exp_path, opt)
            os.makedirs(opt_path, exist_ok=True)
            optimizer_paths[opt] = opt_path

        # loop over k and s
        for k in range(1, 23):
            for s in range(1, 31):

                filename = f"k{k}_s{s}.txt"
                file_path = os.path.join(dataset_data_path, filename)

                if not os.path.exists(file_path):
                    continue

                for optimizer in OPTIMIZERS:
                    json_name = f"k{k}_s{s}.json"
                    json_path = os.path.join(optimizer_paths[optimizer], json_name)

                    # skip if already exists
                    if os.path.exists(json_path):
                        print(f"Skipping (exists): {json_path}")
                        continue

                    try:
                        run_single(file_path, optimizer, json_path)
                    except Exception as e:
                        print(f"Error on {filename} ({optimizer}): {e}")


if __name__ == "__main__":
    main()