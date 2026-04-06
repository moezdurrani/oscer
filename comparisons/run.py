import os
import json
import time
import networkx as nx

from mfit import mfit   # your file

# ---------------- CONFIG ---------------- #
DATA_DIR = "data"
EXP_DIR = "experiments"

INIT_MATRIX = [0.9, 0.7, 0.5, 0.2]
ITERATIONS = 100
LR = 0.05
WARMUP = 10000
GRAD_SAMPLES = 100000
# ---------------------------------------- #

def run_mfit_on_file(file_path):
    print(f"\nRunning MFIT on: {file_path}")

    start_time = time.time()

    graph = nx.read_edgelist(
        file_path,
        nodetype=int,
        create_using=nx.DiGraph(),
        comments="#"
    )

    fitter = mfit(graph, INIT_MATRIX, learning_rate=LR)

    best_P, best_ll = fitter.fit(
        iterations=ITERATIONS,
        warmup_mcmc=WARMUP,
        mcmc_per_iter=GRAD_SAMPLES
    )

    total_time = time.time() - start_time

    return {
        "time": total_time,
        "best_ll": best_ll,
        "best_P": best_P
    }


def main():
    # loop through datasets
    for dataset in os.listdir(DATA_DIR):
        dataset_path = os.path.join(DATA_DIR, dataset)

        if not os.path.isdir(dataset_path):
            continue

        print(f"\n========== DATASET: {dataset} ==========")

        # create experiments folder
        exp_dataset_path = os.path.join(EXP_DIR, dataset)
        os.makedirs(exp_dataset_path, exist_ok=True)

        def extract_k(filename):
            return int(filename.split('_')[0][1:])  # extracts number after 'k'

        # loop through files
        for file in sorted(os.listdir(dataset_path), key=extract_k):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(dataset_path, file)

            # output json file
            json_name = file.replace(".txt", ".json")
            json_path = os.path.join(exp_dataset_path, json_name)

            if os.path.exists(json_path):
                print(f"Skipping (already done): {file}")
                continue

            try:
                result = run_mfit_on_file(file_path)

                with open(json_path, "w") as f:
                    json.dump(result, f, indent=4)

                print(f"Saved → {json_path}")

            except Exception as e:
                print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    main()