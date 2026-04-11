import os
import json
import time
import subprocess
import sys
import os

sys.path.append(os.path.abspath("../../"))
from mfit import mfit

# -------- PATHS -------- #
DATA_DIR = "../../../data/snap_datasets"
RESULTS_DIR = "./experiments"
KRONFIT_BIN = "../../../../snap/examples/kronfit/kronfit"
# ----------------------- #

# Create result directories
os.makedirs(os.path.join(RESULTS_DIR, "mfit"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "kronfit"), exist_ok=True)

# Loop over all datasets
for file in os.listdir(DATA_DIR):
    if not file.endswith(".txt"):
        continue

    filename = os.path.splitext(file)[0]
    full_path = os.path.join(DATA_DIR, file)

    print("\n--------------------------------------")
    print(f"Processing: {filename}")
    print("--------------------------------------")

    mfit_out = os.path.join(RESULTS_DIR, "mfit", f"{filename}.json")
    kronfit_out = os.path.join(RESULTS_DIR, "kronfit", f"{filename}.json")

    # -------- MFIT -------- #
    if os.path.exists(mfit_out):
        print("MFIT exists, skipping")
    else:
        print("Running MFIT...")
        start_time = time.time()

        model = mfit(
            graph_file_path=full_path,
            init_matrix=[0.9, 0.7, 0.5, 0.2],
            iterations=100,
            warmup_mcmc=10000,
            mcmc_per_iter=100000,
            learning_rate=0.05,
            optimizer_name="adam",
            mode="baseline",
            n_threads=None,
            gpu_synchronize=False
        )

        best_P, best_ll, profiling = model.fit()
        total_time = time.time() - start_time

        # SAVE CLEAN JSON (MATCHING KRONFIT STYLE)
        result = {
            "theta": best_P,
             "best_ll": best_ll,
            "time": total_time, 
        }

        with open(mfit_out, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Saved MFIT → {mfit_out}")

    # -------- KRONFIT -------- #
    if os.path.exists(kronfit_out):
        print("Kronfit exists, skipping")
    else:
        print("Running Kronfit...")

        subprocess.run([
            KRONFIT_BIN,
            f"-i:{full_path}",
            f"-o:{kronfit_out}",
            "-gi:100",
            "-s:100000"
        ])

        print(f"Saved Kronfit → {kronfit_out}")

print("\nAll datasets processed.")