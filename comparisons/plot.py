import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG -------- #
RESULTS_DIR = "results"
MFIT_DIR = os.path.join(RESULTS_DIR, "mfit")
KRONFIT_DIR = os.path.join(RESULTS_DIR, "kronfit")

GROUND_TRUTH = {
    "ATP-GR-QC": [0.902, 0.253, 0.221, 0.582],
    "Blog-Nat06all": [0.999, 0.578, 0.517, 0.221]
}
# ------------------------ #


def compute_l2(pred, true):
    return np.linalg.norm(np.array(pred) - np.array(true))


def process_dataset(dataset):
    print(f"\nProcessing: {dataset}")

    mfit_path = os.path.join(MFIT_DIR, f"{dataset}.csv")
    kronfit_path = os.path.join(KRONFIT_DIR, f"{dataset}.csv")

    if not os.path.exists(mfit_path) or not os.path.exists(kronfit_path):
        print(f"Missing files for {dataset}")
        return

    # -------- Load data -------- #
    mfit_df = pd.read_csv(mfit_path)
    kronfit_df = pd.read_csv(kronfit_path)

    # -------- Rename kronfit columns -------- #
    kronfit_df = kronfit_df.rename(columns={
        "theta_0": "a",
        "theta_1": "b",
        "theta_2": "c",
        "theta_3": "d"
    })

    # -------- Average kronfit per k -------- #
    kronfit_avg = kronfit_df.groupby("k").agg({
        "time": "mean",
        "a": "mean",
        "b": "mean",
        "c": "mean",
        "d": "mean"
    }).reset_index()

    # -------- Merge with mfit -------- #
    merged = pd.merge(mfit_df, kronfit_avg, on="k", suffixes=("_mfit", "_kronfit"))

    true_vals = GROUND_TRUTH[dataset]

    # -------- Compute L2 -------- #
    l2_mfit = []
    l2_kronfit = []

    for _, row in merged.iterrows():
        mfit_params = [row["a_mfit"], row["b_mfit"], row["c_mfit"], row["d_mfit"]]
        kronfit_params = [row["a_kronfit"], row["b_kronfit"], row["c_kronfit"], row["d_kronfit"]]

        l2_mfit.append(compute_l2(mfit_params, true_vals))
        l2_kronfit.append(compute_l2(kronfit_params, true_vals))

    merged["l2_mfit"] = l2_mfit
    merged["l2_kronfit"] = l2_kronfit

    # -------- SORT by k -------- #
    merged = merged.sort_values("k")

    # -------- Plot 1: L2 vs k -------- #
    plt.figure()
    plt.plot(merged["k"], merged["l2_mfit"], label="MFIT")
    plt.plot(merged["k"], merged["l2_kronfit"], label="KronFit")
    plt.xlabel("k")
    plt.ylabel("L2 Norm")
    plt.title(f"L2 Norm vs k ({dataset})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, f"{dataset}_l2.png"))
    plt.close()

    # -------- Plot 2: Time vs k -------- #
    plt.figure()
    plt.plot(merged["k"], merged["time_mfit"], label="MFIT")
    plt.plot(merged["k"], merged["time_kronfit"], label="KronFit")
    plt.xlabel("k")
    plt.ylabel("Time (s)")
    plt.title(f"Time vs k ({dataset})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, f"{dataset}_time.png"))
    plt.close()

    print(f"Saved plots for {dataset}")


def main():
    for dataset in GROUND_TRUTH.keys():
        process_dataset(dataset)


if __name__ == "__main__":
    main()