import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = "./kronfit"
PARAMS_FILE = "params.txt"
OUTPUT_DIR = "./kronfit_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Step 1: Read true parameters
# -----------------------------
true_params = {}

with open(PARAMS_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        dataset = parts[0]
        values = np.array(list(map(float, parts[1:])))
        true_params[dataset] = values

# -----------------------------
# Step 2: Parse CSV + compute L2
# -----------------------------
all_l2_values = {}  # dataset -> {k: [l2 values]}

global_min = float("inf")
global_max = float("-inf")

for file in os.listdir(RESULTS_DIR):
    if not file.endswith(".csv"):
        continue

    dataset = file.replace(".csv", "")
    if dataset not in true_params:
        continue

    df = pd.read_csv(os.path.join(RESULTS_DIR, file))

    # Group by k
    grouped = df.groupby("k")

    dataset_l2 = {}

    for k, group in grouped:
        l2_list = []

        for _, row in group.iterrows():
            pred = np.array([
                row["theta_0"],
                row["theta_1"],
                row["theta_2"],
                row["theta_3"]
            ])

            l2 = np.linalg.norm(pred - true_params[dataset])
            l2_list.append(l2)

            global_min = min(global_min, l2)
            global_max = max(global_max, l2)

        dataset_l2[k] = l2_list

    all_l2_values[dataset] = dataset_l2

# -----------------------------
# Step 3: Plotting
# -----------------------------
for dataset, k_dict in all_l2_values.items():

    ks = sorted(k_dict.keys())

    # ---- Avg L2 ----
    avg_l2 = [np.mean(k_dict[k]) for k in ks]

    plt.figure()
    plt.plot(ks, avg_l2, marker='o')
    plt.xlabel("k")
    plt.ylabel("Average L2 Norm")
    plt.title(f"{dataset} - Avg L2 vs k")
    plt.ylim(global_min, global_max)

    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_l2_norm_vs_k.png"))
    plt.close()

    # ---- Violin Plot ----
    data = [k_dict[k] for k in ks]

    plt.figure()
    plt.violinplot(data, showmeans=True)

    plt.xticks(range(1, len(ks)+1), ks)
    plt.xlabel("k")
    plt.ylabel("L2 Norm")
    plt.title(f"{dataset} - L2 Distribution vs k")
    plt.ylim(global_min, global_max)

    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_violin_plot_vs_k.png"))
    plt.close()

print("All plots saved in:", OUTPUT_DIR)
