#kronfit
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths
RESULTS_DIR = "./results"
PARAMS_FILE = "params.txt"
OUTPUT_DIR = "./plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# -----------------------------
# Step 1: Read true parameters
# -----------------------------
true_params = {}

with open(PARAMS_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"Skipping malformed line: {line}")
            continue
        dataset = parts[0]
        values = np.array(list(map(float, parts[1:])))
        true_params[dataset] = values

# -----------------------------
# Step 2: Parse CSV + compute L2
# -----------------------------
all_l2_values = {}

global_min = float("inf")
global_max = float("-inf")

for file in os.listdir(RESULTS_DIR):
    if not file.endswith(".csv"):
        continue

    dataset = file.replace(".csv", "")
    if dataset not in true_params:
        continue

    df = pd.read_csv(os.path.join(RESULTS_DIR, file))
    grouped = df.groupby("k")
    dataset_l2 = {}

    for k, group in grouped:
        l2_list = []
        for _, row in group.iterrows():
            pred = np.array([row["theta_0"], row["theta_1"],
                             row["theta_2"], row["theta_3"]])
            l2 = np.linalg.norm(pred - true_params[dataset])
            l2_list.append(l2)
            global_min = min(global_min, l2)
            global_max = max(global_max, l2)
        dataset_l2[k] = l2_list

    all_l2_values[dataset] = dataset_l2

y_pad = (global_max - global_min) * 0.05
y_lim = (max(0, global_min - y_pad), global_max + y_pad)

# -----------------------------
# Step 3: Plotting
# -----------------------------
for dataset, k_dict in all_l2_values.items():

    ks = sorted(k_dict.keys())
    data_groups = [k_dict[k] for k in ks]
    counts = [len(k_dict[k]) for k in ks]
    avg_l2 = [np.mean(k_dict[k]) for k in ks]

    # ---- Avg L2 line plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ks, avg_l2, marker='o', linewidth=2, color=COLORS[0], label="Avg L2")
    ax.set_ylabel("Average L2 Norm", fontsize=12)
    ax.set_title(f"{dataset} - Avg L2 vs k", fontsize=14, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(*y_lim)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    for k, count in zip(ks, counts):
        ax.annotate(f"n={count}", xy=(k, y_lim[0]), xycoords="data",
                    ha="center", va="top", fontsize=7, color="gray",
                    xytext=(0, -18), textcoords="offset points")

    fig.subplots_adjust(bottom=0.15)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_l2_norm_vs_k.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Violin Plot ----
    fig, ax = plt.subplots(figsize=(max(8, len(ks) * 0.9 + 2), 5))

    valid_idx  = [j for j, g in enumerate(data_groups) if len(g) > 1]
    single_idx = [j for j, g in enumerate(data_groups) if len(g) == 1]

    if valid_idx:
        vp = ax.violinplot(
            [data_groups[j] for j in valid_idx],
            positions=[ks[j] for j in valid_idx],
            widths=0.6, showmedians=True, showextrema=True,
        )
        for pc in vp["bodies"]:
            pc.set_facecolor(COLORS[0])
            pc.set_alpha(0.6)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

    if single_idx:
        ax.scatter(
            [ks[j] for j in single_idx],
            [data_groups[j][0] for j in single_idx],
            color=COLORS[0], zorder=5, s=60, label="Single sample"
        )

    for j, k in enumerate(ks):
        jitter = np.random.uniform(-0.15, 0.15, size=len(data_groups[j]))
        ax.scatter(
            np.full(len(data_groups[j]), k) + jitter,
            data_groups[j],
            color="black", alpha=0.35, s=15, zorder=6
        )

    medians = [np.median(k_dict[k]) for k in ks]
    ax.plot(ks, medians, color="crimson", linewidth=2, linestyle="--",
            marker="D", markersize=6, zorder=7, label="Median L2")

    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(f"{dataset} - L2 Distribution vs k", fontsize=13, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(*y_lim)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for k, count in zip(ks, counts):
        ax.annotate(f"n={count}", xy=(k, y_lim[0]), xycoords="data",
                    ha="center", va="top", fontsize=7, color="gray",
                    xytext=(0, -18), textcoords="offset points")

    fig.subplots_adjust(bottom=0.15)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_L2_violin_plot_vs_k.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

print("All plots saved in:", OUTPUT_DIR)