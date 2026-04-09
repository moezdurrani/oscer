import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
RESULTS_DIR = "./results"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Colors
# -----------------------------
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

TYPE_COLORS = {
    "baseline": COLORS[0],
    "economy": COLORS[1],
    "stress": COLORS[2],
}

data_types = ["baseline", "economy", "stress"]

# -----------------------------
# True parameters
# -----------------------------
true_params = {
    "CA-GR-QC": np.array([0.999, 0.245, 0.245, 0.691]),
    "Blog-Nat06all": np.array([0.999, 0.578, 0.517, 0.221]),
}

# -----------------------------
# Detect invalid k (stress LL = inf)
# -----------------------------
def get_invalid_k(dataset):
    path = os.path.join(RESULTS_DIR, dataset, "stress.csv")
    if not os.path.exists(path):
        return set()

    df = pd.read_csv(path)
    df["best_log_likelihood"] = pd.to_numeric(df["best_log_likelihood"], errors="coerce")

    invalid = df[~np.isfinite(df["best_log_likelihood"])]
    return set(invalid["k"].unique())

# -----------------------------
# Load data
# -----------------------------
all_data = {}
all_vram = {}
global_min = float("inf")
global_max = float("-inf")

for dataset in os.listdir(RESULTS_DIR):
    dataset_path = os.path.join(RESULTS_DIR, dataset)
    if not os.path.isdir(dataset_path):
        continue

    if dataset not in true_params:
        continue

    dataset_dict = {}
    dataset_vram = {}

    for dtype in data_types:
        path = os.path.join(dataset_path, f"{dtype}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["best_log_likelihood"] = pd.to_numeric(df["best_log_likelihood"], errors="coerce")

        grouped = df.groupby("k")

        k_dict = {}
        vram_dict = {}

        for k, group in grouped:
            l2_list = []
            vram_list = group["peak_vram_mb"].dropna().tolist()

            for _, row in group.iterrows():

                if not np.isfinite(row["best_log_likelihood"]):
                    continue

                pred = np.array([row["P00"], row["P01"], row["P10"], row["P11"]])

                if not np.all(np.isfinite(pred)):
                    continue

                l2 = np.linalg.norm(pred - true_params[dataset])
                l2_list.append(l2)

                global_min = min(global_min, l2)
                global_max = max(global_max, l2)

            if l2_list:
                k_dict[k] = l2_list

            if vram_list:
                vram_dict[k] = vram_list

        dataset_dict[dtype] = k_dict
        dataset_vram[dtype] = vram_dict

    all_data[dataset] = dataset_dict
    all_vram[dataset] = dataset_vram

# -----------------------------
# Y scaling
# -----------------------------
y_pad = (global_max - global_min) * 0.05 if global_max > global_min else 0.1
y_lim = (max(0, global_min - y_pad), global_max + y_pad)

# -----------------------------
# Violin plot
# -----------------------------
def violin_plot(dataset, t1, t2, data_dict):
    invalid_k = get_invalid_k(dataset)

    # FULL k range from CSV
    df_full = pd.read_csv(os.path.join(RESULTS_DIR, dataset, f"{t1}.csv"))
    ks = sorted(df_full["k"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(ks)*0.9+2), 5))
    offset = 0.2

    for i, t in enumerate([t1, t2]):
        color = TYPE_COLORS[t]

        positions = []
        groups = []

        for k in ks:
            if k not in data_dict[t]:
                continue

            pos = k + (-offset if i == 0 else offset)
            positions.append(pos)
            groups.append(data_dict[t][k])

        if not groups:
            continue

        vp = ax.violinplot(groups, positions=positions, widths=0.35,
                           showmedians=False, showextrema=True)

        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # scatter
        for pos, vals in zip(positions, groups):
            jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), pos)+jitter, vals,
                       color="black", alpha=0.3, s=10)

        # median
        medians = [np.median(v) for v in groups]
        ax.plot(positions, medians, linestyle="--",
                linewidth=2, marker="D", color=color)

    # sample counts
    for k in ks:
        if k in data_dict[t1]:
            ax.annotate(f"{len(data_dict[t1][k])}", (k, y_lim[0]),
                        xytext=(0,-18), textcoords="offset points",
                        ha="center", fontsize=7, color=TYPE_COLORS[t1])

        if k in data_dict[t2]:
            ax.annotate(f"{len(data_dict[t2][k])}", (k, y_lim[0]),
                        xytext=(0,-32), textcoords="offset points",
                        ha="center", fontsize=7, color=TYPE_COLORS[t2])

    # ❌ stress failure (on top)
    if t2 == "stress":
        bad_x = [k for k in ks if k in invalid_k]
        bad_y = [y_lim[1]*0.9 for _ in bad_x]

        ax.scatter(bad_x, bad_y,
                   marker='x',
                   color='red',
                   s=100,
                   linewidths=2,
                   zorder=10)

    ax.set_title(f"{dataset} - {t1} vs {t2} (L2)", fontweight="bold")
    ax.set_ylabel("L2 Norm")
    ax.set_xticks(ks)
    ax.set_xticklabels([int(k) for k in ks])
    ax.set_ylim(*y_lim)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_elements = [
        Patch(facecolor=TYPE_COLORS[t1], alpha=0.6, label=t1),
        Patch(facecolor=TYPE_COLORS[t2], alpha=0.6, label=t2),
        Line2D([0],[0], color=TYPE_COLORS[t1], linestyle="--", marker="D", label=f"{t1} median"),
        Line2D([0],[0], color=TYPE_COLORS[t2], linestyle="--", marker="D", label=f"{t2} median"),
        Line2D([0],[0], color="red", marker='x', linestyle="", label="stress failed")
    ]

    ax.legend(handles=legend_elements, frameon=False)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_{t1}_vs_{t2}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Performance plot
# -----------------------------
def performance_plot(dataset):
    fig, ax = plt.subplots()
    invalid_k = get_invalid_k(dataset)

    for t in data_types:
        path = os.path.join(RESULTS_DIR, dataset, f"{t}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        ks = sorted(df["k"].unique())
        times = [df[df["k"]==k]["total_time_seconds"].mean() for k in ks]

        ax.plot(ks, times, marker='o', label=t, color=TYPE_COLORS[t])

        if t == "stress":
            bad_x = [k for k in ks if k in invalid_k]
            bad_y = [df[df["k"]==k]["total_time_seconds"].mean() for k in bad_x]

            ax.scatter(bad_x, bad_y,
                       marker='x',
                       color='red',
                       s=100,
                       linewidths=2,
                       zorder=10)

    ax.set_title(f"{dataset} - Time vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(ks)
    ax.set_xticklabels([int(k) for k in ks])
    ax.legend(frameon=False)
    ax.grid(True)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_time.png"))
    plt.close(fig)

# -----------------------------
# Avg L2 plot
# -----------------------------
def avg_l2_plot(dataset, data_dict):
    fig, ax = plt.subplots()
    invalid_k = get_invalid_k(dataset)

    for t in data_types:
        if t not in data_dict:
            continue

        df = pd.read_csv(os.path.join(RESULTS_DIR, dataset, f"{t}.csv"))
        ks = sorted(df["k"].unique())

        avg = []
        for k in ks:
            if k in data_dict[t]:
                avg.append(np.mean(data_dict[t][k]))
            else:
                avg.append(np.nan)

        ax.plot(ks, avg, marker='o', label=t, color=TYPE_COLORS[t])

        if t == "stress":
            bad_x = [k for k in ks if k in invalid_k]
            bad_y = [avg[ks.index(k)] for k in bad_x if k in ks]

            ax.scatter(bad_x, bad_y,
                       marker='x',
                       color='red',
                       s=100,
                       linewidths=2,
                       zorder=10)

    ax.set_title(f"{dataset} - Avg L2 vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("L2")
    ax.set_xticks(ks)
    ax.set_xticklabels([int(k) for k in ks])
    ax.set_ylim(*y_lim)
    ax.legend(frameon=False)
    ax.grid(True)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_avg_l2.png"))
    plt.close(fig)

# -----------------------------
# Avg VRAM plot
# -----------------------------
def avg_vram_plot(dataset, vram_dict):
    fig, ax = plt.subplots()
    invalid_k = get_invalid_k(dataset)

    for t in data_types:
        if t not in vram_dict:
            continue

        df = pd.read_csv(os.path.join(RESULTS_DIR, dataset, f"{t}.csv"))
        ks = sorted(df["k"].unique())

        avg = []
        for k in ks:
            if k in vram_dict[t]:
                avg.append(np.mean(vram_dict[t][k]))
            else:
                avg.append(np.nan)

        ax.plot(ks, avg, marker='o', label=t, color=TYPE_COLORS[t])

        if t == "stress":
            bad_x = [k for k in ks if k in invalid_k]
            bad_y = [avg[ks.index(k)] for k in bad_x if k in ks]

            ax.scatter(bad_x, bad_y,
                       marker='x',
                       color='red',
                       s=100,
                       linewidths=2,
                       zorder=10)

    ax.set_title(f"{dataset} - Avg Peak VRAM vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("VRAM (MB)")
    ax.set_xticks(ks)
    ax.set_xticklabels([int(k) for k in ks])
    ax.legend(frameon=False)
    ax.grid(True)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_vram.png"))
    plt.close(fig)

# -----------------------------
# Run everything
# -----------------------------
for dataset in all_data:
    violin_plot(dataset, "baseline", "economy", all_data[dataset])
    violin_plot(dataset, "baseline", "stress", all_data[dataset])
    performance_plot(dataset)
    avg_l2_plot(dataset, all_data[dataset])
    avg_vram_plot(dataset, all_vram[dataset])

print("All plots saved.")