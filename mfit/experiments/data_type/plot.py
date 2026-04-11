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
OPTIMIZER_RESULTS_DIR = "../optimizers/results"
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
# Detect invalid k
# -----------------------------
def get_invalid_k(dataset):
    path = os.path.join(RESULTS_DIR, dataset, "stress.csv")
    if not os.path.exists(path):
        return set()

    df = pd.read_csv(path)
    df["best_log_likelihood"] = pd.to_numeric(df["best_log_likelihood"], errors="coerce")

    invalid = set()

    for k, group in df.groupby("k"):
        ll_invalid = ~np.isfinite(df[df["k"] == k]["best_log_likelihood"].fillna(float("inf")))
        if ll_invalid.any():
            invalid.add(k)
            continue

        param_cols = ["P00", "P01", "P10", "P11"]
        all_nan = group[param_cols].apply(
            lambda row: not np.all(np.isfinite(row)), axis=1
        ).all()
        if all_nan:
            invalid.add(k)

    return invalid

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
# Override baseline with adam optimizer results (15 samples per k)
# -----------------------------
for dataset in list(all_data.keys()):
    adam_path = os.path.join(OPTIMIZER_RESULTS_DIR, dataset, "adam.csv")
    if not os.path.exists(adam_path):
        continue

    df = pd.read_csv(adam_path)
    df["best_log_likelihood"] = pd.to_numeric(df["best_log_likelihood"], errors="coerce")

    all_ks = sorted(df["k"].unique())
    k_dict = {}

    for k in all_ks:
        group = df[df["k"] == k]
        l2_list = []

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
            k_dict[k] = l2_list[:15]

    all_data[dataset]["baseline"] = k_dict
    print(f"Overrode baseline for {dataset} with adam optimizer data (15 samples per k)")

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

    if t1 == "baseline":
        ks = sorted(data_dict[t1].keys())
    else:
        df_full = pd.read_csv(os.path.join(RESULTS_DIR, dataset, f"{t1}.csv"))
        ks = sorted(df_full["k"].unique())

    if t2 in data_dict:
        ks = sorted(set(ks) | set(data_dict[t2].keys()))

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

        for pos, vals in zip(positions, groups):
            jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), pos)+jitter, vals,
                       color="black", alpha=0.3, s=10)

        medians = [np.median(v) for v in groups]
        ax.plot(positions, medians, linestyle="--",
                linewidth=2, marker="D", markersize=5, color=color)

    for k in ks:
        n1 = len(data_dict[t1][k]) if k in data_dict[t1] else 0
        n2 = len(data_dict[t2][k]) if k in data_dict[t2] else 0

        if n1 > 0:
            ax.annotate(f"n={n1}", xy=(k, y_lim[0]), xycoords="data",
                        ha="center", va="top", fontsize=7, color=TYPE_COLORS[t1],
                        xytext=(0, -18), textcoords="offset points")
        if n2 > 0:
            ax.annotate(f"n={n2}", xy=(k, y_lim[0]), xycoords="data",
                        ha="center", va="top", fontsize=7, color=TYPE_COLORS[t2],
                        xytext=(0, -32), textcoords="offset points")

    if t2 == "stress":
        bad_x = [k for k in ks if k in invalid_k]
        bad_y = [y_lim[1]*0.9 for _ in bad_x]
        ax.scatter(bad_x, bad_y, marker='x', color='red',
                   s=100, linewidths=2, zorder=10)

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
        Line2D([0],[0], color="red", marker='x', linestyle="", markersize=8,
               linewidth=2, label="stress failed"),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    fig.subplots_adjust(bottom=0.18)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_{t1}_vs_{t2}_violin.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved {dataset}_{t1}_vs_{t2}_violin.png")
    plt.close(fig)

# -----------------------------
# Performance plot
# -----------------------------
def performance_plot(dataset):
    fig, ax = plt.subplots(figsize=(10, 5))
    invalid_k = get_invalid_k(dataset)

    ks_all = list(range(1, 21))

    for t in data_types:
        path = os.path.join(RESULTS_DIR, dataset, f"{t}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        ks = sorted(df["k"].unique())
        times = [df[df["k"]==k]["total_time_seconds"].mean() for k in ks]

        if t == "stress":
            clean_ks    = [k for k in ks if k not in invalid_k]
            clean_times = [df[df["k"]==k]["total_time_seconds"].mean() for k in clean_ks]
            ax.plot(clean_ks, clean_times, marker='o', label=t,
                    color=TYPE_COLORS[t], zorder=5)
            bad_times = [df[df["k"]==k]["total_time_seconds"].mean() for k in ks if k in invalid_k]
            ax.scatter([k for k in ks if k in invalid_k], bad_times,
                       marker='x', color='red', s=100, linewidths=2, zorder=10)
        else:
            ax.plot(ks, times, marker='o', label=t,
                    color=TYPE_COLORS[t], zorder=5)

    ax.set_title(f"{dataset} - Time vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(ks_all)
    ax.set_xticklabels(ks_all)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_elements = [
        Line2D([0],[0], color=TYPE_COLORS["baseline"], marker='o', linestyle='-', label='baseline'),
        Line2D([0],[0], color=TYPE_COLORS["economy"],  marker='o', linestyle='-', label='economy'),
        Line2D([0],[0], color=TYPE_COLORS["stress"],   marker='o', linestyle='-', label='stress'),
        Line2D([0],[0], color='red', marker='x', linestyle='', markersize=8, linewidth=2, label='stress failed'),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_time.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved {dataset}_time.png")
    plt.close(fig)

# -----------------------------
# Avg L2 plot
# -----------------------------
def avg_l2_plot(dataset, data_dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    invalid_k = get_invalid_k(dataset)

    ks_all = list(range(1, 21))

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

        if t == "stress":
            clean_ks  = [k for k, v in zip(ks, avg) if k not in invalid_k and not np.isnan(v)]
            clean_avg = [v for k, v in zip(ks, avg) if k not in invalid_k and not np.isnan(v)]
            ax.plot(clean_ks, clean_avg, marker='o', label=t,
                    color=TYPE_COLORS[t], zorder=5)
            bad_ks  = [k for k, v in zip(ks, avg) if k in invalid_k]
            bad_avg = [v if not np.isnan(v) else 0 for k, v in zip(ks, avg) if k in invalid_k]
            ax.scatter(bad_ks, bad_avg, marker='x', color='red',
                       s=100, linewidths=2, zorder=10)
        else:
            ax.plot(ks, avg, marker='o', label=t,
                    color=TYPE_COLORS[t], zorder=5)

    ax.set_title(f"{dataset} - Avg L2 vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("L2")
    ax.set_xticks(ks_all)
    ax.set_xticklabels(ks_all)
    ax.set_ylim(*y_lim)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_elements = [
        Line2D([0],[0], color=TYPE_COLORS["baseline"], marker='o', linestyle='-', label='baseline'),
        Line2D([0],[0], color=TYPE_COLORS["economy"],  marker='o', linestyle='-', label='economy'),
        Line2D([0],[0], color=TYPE_COLORS["stress"],   marker='o', linestyle='-', label='stress'),
        Line2D([0],[0], color='red', marker='x', linestyle='', markersize=8, linewidth=2, label='stress failed'),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_avg_l2.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved {dataset}_avg_l2.png")
    plt.close(fig)

# -----------------------------
# Avg VRAM plot
# -----------------------------
def avg_vram_plot(dataset, vram_dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    invalid_k = get_invalid_k(dataset)

    ks_all = list(range(1, 21))

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

        if t == "stress":
            clean_ks  = [k for k, v in zip(ks, avg) if k not in invalid_k and not np.isnan(v)]
            clean_avg = [v for k, v in zip(ks, avg) if k not in invalid_k and not np.isnan(v)]
            ax.plot(clean_ks, clean_avg, marker='o', label=t,
                    color=TYPE_COLORS[t], zorder=5)
            bad_ks  = [k for k, v in zip(ks, avg) if k in invalid_k]
            bad_avg = [v if not np.isnan(v) else 0 for k, v in zip(ks, avg) if k in invalid_k]
            ax.scatter(bad_ks, bad_avg, marker='x', color='red',
                       s=100, linewidths=2, zorder=10)
        else:
            ax.plot(ks, avg, marker='o', label=t,
                    color=TYPE_COLORS[t], zorder=5)

    ax.set_title(f"{dataset} - Avg Peak VRAM vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("VRAM (MB)")
    ax.set_xticks(ks_all)
    ax.set_xticklabels(ks_all)
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_elements = [
        Line2D([0],[0], color=TYPE_COLORS["baseline"], marker='o', linestyle='-', label='baseline'),
        Line2D([0],[0], color=TYPE_COLORS["economy"],  marker='o', linestyle='-', label='economy'),
        Line2D([0],[0], color=TYPE_COLORS["stress"],   marker='o', linestyle='-', label='stress'),
        Line2D([0],[0], color='red', marker='x', linestyle='', markersize=8, linewidth=2, label='stress failed'),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_vram.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved {dataset}_vram.png")
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