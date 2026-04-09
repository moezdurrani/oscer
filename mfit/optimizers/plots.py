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

OPT_COLORS = {
    "adam": COLORS[0],      # blue
    "adamw": COLORS[1],     # orange
    "rmsprop": COLORS[2],   # green
}

# -----------------------------
# True parameters
# -----------------------------
true_params = {
    "CA-GR-QC": np.array([0.999, 0.245, 0.245, 0.691]),
    "Blog-Nat06all": np.array([0.999, 0.578, 0.517, 0.221]),
}

optimizers = ["adam", "adamw", "rmsprop"]

# -----------------------------
# Load + compute L2
# -----------------------------
all_data = {}
global_min = float("inf")
global_max = float("-inf")

for dataset in os.listdir(RESULTS_DIR):
    dataset_path = os.path.join(RESULTS_DIR, dataset)
    if not os.path.isdir(dataset_path):
        continue

    if dataset not in true_params:
        continue

    dataset_dict = {}

    for opt in optimizers:
        csv_path = os.path.join(dataset_path, f"{opt}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        grouped = df.groupby("k")

        k_dict = {}

        for k, group in grouped:
            l2_list = []

            for _, row in group.iterrows():
                pred = np.array([
                    row["P00"],
                    row["P01"],
                    row["P10"],
                    row["P11"]
                ])

                l2 = np.linalg.norm(pred - true_params[dataset])
                l2_list.append(l2)

                global_min = min(global_min, l2)
                global_max = max(global_max, l2)

            k_dict[k] = l2_list

        dataset_dict[opt] = k_dict

    all_data[dataset] = dataset_dict

# -----------------------------
# Consistent Y-axis
# -----------------------------
y_pad = (global_max - global_min) * 0.05
y_lim = (max(0, global_min - y_pad), global_max + y_pad)

# -----------------------------
# Violin Plot (WITH MEDIAN LINES)
# -----------------------------
def violin_plot(dataset, opt1, opt2, data_dict):
    ks = sorted(set(data_dict[opt1].keys()) & set(data_dict[opt2].keys()))

    fig, ax = plt.subplots(figsize=(max(8, len(ks) * 0.9 + 2), 5))

    offset = 0.2

    for i, opt in enumerate([opt1, opt2]):
        color = OPT_COLORS[opt]

        positions = []
        groups = []

        for k in ks:
            vals = data_dict[opt][k]
            if len(vals) == 0:
                continue

            pos = k + (-offset if i == 0 else offset)
            positions.append(pos)
            groups.append(vals)

        vp = ax.violinplot(
            groups,
            positions=positions,
            widths=0.35,
            showmedians=False,  # we draw our own median
            showextrema=True
        )

        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # scatter points
        for pos, vals in zip(positions, groups):
            jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                color="black",
                alpha=0.3,
                s=10
            )

        # ---- MEDIAN LINE ----
        medians = [np.median(vals) for vals in groups]

        ax.plot(
            positions,
            medians,
            linestyle="--",
            linewidth=2,
            marker="D",
            markersize=5,
            color=color,
        )

        # annotate counts
        # ---- Sample count annotations (per optimizer) ----
        for k in ks:
            n1 = len(data_dict[opt1][k])
            n2 = len(data_dict[opt2][k])

            # First optimizer (slightly higher)
            ax.annotate(
                f"n={n1}",
                xy=(k, y_lim[0]),
                xycoords="data",
                ha="center",
                va="top",
                fontsize=7,
                color=OPT_COLORS[opt1],
                xytext=(0, -18),
                textcoords="offset points"
            )

            # Second optimizer (slightly lower)
            ax.annotate(
                f"n={n2}",
                xy=(k, y_lim[0]),
                xycoords="data",
                ha="center",
                va="top",
                fontsize=7,
                color=OPT_COLORS[opt2],
                xytext=(0, -32),
                textcoords="offset points"
            )

    ax.set_title(f"{dataset} - {opt1} vs {opt2} (L2)", fontweight="bold")
    ax.set_ylabel("L2 Norm")
    ax.set_xticks(ks)
    ax.set_ylim(*y_lim)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ---- Legend ----
    legend_elements = [
        Patch(facecolor=OPT_COLORS[opt1], alpha=0.6, label=opt1),
        Patch(facecolor=OPT_COLORS[opt2], alpha=0.6, label=opt2),
        Line2D([0], [0], color=OPT_COLORS[opt1], linestyle="--", marker="D", label=f"{opt1} median"),
        Line2D([0], [0], color=OPT_COLORS[opt2], linestyle="--", marker="D", label=f"{opt2} median"),
    ]

    ax.legend(handles=legend_elements, fontsize=10, loc="upper right", frameon=False)

    fig.subplots_adjust(bottom=0.15)
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{dataset}_{opt1}_vs_{opt2}_violin.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------
# Performance Plot
# -----------------------------
def performance_plot(dataset, data_dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for opt in optimizers:
        if opt not in data_dict:
            continue

        df = pd.read_csv(os.path.join(RESULTS_DIR, dataset, f"{opt}.csv"))

        ks = sorted(df["k"].unique())
        times = [df[df["k"] == k]["total_time_seconds"].mean() for k in ks]

        ax.plot(
            ks,
            times,
            marker='o',
            linewidth=2,
            label=opt,
            color=OPT_COLORS[opt]
        )

    ax.set_title(f"{dataset} - Performance (Time vs k)", fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("Total Time (seconds)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_performance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Avg L2 Plot
# -----------------------------
def avg_l2_plot(dataset, data_dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for opt in optimizers:
        if opt not in data_dict:
            continue

        ks = sorted(data_dict[opt].keys())
        avg_l2 = [np.mean(data_dict[opt][k]) for k in ks]

        ax.plot(
            ks,
            avg_l2,
            marker='o',
            linewidth=2,
            label=opt,
            color=OPT_COLORS[opt]
        )

    ax.set_title(f"{dataset} - Avg L2 vs k", fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("Average L2 Norm")
    ax.set_ylim(*y_lim)
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_avg_l2.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Generate plots
# -----------------------------
for dataset, data_dict in all_data.items():
    violin_plot(dataset, "adam", "adamw", data_dict)
    violin_plot(dataset, "adam", "rmsprop", data_dict)
    performance_plot(dataset, data_dict)
    avg_l2_plot(dataset, data_dict)

print("All plots saved in:", OUTPUT_DIR)