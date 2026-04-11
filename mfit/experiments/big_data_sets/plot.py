#big data sets
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
BASE_RESULTS_DIR = "./results"
MFIT_DIR = os.path.join(BASE_RESULTS_DIR, "mfit")
KRONFIT_DIR = os.path.join(BASE_RESULTS_DIR, "kronfit")

PARAMS_FILE = "params.txt"
OUTPUT_DIR = "./plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
MFIT_COLOR = "blue"
KRONFIT_COLOR = "orange"

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
        dataset = parts[0]
        values = np.array(list(map(float, parts[1:])))
        true_params[dataset] = values


def compute_time(df, is_mfit):
    grouped = df.groupby("k")
    result = {}

    for k, group in grouped:
        if is_mfit:
            times = group["total_time"].values
        else:
            times = group["time"].values

        result[k] = list(times)

    return result

# -----------------------------
# Step 2: Load + Compute L2
# -----------------------------
def process_file(filepath, dataset, is_mfit):
    df = pd.read_csv(filepath)

    result = {}

    grouped = df.groupby("k")

    for k, group in grouped:
        l2_list = []

        for _, row in group.iterrows():
            if is_mfit:
                pred = np.array([
                    row["P00"], row["P01"],
                    row["P10"], row["P11"]
                ])
            else:
                pred = np.array([
                    row["theta_0"], row["theta_1"],
                    row["theta_2"], row["theta_3"]
                ])

            l2 = np.linalg.norm(pred - true_params[dataset])
            l2_list.append(l2)

        result[k] = l2_list

    return result

# Collect datasets that exist in BOTH
datasets = set(
    f.replace(".csv", "") for f in os.listdir(MFIT_DIR) if f.endswith(".csv")
).intersection(
    f.replace(".csv", "") for f in os.listdir(KRONFIT_DIR) if f.endswith(".csv")
)

# -----------------------------
# Step 3: Compute global limits
# -----------------------------
global_min = float("inf")
global_max = float("-inf")

all_data = {}

for dataset in datasets:
    if dataset not in true_params:
        continue

    mfit_path = os.path.join(MFIT_DIR, f"{dataset}.csv")
    kronfit_path = os.path.join(KRONFIT_DIR, f"{dataset}.csv")

    mfit_df = pd.read_csv(mfit_path)
    kronfit_df = pd.read_csv(kronfit_path)

    mfit_data = process_file(mfit_path, dataset, is_mfit=True)
    kronfit_data = process_file(kronfit_path, dataset, is_mfit=False)

    mfit_time = compute_time(mfit_df, True)
    kronfit_time = compute_time(kronfit_df, False)

    all_data[dataset] = {
        "mfit": mfit_data,
        "kronfit": kronfit_data,
        "mfit_time": mfit_time,
        "kronfit_time": kronfit_time
    }

    # update global limits
    for method in ["mfit", "kronfit"]:
        for k, vals in all_data[dataset][method].items():
            if len(vals) == 0:
                continue
            global_min = min(global_min, min(vals))
            global_max = max(global_max, max(vals))

y_pad = (global_max - global_min) * 0.05
y_lim = (max(0, global_min - y_pad), global_max + y_pad)

# -----------------------------
# Step 4: Plot
# -----------------------------
for dataset, data in all_data.items():

    mfit_dict = data["mfit"]
    kronfit_dict = data["kronfit"]

    ks = sorted(set(mfit_dict.keys()).intersection(kronfit_dict.keys()))

    # Prepare data
    mfit_groups = [mfit_dict[k] for k in ks]
    kronfit_groups = [kronfit_dict[k] for k in ks]

    mfit_avg = [np.mean(mfit_dict[k]) for k in ks]
    kronfit_avg = [np.mean(kronfit_dict[k]) for k in ks]

    # ---------------- AVG LINE PLOT ----------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ks, mfit_avg, marker='o', linewidth=2,
            color=MFIT_COLOR, label="MFIT")

    ax.plot(ks, kronfit_avg, marker='o', linewidth=2,
            color=KRONFIT_COLOR, label="KronFit")

    ax.set_ylabel("Average L2 Norm")
    ax.set_title(f"{dataset} - Avg L2 vs k", fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(*y_lim)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_avg_l2.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- VIOLIN PLOT ----------------
    fig, ax = plt.subplots(figsize=(max(8, len(ks) * 0.9 + 2), 5))

    width = 0.35

    counts_mfit    = [len(mfit_dict[k])    for k in ks]
    counts_kronfit = [len(kronfit_dict[k]) for k in ks]

    # Fix 1: use prop_cycle colors instead of hardcoded strings
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    MFIT_COLOR = COLORS[0]
    KRONFIT_COLOR = COLORS[1]

    # MFIT violins (left)
    vp1 = ax.violinplot(
        mfit_groups,
        positions=[k - width/2 for k in ks],
        widths=width,
        showmedians=True
    )
    for pc in vp1["bodies"]:
        pc.set_facecolor(MFIT_COLOR)
        pc.set_alpha(0.6)

    # KronFit violins (right)
    vp2 = ax.violinplot(
        kronfit_groups,
        positions=[k + width/2 for k in ks],
        widths=width,
        showmedians=True
    )
    for pc in vp2["bodies"]:
        pc.set_facecolor(KRONFIT_COLOR)
        pc.set_alpha(0.6)

    # Fix 2: jitter dots
    for pos, vals in zip([k - width/2 for k in ks], mfit_groups):
        jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color="black", alpha=0.3, s=10, zorder=5)

    for pos, vals in zip([k + width/2 for k in ks], kronfit_groups):
        jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color="black", alpha=0.3, s=10, zorder=5)

    # Median lines aligned to violin centers
    mfit_xs    = [k - width/2 for k in ks]
    kronfit_xs = [k + width/2 for k in ks]
    mfit_ys    = [np.median(mfit_dict[k])    for k in ks]
    kronfit_ys = [np.median(kronfit_dict[k]) for k in ks]

    ax.plot(mfit_xs, mfit_ys, color=MFIT_COLOR,
            linestyle="--", marker="D", label="MFIT Median")
    ax.plot(kronfit_xs, kronfit_ys, color=KRONFIT_COLOR,
            linestyle="--", marker="D", label="KronFit Median")

    ax.set_ylabel("L2 Norm")
    ax.set_title(f"{dataset} - L2 Distribution vs k", fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(*y_lim)
    ax.grid(True, linestyle="--", alpha=0.4, axis="both")

    # Fix 3: Patch+Line2D legend like optimizer script
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=MFIT_COLOR, alpha=0.6, label="MFIT"),
        Patch(facecolor=KRONFIT_COLOR, alpha=0.6, label="KronFit"),
        Line2D([0], [0], color=MFIT_COLOR, linestyle="--", marker="D", label="MFIT Median"),
        Line2D([0], [0], color=KRONFIT_COLOR, linestyle="--", marker="D", label="KronFit Median"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right", frameon=False)

    for k, cm, ck in zip(ks, counts_mfit, counts_kronfit):
        ax.annotate(f"n={cm}", xy=(k, y_lim[0]), xycoords="data",
                    ha="center", va="top", fontsize=7, color=MFIT_COLOR,
                    xytext=(0, -20), textcoords="offset points")
        ax.annotate(f"n={ck}", xy=(k, y_lim[0]), xycoords="data",
                    ha="center", va="top", fontsize=7, color=KRONFIT_COLOR,
                    xytext=(0, -32), textcoords="offset points")

    fig.subplots_adjust(bottom=0.20)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_violin.png"),
                dpi=150, bbox_inches="tight")
    print(f"Violin Plot for {dataset} saved")
    plt.close(fig)


    # ---------------- PERFORMANCE PLOT ----------------
    mfit_time_dict = data["mfit_time"]
    kron_time_dict = data["kronfit_time"]

    mfit_time_avg = [np.mean(mfit_time_dict[k]) for k in ks]
    kron_time_avg = [np.mean(kron_time_dict[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ks, mfit_time_avg, marker='o', linewidth=2,
            color=MFIT_COLOR, label="MFIT")

    ax.plot(ks, kron_time_avg, marker='o', linewidth=2,
            color=KRONFIT_COLOR, label="KronFit")

    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"{dataset} - Performance vs k", fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{dataset}_performance.png"),
                dpi=150, bbox_inches="tight")
    print(f"Performance Plot for {dataset} saved")
    plt.close(fig)

print("All plots saved in:", OUTPUT_DIR)