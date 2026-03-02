#!/usr/bin/env python3
import os
import csv
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def load_params(params_path):
    """
    Parse params.txt. Accepts two formats:
      - With dataset name:   Answers  0.994  0.384  0.414  0.249
      - Without name:        0.994  0.384  0.414  0.249
    Returns dict {dataset_name: (a, b, c, d)}
    """
    params = {}
    with open(params_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                float(parts[0])
                is_named = False
            except ValueError:
                is_named = True

            if is_named and len(parts) >= 5:
                name = parts[0]
                params[name] = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            elif not is_named and len(parts) >= 4:
                params[len(params)] = (float(parts[0]), float(parts[1]),
                                       float(parts[2]), float(parts[3]))
    return params


def load_csv(csv_path):
    """Return list of dicts from a results CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LABEL  = ["a", "b", "c", "d"]


def plot_time_vs_k(dataset_name, rows, plots_dir):
    """Line plot: average time per k value."""
    time_by_k = defaultdict(list)
    for row in rows:
        k = safe_float(row.get("k"))
        t = safe_float(row.get("time"))
        if k is not None and t is not None:
            time_by_k[int(k)].append(t)

    if not time_by_k:
        print(f"  [{dataset_name}] No time data, skipping.")
        return

    ks        = sorted(time_by_k)
    avg_times = [np.mean(time_by_k[k]) for k in ks]
    std_times = [np.std(time_by_k[k])  for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, avg_times, yerr=std_times, marker="o", linewidth=2,
                capsize=4, color=COLORS[0], label="Avg time +/- std")
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"{dataset_name} - Time vs K", fontsize=14, fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(plots_dir, f"{dataset_name}_time_vs_k.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_theta_error_vs_k(dataset_name, rows, orig, plots_dir):
    """
    For each of the 4 initiator params (a, b, c, d):
    Violin plot of |theta_i_pred - orig_i| grouped by k, with median trend line.
    """
    originals = list(orig)

    for i, (param_label, orig_val) in enumerate(zip(LABEL, originals)):
        col = f"theta_{i}"
        errors_by_k = defaultdict(list)

        for row in rows:
            k    = safe_float(row.get("k"))
            pred = safe_float(row.get(col))
            if k is not None and pred is not None:
                errors_by_k[int(k)].append(abs(pred - orig_val))

        if not errors_by_k:
            print(f"  [{dataset_name}] No data for {col}, skipping.")
            continue

        ks          = sorted(errors_by_k)
        data_groups = [errors_by_k[k] for k in ks]

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
                pc.set_facecolor(COLORS[i % len(COLORS)])
                pc.set_alpha(0.6)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)

        if single_idx:
            ax.scatter(
                [ks[j] for j in single_idx],
                [data_groups[j][0] for j in single_idx],
                color=COLORS[i % len(COLORS)], zorder=5, s=60,
                label="Single sample"
            )

        # Individual point overlay
        for j, k in enumerate(ks):
            jitter = np.random.uniform(-0.15, 0.15, size=len(data_groups[j]))
            ax.scatter(
                np.full(len(data_groups[j]), k) + jitter,
                data_groups[j],
                color="black", alpha=0.35, s=15, zorder=6
            )

        # Median trend line
        medians = [np.median(errors_by_k[k]) for k in ks]
        ax.plot(ks, medians, color="crimson", linewidth=2, linestyle="--",
                marker="D", markersize=6, zorder=7, label="Median error")
        ax.legend(fontsize=10)

        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel(f"|{param_label}_pred - {param_label}_orig|", fontsize=12)
        ax.set_title(
            f"{dataset_name} - {param_label} error vs K  "
            f"(original {param_label} = {orig_val})",
            fontsize=13, fontweight="bold"
        )
        ax.set_xticks(ks)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        fig.tight_layout()

        out = os.path.join(plots_dir, f"{dataset_name}_{param_label}_error_vs_k.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


def compute_mscore(avg_preds, orig):
    """
    Squared-weight contribution-weighted relative error (M-score).

        w_i  = orig_i^2 / sum(orig_i^2)
        M    = sum_i ( |orig_i - avg_pred_i| / orig_i ) * w_i

    Uses avg predicted values per k -> one clean scalar per k value.
    """
    denom_sq = sum(v ** 2 for v in orig)
    if denom_sq == 0:
        return None

    score = 0.0
    for orig_i, pred_i in zip(orig, avg_preds):
        if orig_i == 0:
            continue
        w_i    = (orig_i ** 2) / denom_sq
        score += (abs(orig_i - pred_i) / orig_i) * w_i
    return score


def plot_mscore_vs_k(dataset_name, rows, orig, plots_dir):
    """
    Line plot of M-score vs K.
    For each k: average the predicted theta values across all samples,
    then compute one M-score from those averages -> one point per k.
    """
    preds_by_k = defaultdict(lambda: defaultdict(list))

    for row in rows:
        k = safe_float(row.get("k"))
        if k is None:
            continue
        for i in range(4):
            val = safe_float(row.get(f"theta_{i}"))
            if val is not None:
                preds_by_k[int(k)][i].append(val)

    if not preds_by_k:
        print(f"  [{dataset_name}] No M-score data, skipping.")
        return

    ks      = sorted(preds_by_k)
    mscores = []

    for k in ks:
        avg_preds = tuple(
            np.mean(preds_by_k[k][i]) if preds_by_k[k][i] else None
            for i in range(4)
        )
        if any(p is None for p in avg_preds):
            mscores.append(None)
        else:
            mscores.append(compute_mscore(avg_preds, orig))

    valid_ks = [k for k, m in zip(ks, mscores) if m is not None]
    valid_ms = [m for m in mscores if m is not None]

    if not valid_ks:
        print(f"  [{dataset_name}] Could not compute M-scores, skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(valid_ks) * 0.6 + 2), 5))

    ax.plot(valid_ks, valid_ms, color="#7B2D8B", linewidth=2,
            marker="o", markersize=7, zorder=5, label="M-score")

    # Highlight the best (lowest M-score) k
    best_idx = int(np.argmin(valid_ms))
    ax.scatter(valid_ks[best_idx], valid_ms[best_idx],
               color="crimson", s=150, zorder=6, marker="*",
               label=f"Best k={valid_ks[best_idx]}  (M={valid_ms[best_idx]:.4f})")

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("M-score (lower = better)", fontsize=12)
    ax.set_title(f"{dataset_name} - M-score vs K", fontsize=13, fontweight="bold")
    ax.set_xticks(valid_ks)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Formula annotation
    ax.text(0.98, 0.97,
            r"$M = \sum_i \frac{|\bar{\theta}_i - i|}{i} \cdot \frac{i^2}{\sum i^2}$",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    fig.tight_layout()
    out = os.path.join(plots_dir, f"{dataset_name}_mscore_vs_k.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(params_path, results_dir, plots_dir):
    if not os.path.isfile(params_path):
        print(f"Error: params file not found: {params_path}")
        return
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        return

    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(plots_dir)}\n")

    params = load_params(params_path)
    print(f"Loaded params for {len(params)} datasets from {params_path}")

    csv_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".csv"))
    if not csv_files:
        print("No CSV files found in results directory.")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    for csv_file in csv_files:
        dataset_name = os.path.splitext(csv_file)[0]
        csv_path     = os.path.join(results_dir, csv_file)

        orig = params.get(dataset_name)
        if orig is None:
            for key in params:
                if isinstance(key, str) and key.lower() == dataset_name.lower():
                    orig = params[key]
                    break

        if orig is None:
            print(f"[{dataset_name}] WARNING: no entry in params.txt - skipping error plots.")

        rows = load_csv(csv_path)
        if not rows:
            print(f"[{dataset_name}] Empty CSV, skipping.")
            continue

        print(f"[{dataset_name}] {len(rows)} rows")
        plot_time_vs_k(dataset_name, rows, plots_dir)

        if orig is not None:
            plot_theta_error_vs_k(dataset_name, rows, orig, plots_dir)
            plot_mscore_vs_k(dataset_name, rows, orig, plots_dir)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot KronFit results: time vs k, theta error violins, and M-score."
    )
    parser.add_argument("--params",  default="params.txt",
                        help="Path to params.txt (default: ./params.txt)")
    parser.add_argument("--results", default="results",
                        help="Path to results directory (default: ./results)")
    parser.add_argument("--plots",   default="plots",
                        help="Output directory for plots (default: ./plots)")
    args = parser.parse_args()
    main(args.params, args.results, args.plots)