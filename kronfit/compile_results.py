import os
import json
import csv
import re
import argparse


def parse_filename(filename):
    """Extract k and sample values from a filename like k10_s5.json."""
    match = re.match(r"k(\d+)_s(\d+)\.json$", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def process_dataset(dataset_dir, results_dir):
    """Read all JSON files in dataset_dir and write a CSV to results_dir."""
    dataset_name = os.path.basename(dataset_dir)
    csv_path = os.path.join(results_dir, f"{dataset_name}.csv")

    rows = []
    max_theta_len = 0

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".json"):
            continue

        k, sample = parse_filename(filename)
        if k is None:
            print(f"  Skipping unrecognized file: {filename}")
            continue

        filepath = os.path.join(dataset_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Could not read {filename}: {e}")
            continue

        theta = data.get("theta", [])
        best_ll = data.get("best_ll", None)
        time_val = data.get("time", None)

        max_theta_len = max(max_theta_len, len(theta))
        rows.append({
            "k": k,
            "sample": sample,
            "time": time_val,
            "best_ll": best_ll,
            "theta": theta,
        })

    if not rows:
        print(f"  No valid JSON files found in {dataset_name}, skipping.")
        return

    # Sort by k then sample for readability
    rows.sort(key=lambda r: (r["k"], r["sample"]))

    # Build dynamic headers for theta columns
    theta_headers = [f"theta_{i}" for i in range(max_theta_len)]
    fieldnames = ["k", "sample", "time", "best_ll"] + theta_headers

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            flat = {
                "k": row["k"],
                "sample": row["sample"],
                "time": row["time"],
                "best_ll": row["best_ll"],
            }
            # Pad theta values if shorter than max
            for i, val in enumerate(row["theta"]):
                flat[f"theta_{i}"] = val
            writer.writerow(flat)

    print(f"  Wrote {len(rows)} rows -> {csv_path}")


def main(parent_dir):
    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        print(f"Error: '{parent_dir}' is not a valid directory.")
        return

    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")

    # Iterate over immediate subdirectories (dataset folders)
    entries = sorted(os.listdir(parent_dir))
    for entry in entries:
        if entry == "results":
            continue  # skip the results folder itself

        dataset_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(dataset_path):
            continue  # skip non-directory files

        print(f"Processing dataset: {entry}")
        process_dataset(dataset_path, results_dir)

    print("\nDone.")


if __name__ == "__main__":
    directory = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/experiments/kronfit"
    main(directory)
