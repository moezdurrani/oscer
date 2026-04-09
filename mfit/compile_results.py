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
    all_param_keys = set()
    all_profile_keys = set()

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

        # --- Extract new fields ---
        params = data.get("best_parameters", {})
        best_ll = data.get("best_log_likelihood", None)
        total_time = data.get("total_time_seconds", None)
        profiling = data.get("profiling_timings_seconds", {})

        # Track all keys for dynamic headers
        all_param_keys.update(params.keys())
        all_profile_keys.update(profiling.keys())

        rows.append({
            "k": k,
            "sample": sample,
            "total_time": total_time,
            "best_ll": best_ll,
            "params": params,
            "profiling": profiling,
        })

    if not rows:
        print(f"  No valid JSON files found in {dataset_name}, skipping.")
        return

    # Sort by k then sample
    rows.sort(key=lambda r: (r["k"], r["sample"]))

    # Sort keys for consistent column order
    param_headers = sorted(all_param_keys)
    profile_headers = sorted(all_profile_keys)

    # Final column order
    fieldnames = (
        ["k", "sample", "total_time", "best_ll"]
        + param_headers
        + profile_headers
    )

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            flat = {
                "k": row["k"],
                "sample": row["sample"],
                "total_time": row["total_time"],
                "best_ll": row["best_ll"],
            }

            # Add parameters
            for key in param_headers:
                flat[key] = row["params"].get(key)

            # Add profiling timings
            for key in profile_headers:
                flat[key] = row["profiling"].get(key)

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
    directory = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/experiments/mfit"
    main(directory)
