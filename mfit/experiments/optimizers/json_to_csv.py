import os
import json
import csv

# -------- PATHS -------- #
BASE_DIR = "./"  # should be optimizers/
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# ----------------------- #

def extract_k_s(filename):
    """
    Extract k and s from filename like k13_s5.json
    """
    name = filename.replace(".json", "")
    parts = name.split("_")
    k = int(parts[0][1:])  # remove 'k'
    s = int(parts[1][1:])  # remove 's'
    return k, s


def process_optimizer(dataset_path, optimizer_name, output_csv_path):
    optimizer_path = os.path.join(dataset_path, optimizer_name)

    if not os.path.exists(optimizer_path):
        print(f"Skipping missing optimizer dir: {optimizer_path}")
        return

    rows = []
    all_profile_keys = set()

    # -------- FIRST PASS: collect all keys -------- #
    for file in os.listdir(optimizer_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(optimizer_path, file), "r") as f:
            data = json.load(f)

        profiling = data.get("profiling_timings_seconds", {})
        all_profile_keys.update(profiling.keys())

    all_profile_keys = sorted(list(all_profile_keys))

    # -------- SECOND PASS: build rows -------- #
    for file in os.listdir(optimizer_path):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(optimizer_path, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        k, s = extract_k_s(file)

        params = data.get("best_parameters", {})

        row = {
            "k": k,
            "s": s,
            "total_time_seconds": data.get("total_time_seconds"),
            "best_log_likelihood": data.get("best_log_likelihood"),

            "P00": params.get("P00"),
            "P01": params.get("P01"),
            "P10": params.get("P10"),
            "P11": params.get("P11"),
        }

        profiling = data.get("profiling_timings_seconds", {})

        for key in all_profile_keys:
            row[key] = profiling.get(key, None)

        rows.append(row)

    # sort nicely
    rows.sort(key=lambda x: (x["k"], x["s"]))

    # -------- WRITE CSV -------- #
    fieldnames = [
    "k", "s",
    "total_time_seconds",
    "best_log_likelihood",
    "P00", "P01", "P10", "P11"
    ] + all_profile_keys


    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv_path}")


def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    datasets = os.listdir(EXPERIMENTS_DIR)

    for dataset in datasets:
        dataset_path = os.path.join(EXPERIMENTS_DIR, dataset)

        if not os.path.isdir(dataset_path):
            continue

        print(f"\nProcessing dataset: {dataset}")

        # create results/<dataset>/
        dataset_result_dir = os.path.join(RESULTS_DIR, dataset)
        os.makedirs(dataset_result_dir, exist_ok=True)

        for optimizer in ["adam", "adamw", "rmsprop"]:
            output_csv = os.path.join(dataset_result_dir, f"{optimizer}.csv")

            process_optimizer(
                dataset_path,
                optimizer,
                output_csv
            )


if __name__ == "__main__":
    main()