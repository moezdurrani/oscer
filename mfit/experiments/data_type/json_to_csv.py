import os
import json
import csv
import math

# -------- PATHS -------- #
BASE_DIR = "./"  # inside data_type/
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# ----------------------- #

DATA_TYPES = ["baseline", "economy", "stress"]


def extract_k_s(filename):
    name = filename.replace(".json", "")
    parts = name.split("_")
    k = int(parts[0][1:])
    s = int(parts[1][1:])
    return k, s


def safe_ll(value):
    """
    Keep inf / -inf as-is (string), otherwise numeric
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isinf(value):
            return value  # keeps inf / -inf
    return value


def process_data_type(dataset_path, dtype_name, output_csv_path):
    dtype_path = os.path.join(dataset_path, dtype_name)

    if not os.path.exists(dtype_path):
        print(f"Skipping missing dir: {dtype_path}")
        return

    rows = []
    all_profile_keys = set()

    # -------- FIRST PASS -------- #
    for file in os.listdir(dtype_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(dtype_path, file), "r") as f:
            data = json.load(f)

        profiling = data.get("profiling_timings_seconds", {})
        all_profile_keys.update(profiling.keys())

    all_profile_keys = sorted(list(all_profile_keys))

    # -------- SECOND PASS -------- #
    for file in os.listdir(dtype_path):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(dtype_path, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        k, s = extract_k_s(file)
        params = data.get("best_parameters", {})

        row = {
            "k": k,
            "s": s,
            "total_time_seconds": data.get("total_time_seconds"),
            "best_log_likelihood": safe_ll(data.get("best_log_likelihood")),

            # 🔥 NEW FIELD (AFTER best_ll)
            "peak_vram_mb": data.get("peak_vram_mb"),

            "P00": params.get("P00"),
            "P01": params.get("P01"),
            "P10": params.get("P10"),
            "P11": params.get("P11"),
        }

        profiling = data.get("profiling_timings_seconds", {})

        for key in all_profile_keys:
            row[key] = profiling.get(key, None)

        rows.append(row)

    # sort
    rows.sort(key=lambda x: (x["k"], x["s"]))

    # -------- WRITE CSV -------- #
    fieldnames = [
        "k", "s",
        "total_time_seconds",
        "best_log_likelihood",
        "peak_vram_mb", 
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

        dataset_result_dir = os.path.join(RESULTS_DIR, dataset)
        os.makedirs(dataset_result_dir, exist_ok=True)

        for dtype in DATA_TYPES:
            output_csv = os.path.join(dataset_result_dir, f"{dtype}.csv")

            process_data_type(
                dataset_path,
                dtype,
                output_csv
            )


if __name__ == "__main__":
    main()