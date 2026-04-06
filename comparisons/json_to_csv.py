import os
import json
import csv

EXP_DIR = "experiments"

def extract_k(filename):
    # k10_s2.json → 10
    return int(filename.split('_')[0][1:])


def main():
    # loop through datasets
    for dataset in os.listdir(EXP_DIR):
        dataset_path = os.path.join(EXP_DIR, dataset)

        if not os.path.isdir(dataset_path):
            continue

        print(f"\nProcessing dataset: {dataset}")

        # collect json files only
        files = [f for f in os.listdir(dataset_path) if f.endswith(".json")]

        # sort numerically by k
        files = sorted(files, key=extract_k)

        # output CSV path
        csv_path = os.path.join(EXP_DIR, f"{dataset}.csv")

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # header
            writer.writerow(["k", "time", "a", "b", "c", "d", "best_ll"])

            for file in files:
                json_path = os.path.join(dataset_path, file)

                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # skip failed runs (if you added error logging)
                    if data.get("best_P") is None:
                        print(f"Skipping failed: {file}")
                        continue

                    k = extract_k(file)
                    time_val = data["time"]
                    best_ll = data["best_ll"]
                    a, b, c, d = data["best_P"]

                    writer.writerow([k, time_val, a, b, c, d, best_ll])

                except Exception as e:
                    print(f"Error reading {file}: {e}")

        print(f"Saved CSV → {csv_path}")


if __name__ == "__main__":
    main()