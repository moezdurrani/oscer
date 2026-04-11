import os
import json
import csv

# -------- PATHS -------- #
BASE_DIR = "./"
MFIT_DIR = os.path.join(BASE_DIR, "experiments", "mfit")
KRONFIT_DIR = os.path.join(BASE_DIR, "experiments", "kronfit")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# ----------------------- #

os.makedirs(RESULTS_DIR, exist_ok=True)


def process_folder(input_dir, output_csv):
    rows = []

    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue

        dataset_name = os.path.splitext(file)[0]
        filepath = os.path.join(input_dir, file)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

        theta = data.get("theta", [None, None, None, None])
        best_ll = data.get("best_ll", None)
        time_val = data.get("time", None)

        if len(theta) != 4:
            theta = (theta + [None]*4)[:4]

        row = [
            dataset_name,
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            best_ll,
            time_val
        ]

        rows.append(row)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "dataset_name",
            "theta_00",
            "theta_01",
            "theta_10",
            "theta_11",
            "best_ll",
            "time"
        ])

        writer.writerows(rows)

    print(f"Saved: {output_csv}")


# -------- RUN -------- #
process_folder(MFIT_DIR, os.path.join(RESULTS_DIR, "mfit.csv"))
process_folder(KRONFIT_DIR, os.path.join(RESULTS_DIR, "kronfit.csv"))