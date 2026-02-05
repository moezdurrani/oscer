import json
import csv
import os
import re

KRONFIT_DIR = "results/kronfit_backup"
MFIT_DIR = "results/mfit"
OUTPUT_FILE = "results.csv"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_k(filename):
    match = re.search(r"k(\d+)\.json", filename)
    return int(match.group(1)) if match else None

rows = []

for fname in os.listdir(KRONFIT_DIR):
    if not fname.endswith(".json"):
        continue

    k = extract_k(fname)
    if k is None:
        continue

    kron_path = os.path.join(KRONFIT_DIR, fname)
    mfit_path = os.path.join(MFIT_DIR, fname)

    if not os.path.exists(mfit_path):
        print(f"Skipping k={k}, missing mfit file")
        continue

    kron = load_json(kron_path)
    mfit = load_json(mfit_path)

    row = {
        "k": k,
        "kronfit_theta": kron["theta"],
        "kronfit_best_ll": kron["best_ll"],
        "kronfit_time": kron["time"],
        "mfit_theta": mfit["theta"],
        "mfit_best_ll": mfit["best_ll"],
        "mfit_time": mfit["time"],
    }

    rows.append(row)

# Sort by k
rows.sort(key=lambda x: x["k"])

# Write CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "k",
        "kronfit_theta_1", "kronfit_theta_2", "kronfit_theta_3", "kronfit_theta_4",
        "kronfit_best_ll", "kronfit_time",
        "mfit_theta_1", "mfit_theta_2", "mfit_theta_3", "mfit_theta_4",
        "mfit_best_ll", "mfit_time"
    ])

    for r in rows:
        writer.writerow([
            r["k"],
            *r["kronfit_theta"],
            r["kronfit_best_ll"],
            r["kronfit_time"],
            *r["mfit_theta"],
            r["mfit_best_ll"],
            r["mfit_time"]
        ])

print(f"Saved merged results to {OUTPUT_FILE}")

