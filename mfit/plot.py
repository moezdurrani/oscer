import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Config
# -----------------------------
CSV_FILE = "results.csv"
TRUE_THETA = np.array([0.902, 0.253, 0.221, 0.582])

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_FILE)

k = df["k"]

# -----------------------------
# 1. Time vs k
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(k, df["kronfit_time"], marker="o", label="KronFit")
plt.plot(k, df["mfit_time"], marker="s", label="MFit")

plt.xlabel("k")
plt.ylabel("Time (seconds)")
plt.title("Runtime Comparison: Time vs k")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("time_vs_k.png", dpi=300)
plt.show()

# -----------------------------
# Delta log-likelihood vs k
# -----------------------------
delta_ll = df["mfit_best_ll"] - df["kronfit_best_ll"]

plt.figure(figsize=(8, 5))
plt.plot(k, delta_ll, marker="o")

plt.xlabel("k")
plt.ylabel("Δ Log-Likelihood")
plt.title("Log-Likelihood Improvement vs k")

# Annotation box
annotation_text = (
    "ΔLL = MFIT − KronFit\n"
    "ΔLL > 0: MFIT better fit\n"
    "ΔLL < 0: KronFit better fit"
)

plt.text(
    0.02, 0.95,
    annotation_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.grid(True)
plt.tight_layout()
plt.savefig("delta_best_ll_vs_k.png", dpi=300)
plt.close()



# -----------------------------
# 2. Parameter absolute error plots
# -----------------------------
for i in range(4):
    kron_col = f"kronfit_theta_{i+1}"
    mfit_col = f"mfit_theta_{i+1}"

    kron_err = np.abs(df[kron_col] - TRUE_THETA[i])
    mfit_err = np.abs(df[mfit_col] - TRUE_THETA[i])

    plt.figure(figsize=(8, 5))
    plt.plot(k, kron_err, marker="o", label="KronFit Error")
    plt.plot(k, mfit_err, marker="s", label="MFit Error")

    plt.xlabel("k")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error vs k for θ{i+1} (true = {TRUE_THETA[i]})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"theta_{i+1}_error_vs_k.png", dpi=300)
    plt.show()

