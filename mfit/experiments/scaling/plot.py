import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# -------- CONFIGURATION -------- #
RESULTS_DIR = "./results/Blog-Nat06all"
GPU_CSV_PATH = "../optimizers/results/Blog-Nat06all/adam.csv"
PLOTS_DIR = "./plots"
BREAKDOWN_DIR = os.path.join(PLOTS_DIR, "breakdowns")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(BREAKDOWN_DIR, exist_ok=True)

GT = np.array([0.999, 0.578, 0.517, 0.221])
THREAD_FILES = {1: "threads_1.csv", 2: "threads_2.csv", 4: "threads_4.csv", 8: "threads_8.csv", 16: "threads_16.csv"}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# -------- 1. DATA LOADING & PROCESSING -------- #
all_data = []

print("Loading data...")
for n_threads, filename in THREAD_FILES.items():
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df_avg = df.groupby('k').mean(numeric_only=True).reset_index()
        df_avg['label'] = f'{n_threads}T'
        df_avg['n_threads'] = n_threads
        df_avg['is_gpu'] = False
        all_data.append(df_avg)

if os.path.exists(GPU_CSV_PATH):
    gpu_df = pd.read_csv(GPU_CSV_PATH)
    gpu_avg = gpu_df.groupby('k').mean(numeric_only=True).reset_index()
    gpu_avg['label'] = 'GPU'
    gpu_avg['n_threads'] = 99
    gpu_avg['is_gpu'] = True
    all_data.append(gpu_avg)

master_df = pd.concat(all_data, ignore_index=True)
master_df['l2_norm'] = np.sqrt(((master_df[["P00","P01","P10","P11"]].values - GT)**2).sum(axis=1))

# -------- 2. LINE PLOTS -------- #

def format_line_plot(title, ylabel, filename, is_log=False):
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xticks(range(1, 19))
    plt.title(title, fontweight="bold")
    plt.ylabel(ylabel)
    plt.xlabel('k')
    if is_log:
        plt.yscale('log')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

# 01a: Log Performance
plt.figure(figsize=(10, 6))
sns.lineplot(data=master_df, x='k', y='full_fit', hue='label', marker='o')
format_line_plot('Execution Time vs k (Log Scale)', 'Time (s) - Log Scale', 'performance_vs_k_log.png', is_log=True)

# 01b: Linear Performance
plt.figure(figsize=(10, 6))
sns.lineplot(data=master_df, x='k', y='full_fit', hue='label', marker='o')
format_line_plot('Execution Time vs k (Linear Scale)', 'Time (s)', 'performance_vs_k_linear.png')

# 02: Speedup
base_time = master_df[master_df['n_threads'] == 1][['k', 'full_fit']].rename(columns={'full_fit': 'base_time'})
speedup_df = master_df[master_df['is_gpu'] == False].merge(base_time, on='k')
speedup_df['speedup'] = speedup_df['base_time'] / speedup_df['full_fit']

plt.figure(figsize=(10, 6))
sns.lineplot(data=speedup_df[speedup_df['n_threads'] > 1], x='k', y='speedup', hue='label', marker='o')
plt.axhline(1, color='gray', linestyle='--')
format_line_plot('Parallel Speedup vs k', 'Speedup Factor', 'speedup_vs_k.png')

# 03: Accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(data=master_df, x='k', y='l2_norm', hue='label', marker='s')
format_line_plot('Parameter Accuracy (L2 Error) vs k', 'L2 Distance to Ground Truth', 'accuracy_vs_k.png')

# -------- 3. STACKED BARS FOR ALL K -------- #
print("Generating breakdown bars for all k...")

k_values = sorted(master_df[master_df['n_threads'] == 1]['k'].unique())

for k in k_values:
    bar_data = master_df[master_df['k'] == k].sort_values('n_threads')

    if bar_data.empty:
        continue

    mcmc     = bar_data['mcmc_cpu'] + bar_data['warm_up_mcmc_cpu']
    gradient = bar_data['gradient_gpu'] + bar_data['optimizer_gpu']
    overhead = bar_data['init_cpu_gpu'] + bar_data['transfer_pcie']
    labels   = bar_data['label'].tolist()

    plt.figure(figsize=(12, 7))
    plt.bar(labels, mcmc,     label='MCMC (Sequential)',          color=COLORS[3])
    plt.bar(labels, gradient, bottom=mcmc,          label='Gradient + Opt (Parallel)', color=COLORS[0])
    plt.bar(labels, overhead, bottom=mcmc+gradient, label='Overhead',                  color=COLORS[7])

    plt.title(f'Execution Time Breakdown Analysis (k={int(k)})', fontweight="bold")
    plt.ylabel('Time (seconds)')
    plt.xlabel('Execution Mode')
    plt.legend(frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(BREAKDOWN_DIR, f'breakdown_k{int(k):02d}.png'),
                dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nDone! Summary plots are in /plots and breakdown bars for each k are in /plots/breakdowns/")