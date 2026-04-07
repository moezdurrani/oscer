import networkx as nx
from mfit import mfit
import torch
import json
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--gpu_synchronize", action="store_true", help="Enable accurate GPU profiling")
    args = parser.parse_args()
    start_time = time.time()
    
    # Strictly following the hyperparams from your kronfit command line
    # grad_samples in SNAP is usually 100,000, we mimic the step logic here
    model = mfit(
        graph_file_path=args.data_path, 
        init_matrix=[0.9, 0.7, 0.5, 0.2], 
        iterations=100,
        warmup_mcmc=10000, 
        mcmc_per_iter=100000,
        learning_rate=0.05,
        gpu_synchronize=args.gpu_synchronize
    )

    best_P, best_ll, profiling_data = model.fit()

    total_time = time.time() - start_time

    output_data = {
        "best_parameters": {
            "P00": best_P[0],
            "P01": best_P[1],
            "P10": best_P[2],
            "P11": best_P[3]
        },
        "best_log_likelihood": best_ll,
        "total_time_seconds": total_time,
        "profiling_timings_seconds": profiling_data
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nOptimization Finished. Results successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()