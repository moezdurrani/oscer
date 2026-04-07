import networkx as nx
from mfit import mfit
import torch
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    # Kronfit usually processes directed edge lists
    G = nx.read_edgelist(args.data_path, create_using=nx.DiGraph(), nodetype=int)
    
    # Strictly following the hyperparams from your kronfit command line
    # grad_samples in SNAP is usually 100,000, we mimic the step logic here
    model = mfit(
        graph_temp=args.data_path, 
        init_matrix=[0.9, 0.7, 0.5, 0.2], 
        iterations=100,
        learning_rate=1e-5, # Default LrnRate from kronfit.cpp
        warmup_mcmc=10000, 
        grad_samples=100000, 
        
    )

    results = model.fit()

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print("\nOptimization Finished.")

if __name__ == "__main__":
    main()