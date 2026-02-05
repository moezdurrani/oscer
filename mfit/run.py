import networkx as nx
from mfit import mfit
import torch
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="data path")
    parser.add_argument("output_file", type=str, help="output file")
    args = parser.parse_args()
    data_path = args.data_path
    output_file = args.output_file
    # data_path = "./data/ATP-GR-QC/k4_s2.txt"
    # output_file = "results/k4.json"
    
    # Hyperparameters
    init_matrix = [0.9, 0.7, 0.5, 0.2]
    iterations = 100
    lr = 2e-4
    grad_samples = 100000
    warmup = 10000
    mcmc_steps = 1000
    
    # 1. Load the graph
    print(f"Loading graph from {data_path}...")
    # Assuming the text file is an edge list (u v)
    G = nx.read_edgelist(data_path, create_using=nx.DiGraph(), nodetype=int)
    
    # 2. Initialize the model
    model = mfit(
        graph_temp=G, 
        init_matrix=init_matrix, 
        learning_rate=lr, 
        warmup_mcmc=warmup, 
        grad_samples=grad_samples, 
        iterations=iterations
    )
    
    # 3. Run the optimization
    results = model.fit(
        iterations=iterations, 
        grad_samples=grad_samples, 
        warmup_mcmc=warmup, 
        mcmc_per_iter=mcmc_steps,
        verbose=True
    )
    
    # 4. Write results to results.txt
    print(f"\nWriting results to {output_file}...")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("Done!")

if __name__ == "__main__":
    main()
