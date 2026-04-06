import argparse
import json
import time
import networkx as nx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "rmsprop", "lbfgs"],
                        help="Which optimizer to use (default: adam)")
    args = parser.parse_args()

    G = nx.read_edgelist(args.data_path, create_using=nx.DiGraph(), nodetype=int)

    if args.optimizer == "adamw":
        from mfit_adamw import mfit
    elif args.optimizer == "rmsprop":
        from mfit_rmsprop import mfit
    elif args.optimizer == "lbfgs":
        from mfit_lbfgs import mfit
    else:
        from mfit_adam import mfit  # your original file, rename it to mfit_adam.py

    model = mfit(
        graph_temp=G,
        init_matrix=[0.9, 0.7, 0.5, 0.2],
        learning_rate=1e-5,
        warmup_mcmc=10000,
        grad_samples=100000,
        iterations=100
    )

    start_time = time.time()
    model.fit(
        iterations=100,
        grad_samples=100000,
        warmup_mcmc=10000,
        mcmc_per_iter=1000
    )
    total_time = time.time() - start_time

    results = {
        "optimizer": args.optimizer,
        "theta": model.P.detach().cpu().numpy().flatten().tolist(),
        "time": total_time
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nOptimization Finished.")
    print(f"Time: {total_time:.1f}s")
    print(f"Theta: {results['theta']}")


if __name__ == "__main__":
    main()
