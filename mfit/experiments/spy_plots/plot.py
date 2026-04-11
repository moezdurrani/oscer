import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def parse_path(filepath):
    # expects .../CA-GR-QC/k12_s5.txt
    filename = os.path.splitext(os.path.basename(filepath))[0]  # k12_s5
    dataset  = os.path.basename(os.path.dirname(filepath))       # CA-GR-QC

    # parse k and s from filename like k12_s5
    k, s = None, None
    parts = filename.split('_')
    for p in parts:
        if p.startswith('k'):
            try: k = int(p[1:])
            except ValueError: pass
        if p.startswith('s'):
            try: s = int(p[1:])
            except ValueError: pass

    return dataset, filename, k, s


def load_graph(filepath):
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    edges.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return edges


def spy_plot(filepath):
    dataset, filename, k, s = parse_path(filepath)

    edges = load_graph(filepath)
    if not edges:
        print(f"No edges found in {filepath}")
        return

    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    n   = max(src.max(), dst.max()) + 1

    A = sp.coo_matrix(
        (np.ones(len(edges)), (src, dst)),
        shape=(n, n)
    )

    title_line1 = f"Dataset: {dataset}    k: {k}"
    title_line2 = f"{len(edges)} edges,  {n} nodes"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spy(A, markersize=max(0.5, 5 - np.log10(n)))
    ax.set_title(f"{title_line1}\n{title_line2}", fontweight="bold")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")

    outdir = os.path.join("./plots", dataset)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{filename}_spy.png")

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spy plot to: {outpath}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spy.py <file.txt>")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        print(f"Processing: {filepath}")
        spy_plot(filepath)