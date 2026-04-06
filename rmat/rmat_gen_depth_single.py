import random
import time
import os
import math
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import subprocess
import shutil

# Each worker writes to its own PID-based file
def get_worker_file(output_dir):
    pid = os.getpid()
    return os.path.join(output_dir, f"part_{pid}.txt")

# Buffered write to reduce IO overhead
def write_edge(u, v, output_dir, buffer, BUFFER_SIZE=50000):
    buffer.append(f"{u} {v}\n")
    if len(buffer) >= BUFFER_SIZE:
        filepath = get_worker_file(output_dir)
        with open(filepath, "a") as f:
            f.writelines(buffer)
        buffer.clear()

# Flush remaining buffered edges to disk
def flush_buffer(output_dir, buffer):
    if buffer:
        filepath = get_worker_file(output_dir)
        with open(filepath, "a") as f:
            f.writelines(buffer)
        buffer.clear()


def collect_tasks(exp_edges, k, u_offset, v_offset, size, a_n, b_n, c_n, d_n, depth, depth_limit):
    """
    Recursively collect leaf tasks on the main process without spawning anything.
    Returns a flat list of (exp_edges, k, u_offset, v_offset, size, a_n, b_n, c_n, d_n)
    tuples — output_dir is added later before submission.
    """
    if exp_edges <= 0:
        return []

    # Reached depth limit OR k too small to recurse further — hand off to iter_rmat
    if depth >= depth_limit or k == 0:
        return [(exp_edges, k, u_offset, v_offset, size, a_n, b_n, c_n, d_n)]

    partition = size // 2
    tasks = []

    quadrants = [
        (a_n, u_offset,              v_offset),
        (b_n, u_offset,              v_offset + partition),
        (c_n, u_offset + partition,  v_offset),
        (d_n, u_offset + partition,  v_offset + partition),
    ]

    for prob, u_off, v_off in quadrants:
        tasks += collect_tasks(
            exp_edges * prob, k - 1,
            u_off, v_off,
            partition, a_n, b_n, c_n, d_n,
            depth + 1, depth_limit
        )

    return tasks


# Worker function called by the pool — unpacks tuple and runs iter_rmat
def iter_rmat_worker(args):
    exp_edges, k, u_offset, v_offset, size, a_n, b_n, c_n, d_n, output_dir = args
    iter_rmat(exp_edges, k, size, a_n, b_n, c_n, d_n, u_offset, v_offset, output_dir)


# Iterative RMAT edge generation
def iter_rmat(exp_edges, k, num_nodes, a, b, c, d, u_offset, v_offset, output_dir):

    exp_edges_int = int(exp_edges)
    if exp_edges_int == 0:
        return

    buffer = []

    for _ in range(exp_edges_int):
        u, v = 0, 0
        partition = num_nodes // 2

        for _ in range(k):
            r = random.random()
            if r < a:
                pass
            elif r < a + b:
                v += partition
            elif r < a + b + c:
                u += partition
            else:
                u += partition
                v += partition
            partition //= 2

        write_edge(u + u_offset, v + v_offset, output_dir, buffer)

    flush_buffer(output_dir, buffer)


# Fast OS-Accelerated merge using CAT
def merge_temp_files(temp_dir, output_dir, k, s, exp_edges, graph_gen_time):
    merge_time = time.time()
    print("\nMerging temp files using OS fast path")

    part_files = glob.glob(os.path.join(temp_dir, "part_*.txt"))
    print(f"Found {len(part_files)} partial files.")

    final_filename = f"k{k}_s{s}.txt"
    final_path = os.path.join(output_dir, final_filename)
    print(f"Writing to final file: {final_filename}")

    print("Counting edges")
    total_edges = 0
    for pf in part_files:
        with open(pf, "r") as f:
            for _ in f:
                total_edges += 1

    print(f"Total edges counted: {total_edges}")
    edges_diff = ((exp_edges - total_edges) / exp_edges) * 100
    print(f"Edges difference percentage from expected: {edges_diff:.4f}%")

    with open(final_path, "w") as f:
        f.write("# Synthetic Kronecker Graph Using RMAT\n")
        f.write(f"# K : {k}, Sample: {s}\n")
        f.write(f"# Number of edges: {total_edges}\n")
        f.write(f"# Graph generation time (s): {graph_gen_time:.2f}\n")

    subprocess.run(
        f"cat {temp_dir}/part_*.txt >> {final_path}",
        shell=True,
        check=False
    )

    print(f"Merged into {final_filename} in {time.time() - merge_time:.2f}s")

    for pf in part_files:
        os.remove(pf)
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("Cleanup done")


def generate(a, b, c, d, k, s, seed_val, output_dir):
    start_time = time.time()
    random.seed(seed_val)

    tot = a + b + c + d
    a_n, b_n, c_n, d_n = a/tot, b/tot, c/tot, d/tot

    exp_edges = tot ** k
    size = 2 ** k

    cpu_count = os.cpu_count()
    # Base depth: how many levels to recurse before handing off to iter_rmat
    # 4^depth tasks will be generated, aim for ~4x cpu_count tasks for good pool utilization
    base_depth = min(k, math.ceil(math.log(cpu_count * 4, 4)))

    print(f"Generating RMAT graph for k={k} sample={s}")
    print(f"Expected edges: {exp_edges}")
    print(f"CPU cores detected: {cpu_count}")
    print(f"Base recursion depth: {base_depth}")

    temp_dir = os.path.join(output_dir, f"temp_k{k}_s{s}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary files directory: {temp_dir}")

    # Collect all leaf tasks on the main process — no processes spawned here
    all_tasks = collect_tasks(exp_edges, k, 0, 0, size, a_n, b_n, c_n, d_n, 0, base_depth)

    # Attach output_dir to each task
    all_tasks = [(*t, temp_dir) for t in all_tasks]

    print(f"Total leaf tasks collected: {len(all_tasks)}")

    # Single flat pool — no nesting, no overhead
    with ProcessPoolExecutor(max_workers=cpu_count) as pool:
        list(pool.map(iter_rmat_worker, all_tasks))

    graph_gen_time = time.time() - start_time
    print(f"Graph generation complete in {graph_gen_time:.2f}s")

    merge_temp_files(temp_dir, output_dir, k, s, exp_edges, graph_gen_time)

    print(f"Total execution time: {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("init_matrix", nargs=4, type=float, help="Four initiator matrix values")
    parser.add_argument("k", type=int, help="Scale factor, nodes = 2^k")
    parser.add_argument("s", type=int, help="Sample number")
    parser.add_argument("seed_val", type=int, help="Random seed")
    parser.add_argument("output_dir", type=str, help="Output directory for edge list")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generate(args.init_matrix[0], args.init_matrix[1], args.init_matrix[2], args.init_matrix[3],
             args.k, args.s, args.seed_val, args.output_dir)


if __name__ == "__main__":
    main()