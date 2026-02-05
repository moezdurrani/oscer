import random
import time
import os
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

# Wrapper for multiprocessing calls
def rec_wrapper(args):
    rec_gen_parallel(*args)
    return None   # no return to avoid serialization cost

def rec_gen_parallel(exp_edges, k, u_offset, v_offset, size, a_n, b_n, c_n, d_n, total_cores, output_dir):

    # Stop if no work
    if exp_edges <= 0:
        return

    # Base case: 1x1 cell
    if k == 0:
        prob = min(1.0, exp_edges)
        if random.random() < prob:
            buffer = []
            write_edge(u_offset, v_offset, output_dir, buffer)
            flush_buffer(output_dir, buffer)
        return

    partition = size // 2

    # If enough cores are remaining, fork 4 tasks
    if total_cores >= 4:

        next_cores = total_cores - 4 # Can be improved

        tasks = [
            # top-left
            (exp_edges * a_n, k - 1,
             u_offset, v_offset,
             partition, a_n, b_n, c_n, d_n,
             next_cores, output_dir),

            # top-right
            (exp_edges * b_n, k - 1,
             u_offset, v_offset + partition,
             partition, a_n, b_n, c_n, d_n,
             next_cores, output_dir),

            # bottom-left
            (exp_edges * c_n, k - 1,
             u_offset + partition, v_offset,
             partition, a_n, b_n, c_n, d_n,
             next_cores, output_dir),

            # bottom-right
            (exp_edges * d_n, k - 1,
             u_offset + partition, v_offset + partition,
             partition, a_n, b_n, c_n, d_n,
             next_cores, output_dir)
        ]

        # Spawn parallel workers
        with ProcessPoolExecutor(max_workers=4) as ex:
            list(ex.map(rec_wrapper, tasks))

        return

    # If not enough cores, fall back to iterative RMAT
    iter_rmat(exp_edges, k, size, a_n, b_n, c_n, d_n,
              u_offset, v_offset, output_dir)


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

    # Count edges by counting lines in part files
    part_files = glob.glob(os.path.join(temp_dir, "part_*.txt"))
    print(f"Found {len(part_files)} partial files.")

    # Output filename
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
    edges_diff = ( (exp_edges) - (total_edges) ) // exp_edges * 100
    print(f"Edges difference percentage from expected: {edges_diff:.4f}%")

    # Write header with correct number of edges
    with open(final_path, "w") as f:
        f.write("# Synthetic Kronecker Graph Using RMAT\n")
        f.write(f"# K : {k}, Sample: {s}\n")
        f.write(f"# Number of edges: {total_edges}\n")
        f.write(f"# Graph generation time (s): {graph_gen_time:.2f}\n")

    # OS-accelerated merge (fastest possible)
    subprocess.run(
        f"cat {temp_dir}/part_*.txt >> {final_path}",
        shell=True,
        check=False
    )

    print(f"Merged into {final_filename} in {time.time() - merge_time:.2f}s")

    # Cleanup
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
    total_cores = os.cpu_count()

    print(f"Generating RMAT graph for k {k} sample {s}")
    print(f"Expected edges: {exp_edges}")
    print(f"CPU cores detected: {total_cores}")

    temp_dir = os.path.join(output_dir, f"temp_k{k}_s{s}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Folder for Temporary Files: {temp_dir}")

    rec_gen_parallel(exp_edges, k, 0, 0, size, a_n, b_n, c_n, d_n, total_cores, temp_dir)

    graph_gen_time = time.time() - start_time
    print(f"Graph generation complete in {graph_gen_time:.2f}s")
    merge_temp_files(temp_dir, output_dir, k, s, exp_edges, graph_gen_time)

    print(f"Total execution time: {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("init_matrix", nargs=4, type=float, help="Four parameters for the initiator matrix separated by space")
    parser.add_argument("k", type=int, help="scale factor, # of nodes: 2^k")
    parser.add_argument("s", type=int, help="sample number")
    parser.add_argument("seed_val", type=int, help="Seed Value for random number generator")
    parser.add_argument("output_dir", type=str, help="where do you want the edge list to be stored")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    a, b, c, d = args.init_matrix
    generate(args.init_matrix[0], args.init_matrix[1], args.init_matrix[2], args.init_matrix[3], args.k, args.s, args.seed_val, args.output_dir)


if __name__ == "__main__":
    main()
