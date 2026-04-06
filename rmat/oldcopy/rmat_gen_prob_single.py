import random
import time
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import subprocess
import shutil

# --- WORKER LOGIC ---

def get_worker_file(output_dir):
    return os.path.join(output_dir, f"part_{os.getpid()}.txt")

def write_edge(u, v, output_dir, buffer, BUFFER_SIZE=50000):
    buffer.append(f"{u} {v}\n")
    if len(buffer) >= BUFFER_SIZE:
        with open(get_worker_file(output_dir), "a") as f:
            f.writelines(buffer)
        buffer.clear()

def flush_buffer(output_dir, buffer):
    if buffer:
        with open(get_worker_file(output_dir), "a") as f:
            f.writelines(buffer)
        buffer.clear()

def worker_task(args):
    """The function actually executed by the ProcessPool"""
    exp_edges, k, size, a, b, c, d, u_offset, v_offset, output_dir = args
    iter_rmat(exp_edges, k, size, a, b, c, d, u_offset, v_offset, output_dir)
    return None

def iter_rmat(exp_edges, k, num_nodes, a, b, c, d, u_offset, v_offset, output_dir):
    exp_edges_int = int(exp_edges)
    if exp_edges_int <= 0: return
    
    buffer = []
    for _ in range(exp_edges_int):
        u, v = 0, 0
        partition = num_nodes // 2
        for _ in range(k):
            r = random.random()
            if r < a: pass
            elif r < a + b: v += partition
            elif r < a + b + c: u += partition
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

# --- COORDINATION LOGIC ---

def split_cores(rem_cores, a_n, b_n, c_n, d_n):
    probs = [a_n, b_n, c_n, d_n]
    splits = [int(p * rem_cores) for p in probs]
    remaining = rem_cores - sum(splits)
    # Assign remaining cores to highest probability quadrants
    order = sorted(range(4), key=lambda i: probs[i], reverse=True)
    for i in range(remaining):
        splits[order[i]] += 1
    return splits

def get_tasks_recursive(exp_edges, k, u_offset, v_offset, size, a_n, b_n, c_n, d_n, total_cores, output_dir):
    """Returns a list of task arguments for the process pool"""
    if exp_edges <= 0: return []
    
    # Base Case: Only 1 core assigned to this branch, or reached single node
    if total_cores <= 1 or k == 0:
        return [(exp_edges, k, size, a_n, b_n, c_n, d_n, u_offset, v_offset, output_dir)]

    partition = size // 2
    splits = split_cores(total_cores, a_n, b_n, c_n, d_n)
    
    tasks = []
    # Recursively collect tasks for each quadrant
    quadrants = [
        (a_n, u_offset, v_offset),              # Top-Left
        (b_n, u_offset, v_offset + partition),  # Top-Right
        (c_n, u_offset + partition, v_offset),  # Bottom-Left
        (d_n, u_offset + partition, v_offset + partition) # Bottom-Right
    ]
    
    for i, (prob, u_o, v_o) in enumerate(quadrants):
        tasks.extend(get_tasks_recursive(
            exp_edges * prob, k - 1, u_o, v_o, partition, 
            a_n, b_n, c_n, d_n, splits[i], output_dir
        ))
    return tasks

# --- MAIN EXECUTION ---

def generate(a, b, c, d, k, s, seed_val, output_dir):
    start_time = time.time()
    random.seed(seed_val)
    
    tot = a + b + c + d
    a_n, b_n, c_n, d_n = a/tot, b/tot, c/tot, d/tot
    exp_edges = tot ** k
    size = 2 ** k
    total_cores = os.cpu_count()

    temp_dir = os.path.join(output_dir, f"temp_k{k}_s{s}")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Generate the task list (Very fast, single-threaded)
    all_tasks = get_tasks_recursive(exp_edges, k, 0, 0, size, a_n, b_n, c_n, d_n, total_cores, temp_dir)
    
    # 2. Process tasks in parallel using ONE pool
    print(f"Executing {len(all_tasks)} parallel tasks on {total_cores} cores...")
    with ProcessPoolExecutor(max_workers=total_cores) as executor:
        list(executor.map(worker_task, all_tasks))

    graph_gen_time = time.time() - start_time
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
