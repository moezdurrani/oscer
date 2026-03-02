import os
import argparse

DATA_ROOT = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/data"
EXPERIMENT_ROOT = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/experiments/mfit"
PROJECT_ROOT = os.path.expanduser("~/mfit")
SBATCH_DIR = os.path.join(PROJECT_ROOT, "sbatches")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PARTITION = "sooner_gpu_test"
LOW_K_THRESHOLD, HIGH_K_THRESHOLD = 15, 21

# Run times (seconds) for kronfit per k values
RUN_TIMES = {1:107.65, 2:107.65, 3:107.65, 4:107.65, 5:107.65, 6:107.65, 7:107.65, 8:107.65, 9:107.65, 10:107.65, 11:107.65, 12:107.65, 13:107.65, 14:147.29, 15:147.29, 16:147.29, 17:147.29, 18:147.29, 19:147.29, 20:147.29, 21:203.82, 22:310.7, 23:550.45}

def write_sbatch_header(f, dataset, k, k_start, k_end, sample):
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --partition={PARTITION}\n")
    # f.write("set -e\n")
    f.write("#SBATCH --ntasks=1\n")
    if sample == -1:
        if k < LOW_K_THRESHOLD:
            f.write(f"#SBATCH --output={LOG_DIR}/{dataset}/k{k_start}_k{k_end}_%J_out.txt\n")
            f.write(f"#SBATCH --error={LOG_DIR}/{dataset}/k{k_start}_k{k_end}_%J_err.txt\n")
            f.write(f"#SBATCH --job-name={dataset}_k{k_start}_k{k_end}\n")
        else:
            f.write(f"#SBATCH --output={LOG_DIR}/{dataset}/k{k}_%J_out.txt\n")
            f.write(f"#SBATCH --error={LOG_DIR}/{dataset}/k{k}_%J_err.txt\n")
            f.write(f"#SBATCH --job-name={dataset}_k_{k}\n")
    else:
        f.write(f"#SBATCH --output={LOG_DIR}/{dataset}/k{k}_s{sample}_%J_out.txt\n")
        f.write(f"#SBATCH --error={LOG_DIR}/{dataset}/k{k}_s{sample}_%J_err.txt\n")
        f.write(f"#SBATCH --job-name={dataset}_k{k}_s{sample}\n")

    # These need to be added after the time requeted is calculated    
    remaining = []
    remaining.append("#SBATCH --mail-user=moezdurrani@ou.edu\n")
    remaining.append("#SBATCH --mail-type=END,FAIL\n")
    remaining.append(f"#SBATCH --chdir={PROJECT_ROOT}\n")
    remaining.append("#SBATCH --gres=gpu:H100:1\n")
    remaining.append("#SBATCH --cpus-per-task=12\n")
    remaining.append("#SBATCH --mem=128G\n")
    # remaining.append("module load GCC/13.3.0\n")
    remaining.append("module load CUDA/12.1.1\n")
    remaining.append("module load Python/3.10.4-GCCcore-11.3.0\n")
    remaining.append("hostname\npwd\nwhich python3\npython3 --version\n\n")

    # f.write(f"#SBATCH --time={time_string}\n")
    # f.write("#SBATCH --mail-user=moezdurrani@ou.edu\n")
    # f.write("#SBATCH --mail-type=END,FAIL\n")
    # f.write(f"#SBATCH --chdir={PROJECT_ROOT}\n\n")
    # f.write("#SBATCH --cpus-per-task=1\n")

    # f.write("module purge\n")
    # f.write("module load GCC/13.3.0\n")
    # f.write("module load Python/3.10.4-GCCcore-11.3.0\n")
        # f.write("module load CUDA/12.1.1\n")
        # f.write("source ~/pyenv/bin/activate\n\n")

    # f.write("hostname\npwd\nwhich python3\npython3 --version\n\n")
    return remaining

def write_remaining_header(tot_exp_time, remaining, sb_write):
    tot_exp_time = (tot_exp_time * 1.2) + 600
    hrs = tot_exp_time // 3600
    rem_secs = tot_exp_time % 3600
    mins = rem_secs // 60
    secs = rem_secs % 60
    time_string = f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"
    sb_write.write(f"#SBATCH --time={time_string}\n")
    for lines in remaining:
        sb_write.write(lines)


# def create_sbatch(experiments_dir, data_dir, fname, dataset):
def create_sbatch(k,dataset, k_start, k_end, sample):
    """Creates one sbatch file for each k value with its 30 samples"""
    if sample == -1:
        sbatch_path = f"{SBATCH_DIR}/{dataset}_k{k}.sbatch"
    else:
        sbatch_path = f"{SBATCH_DIR}/{dataset}_k{k}_s{sample}.sbatch"

    sb_write = open(sbatch_path, "w")
    remaining = write_sbatch_header(sb_write, dataset, k, k_start, k_end, sample)    
    return sbatch_path, sb_write, remaining

def specific(k_start, k_end, samples, dataset, runf):
    data_dir = os.path.join(DATA_ROOT, dataset)
    if not os.path.isdir(data_dir):
        return
    experiments_dir = os.path.join(EXPERIMENT_ROOT, dataset)
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Create seperate directory for each data set for logs
    os.makedirs(os.path.join(LOG_DIR, dataset), exist_ok=True)

    existing_data = set(os.listdir(data_dir))
    existing_experiments = set(os.listdir(experiments_dir))

    sbatch_created = False

    for k in range(k_start, k_end+1):
        # if k is less than lower threshold
        if k < LOW_K_THRESHOLD:
            if not sbatch_created:
                sbatch_path, sb_write, remaining = create_sbatch(k, dataset, k_start, k_end, sample=-1)
                sbatch_created = True
                tot_exp_time = 0
            for s in range(1, samples+1):
                data_name, exp_name = f"k{k}_s{s}.txt", f"k{k}_s{s}.json"
                if data_name in existing_data and exp_name not in existing_experiments:
                    tot_exp_time += RUN_TIMES[k]
                    remaining.append(f"python3 {PROJECT_ROOT}/run.py {data_dir}/{data_name} {experiments_dir}/{exp_name}\n")
                
        # if k is between lower and high threshold
        elif k < HIGH_K_THRESHOLD:
            if sbatch_created:
                write_remaining_header(tot_exp_time, remaining, sb_write)
                sb_write.close()
                runf.write(f"sbatch {sbatch_path}\n")
                sbatch_created = False
            sbatch_path, sb_write, remaining = create_sbatch(k, dataset, k_start, k_end, sample=-1)
            tot_exp_time = 0
            for s in range(1, samples+1):
                data_name, exp_name = f"k{k}_s{s}.txt", f"k{k}_s{s}.json"
                if data_name in existing_data and exp_name not in existing_experiments:
                    tot_exp_time += RUN_TIMES[k]
                    remaining.append(f"python3 {PROJECT_ROOT}/run.py {data_dir}/{data_name} {experiments_dir}/{exp_name}\n")
            write_remaining_header(tot_exp_time, remaining, sb_write)
            sb_write.close()
            runf.write(f"sbatch {sbatch_path}\n")

        # if k is greater than high threshold
        else:
            if sbatch_created:
                write_remaining_header(tot_exp_time, remaining, sb_write)
                sb_write.close()
                runf.write(f"sbatch {sbatch_path}\n")
                sbatch_created = False
            for s in range(1, samples+1):
                data_name, exp_name = f"k{k}_s{s}.txt", f"k{k}_s{s}.json"
                if data_name in existing_data and exp_name not in existing_experiments:
                    sbatch_path, sb_write, remaining = create_sbatch(k, dataset, k_start, k_end, s)
                    tot_exp_time = RUN_TIMES[k]
                    remaining.append(f"python3 {PROJECT_ROOT}/run.py {data_dir}/{data_name} {experiments_dir}/{exp_name}\n")
                    write_remaining_header(tot_exp_time, remaining, sb_write)
                    sb_write.close()
                    runf.write(f"sbatch {sbatch_path}\n")

    if sbatch_created:
        write_remaining_header(tot_exp_time, remaining, sb_write)
        sb_write.close()
        runf.write(f"sbatch {sbatch_path}\n")

            

def parent(k_start, k_end, samples, runf):
    for dataset in sorted(os.listdir(DATA_ROOT)):
        specific(k_start, k_end, samples, dataset, runf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k_start", type=int, help="Start value of K")
    parser.add_argument("k_end", type=int, help="End value of K")
    parser.add_argument("samples", type=int, help="Number of samples per k value")
    parser.add_argument("--dataset", type=str, help="Specific dataset to run Kronfit on")
    args = parser.parse_args()

    # Make sure the directories for sbatches and logs exist
    os.makedirs(SBATCH_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create Shell Script to run everything
    runfile = os.path.join(PROJECT_ROOT, "run_all_batches.sh")
    with open(runfile, "w") as runf:
        runf.write('#!/bin/bash\n\n')

        if args.dataset:
            dataset = args.dataset
            specific(args.k_start, args.k_end, args.samples, args.dataset, runf)
        else:
            parent(args.k_start, args.k_end, args.samples, runf)
    print("Sbatch file generation complete")

if __name__ == "__main__":
    main()
