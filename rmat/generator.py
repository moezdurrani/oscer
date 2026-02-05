import os
import sys
import argparse

# Configurable Paths
PROJECT_ROOT = os.path.expanduser("~/rmat")
SBATCH_DIR = os.path.join(PROJECT_ROOT, "sbatches")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_ROOT = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/data"

PARTITION = "32gb_20core"
SEED = 42

# K val : Total Run time for one sample (seconds)
RUN_TIMES = {1:0.01, 2:0.03, 3:0.05, 4:0.22, 5:0.22, 6:0.22, 7:0.22, 8:0.23, 9:0.24, 10:0.24, 11:0.24, 12:0.24, 13:0.24, 14:0.25, 15:0.26, 16:0.29, 17:0.34, 18:0.46, 19:0.79, 20:1.41, 21:2.79, 22:5.61, 23:12.12, 24:26.19, 25:56.52, 26:120.6, 27:265.34, 28:605.03, 29:1468.99}

def file_exists(dataset, k, s):
    return os.path.exists(f"{DATA_ROOT}/{dataset}/k{k}_s{s}.txt")

def write_sbatch_header(f, dataset, k, time_string):
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --partition={PARTITION}\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write(f"#SBATCH --output={LOG_DIR}/{dataset}/k_{k}_%J_out.txt\n")
    f.write(f"#SBATCH --error={LOG_DIR}/{dataset}/k_{k}_%J_err.txt\n")
    f.write(f"#SBATCH --time={time_string}\n")
    f.write(f"#SBATCH --job-name={dataset}_k_{k}\n")
    f.write("#SBATCH --mail-user=moezdurrani@ou.edu\n")
    f.write("#SBATCH --mail-type=END,FAIL\n")
    f.write(f"#SBATCH --chdir={PROJECT_ROOT}\n\n")

    f.write("module purge\n")
    f.write("module load Python/3.10.4-GCCcore-11.3.0\n")
    f.write("module load CUDA/12.1.1\n")
    # f.write("source ~/pyenv/bin/activate\n\n")

    f.write("hostname\npwd\nwhich python3\npython3 --version\n\n")

def create_sbatch(dataset, a, b, c, d, k, samples):
	"""Creates one sbatch file for each k value with its 30 samples"""
	sbatch_path = f"{SBATCH_DIR}/{dataset}_k_{k}.sbatch"
	outdir = f"{DATA_ROOT}/{dataset}"

	with open(sbatch_path, "w") as f:
		tot_exp_time = (RUN_TIMES[k]  * samples * 3) + 600 # Buffer time + expected time for the job to run
		hrs = tot_exp_time // 3600
		rem_secs = tot_exp_time % 3600
		mins = rem_secs // 60
		secs = rem_secs % 60
		time_string = f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"
		write_sbatch_header(f, dataset, k, time_string)

		for s in range(1, samples + 1):
			if not file_exists(dataset, k, s):
				f.write(
					f"python3 rmat_gen.py {a} {b} {c} {d} {k} {s} {SEED} {outdir}\n"
				)

	return sbatch_path


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("params_file", type=str, help="File where list of datasets names and their parameters are stored")
	parser.add_argument("k_initial", type=int, help="Starting value of K")
	parser.add_argument("k_end", type=int, help="End value of K")
	parser.add_argument("samples", type=int, help="Number of Samples per K value")
	args = parser.parse_args()

	# Make sure the directories for sbatches and logs exist
	os.makedirs(SBATCH_DIR, exist_ok=True)
	os.makedirs(LOG_DIR, exist_ok=True)

	params_file = args.params_file
	k_initial = args.k_initial
	k_end = args.k_end
	samples = args.samples

	# Create Shell Script to run everything
	runfile = os.path.join(PROJECT_ROOT, "run_all_batches.sh")
	with open(runfile, "w") as runf:
		runf.write('#!/bin/bash\n\n')

		# Go line by line inside params file and create batches for diff k values
		with open(params_file, "r") as pf:
			for line in pf:

				# If STOP file exists, stop execution
				if os.path.exists(os.path.join(PROJECT_ROOT, "STOP")):
					print("STOP file detected. Exiting batch generation.")
					return

				line = line.strip()
				if not line or line.startswith('#'):
					continue

				dataset, a, b, c, d = line.split()
				a = float(a)
				b = float(b)
				c = float(c)
				d = float(d)

				# Create seperate directory for each data set
				os.makedirs(os.path.join(LOG_DIR, dataset), exist_ok=True)
				os.makedirs(f"{DATA_ROOT}/{dataset}", exist_ok=True)

				for k in range(k_initial, k_end+1):

					# If STOP file exists, stop execution
					if os.path.exists(os.path.join(PROJECT_ROOT, "STOP")):
						print("STOP file detected. Exiting batch generation.")
						return

					# Check if at least one file is missing
					missing = any(
						not file_exists(dataset, k, s)
						for s in range(1, samples+1)
					)

					if not missing:
						print(f"SKIP: {dataset}/k:{k} already complete")
						continue

					sbatch_path = create_sbatch(dataset, a, b, c, d, k, samples)
					runf.write(f"sbatch {sbatch_path}\n")
					print(f"[CREATE] {sbatch_path}")

	print("Sbatch file generation complete")

if __name__ == "__main__":
	main()
