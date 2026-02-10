import argparse
import os

DATA_ROOT = "/ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/experiments/kronfit"

def specific(k_start, k_end, samples, dataset):
	directory = os.path.join(DATA_ROOT, dataset)
	if not os.path.isdir(directory):
		return
	existing = set(os.listdir(directory))
	print("Missing Files in {} directory".format(directory))
	i=0
	for k in range(k_start, k_end+1):
		for s in range(1, samples+1):
			fname = f"k{k}_s{s}.txt"
			if fname not in existing:
				i += 1
				print(fname)
	print("Missing {} files in {} directory\n\n".format(i, directory))

def parent(k_start, k_end, samples):
	print("Missing Files")
	for dataset in sorted(os.listdir(DATA_ROOT)):
		specific(k_start, k_end, samples, dataset)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("k_start", type=int, help="Start Value of K")
	parser.add_argument("k_end", type=int, help="End Value of K")
	parser.add_argument("samples", type=int, help="Number of Samples per K Value")	
	parser.add_argument("--dataset", type=str, help="Specific directory you want to look at") 
	args = parser.parse_args()

	# Specific Directory
	if args.dataset:
		specific(args.k_start, args.k_end, args.samples, args.dataset)
	# Parent Directory
	else:
		parent(args.k_start, args.k_end, args.samples)

if __name__ == "__main__":
	main()