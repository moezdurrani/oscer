data ?= params.txt
#DATA_FILE = paramfiles/answers.txt

k_start ?= 1
k_end ?= 29
samples ?= 30

# Tells make these are commands, not filenames
.PHONY: all prepare_dirs generate_batches run_batches clean

# Define the default targets
all: prepare_dirs generate_batches run_batches

prepare_dirs:
	mkdir -p sbatches
	mkdir -p logs


generate_batches:
	python3 generator.py $(data) $(k_start) $(k_end) $(samples)
	chmod +x run_all_batches.sh


run_batches:
	bash run_all_batches.sh


clean:
	rm -rf sbatches/*.sbatch run_all_batches.sh
	rm -rf /ourdisk/hpc/soonerhpclab/dont_archive/moezdurrani/data/*/temp_k*_s*
