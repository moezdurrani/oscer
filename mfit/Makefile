k_start ?= 1
k_end   ?= 23
samples ?= 30

# Optional: run only one dataset, example: make run dataset=my_dataset
dataset ?=

# Files / dirs
SBATCHDIR = sbatches
LOGDIR    = logs

.PHONY: all prepare_dirs generate_batches run_batches clean

# Default target
all: prepare_dirs generate_batches run_batches

prepare_dirs:
	mkdir -p $(SBATCHDIR)
	mkdir -p $(LOGDIR)

generate_batches:
ifeq ($(strip $(dataset)),)
	python3 generator.py $(k_start) $(k_end) $(samples)
else
	python3 generator.py $(k_start) $(k_end) $(samples) --dataset $(dataset)
endif
	chmod +x run_all_batches.sh

run_batches:
	bash run_all_batches.sh

clean:
	rm -f $(SBATCHDIR)/*.sbatch
	rm -f run_all_batches.sh
#	rm -rf $(LOGDIR)/*
