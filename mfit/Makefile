k_start ?= 1
k_end   ?= 23
s_start ?= 1
s_end ?= 15

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
	python3 generator.py $(k_start) $(k_end) $(s_start) $(s_end)
else
	python3 generator.py $(k_start) $(k_end) $(s_start) $(s_end) --dataset $(dataset)
endif
	chmod +x run_all_batches.sh

run_batches:
	bash run_all_batches.sh

clean:
	rm -f $(SBATCHDIR)/*.sbatch
	rm -f run_all_batches.sh
#	rm -rf $(LOGDIR)/*