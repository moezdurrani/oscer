#!/bin/bash
#SBATCH --job-name=mfit_batch          # Job name
#SBATCH --output=./mfit_%j.out      # Standard output log (%j = Job ID)
#SBATCH --error=./mfit_%j.err       # Standard error log
#SBATCH --partition=gpu                # Request a GPU partition
#SBATCH --gres=gpu:1                   # Request 1 GPU (essential for PyTorch)
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # CPU cores for data loading/Numba
#SBATCH --mem=32G                      # Memory (adjust based on graph size)
#SBATCH --time=00:00:00                # Time limit (hrs:min:sec)
#SBATCH --account=your_project_name    # Your cluster project/account name

# 1. Load necessary modules (Depends on your cluster's setup)
module load cuda/11.8                  # Match this to your PyTorch version
module load anaconda3                  # Or miniconda

# 2. Activate your environment
# This environment must have: torch, networkx, numba, tqdm, numpy
source activate your_env_name          

# 3. Create the output directory structure before running
mkdir -p outputs logs

# 4. Run your python script
# You can pass arguments here if your script uses argparse
python run.py
