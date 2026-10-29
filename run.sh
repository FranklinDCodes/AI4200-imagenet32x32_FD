#! /bin/bash

#SBATCH --account=bs-fgdoane42
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

source ../dl_env/bin/activate

python -u harness_run_script.py > "$1.out" 2>&1
