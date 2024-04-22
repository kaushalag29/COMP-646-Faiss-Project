#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --partition=scavenge
#SBATCH --nodes=1                # node count
#SBATCH --reservation=comp646
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --export=ALL

ml purge
ml GCCcore/12.3.0 Python/3.11.3
python ImageFeaturesExtractor.py
