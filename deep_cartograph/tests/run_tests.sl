#!/bin/bash
#SBATCH --job-name=DeepCarto-tests
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# 
#SBATCH --output=Tests_%j.out
#SBATCH --error=Tests_%j.err
#SBATCH --time=02:00:00
## SBATCH --partition=short
## SBATCH --qos=short

module purge

ml Mamba
source activate /home/pnavarro/.conda/envs/deep_cartograph

pytest 