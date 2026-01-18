#!/bin/bash
#SBATCH --job-name=extract_parts   
#SBATCH --ntasks=2                        # Number of tasks 
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --time=02:00:00   
#SBATCH --partition=short    
#SBATCH --qos=short          
#SBATCH --mem-per-cpu=2000      # 1 Gb of RAM per cpu
#SBATCH --output=report_%j.out   # Name of file with standard output
#SBATCH --error=report_%j.err    # Name of file with standard input

# Purge loaded modules - erase all previously loaded modules
module purge

# Load conda (Miniconda is the lightweight version)
ml Miniconda3

# Activate previously created conda environment from environment.yml
source activate /home/pnavarro/.conda/envs/mdtraj

TRAJECTORY_IN=GOdMD_v1_chimera/trajs/GOdMD_6IRS_7DSQ_chimeric_890_symmetric_smoothed.dcd
TOPOLOGY_IN=GOdMD_v1_chimera/tops/GOdMD_6IRS_7DSQ_chimeric_890_symmetric_smoothed.pdb

TRAJ_1=GOdMD_v1_parts/trajs/GOdMD_6IRS_7DSQ_stateA_430.dcd
TRAJ_2=GOdMD_v1_parts/trajs/GOdMD_6IRS_7DSQ_transition.dcd
TRAJ_3=GOdMD_v1_parts/trajs/GOdMD_6IRS_7DSQ_stateB_430.dcd

TRAJECTORY_OUT=GOdMD_v1_chimera/trajs/GOdMD_6IRS_7DSQ_chimeric_890_symmetric_smoothed_clean.dcd

PATH_TO_PYTHON=/home/pnavarro/.conda/envs/mdtraj/bin
export PYTHONPATH=$PATH_TO_PYTHON

mdconvert -o $TRAJECTORY_OUT -t $TOPOLOGY_IN $TRAJECTORY_IN -i 0:889