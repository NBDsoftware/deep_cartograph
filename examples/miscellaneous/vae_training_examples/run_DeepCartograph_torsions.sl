#!/bin/bash
#SBATCH --job-name=DeepCarto
#SBATCH --ntasks-per-node=2
#SBATCH --output=DeepCarto_%j.out
#SBATCH --error=DeepCarto_%j.err
#SBATCH --time=02:00:00

# Standard CPU settings
#SBATCH --partition=standard-cpu

# Short CPU settings
##SBATCH --partition=short
##SBATCH --qos=short

# GPU settings
##SBATCH --partition=gpu_short
##SBATCH --qos=gpu_short
##SBATCH --gres=gpu:1

module purge

ml Miniconda3
source activate /home/pnavarro/.conda/envs/deep_cartograph

TRAJ_PATH=../deepCarto_input/GOdMD_traj               # <------ Trajectories used for training
TOPOLOGY_PATH=../deepCarto_input/GOdMD_top            # <------ Corresponding topology files
SUP_TRAJ_PATH=../deepCarto_input/MDequilibration_traj # <------ Supplementary trajectories (not used for training, just projected onto the latent space) - useful to obtain the corresponding plumed input
SUP_TOP_PATH=../deepCarto_input/MDequilibration_top   # <------ Corresponding topology files for the supplementary trajectories
CONFIG_PATH=config_torsions.yml                       # Configuration file - see other examples in the repository and yaml_schemas for the default values

deep_carto -conf $CONFIG_PATH \
           -top_data $TOPOLOGY_PATH \
           -traj_data $TRAJ_PATH \
           -sup_traj_data $SUP_TRAJ_PATH \
           -sup_top_data $SUP_TOP_PATH \
           -out output_torsions \
           -restart