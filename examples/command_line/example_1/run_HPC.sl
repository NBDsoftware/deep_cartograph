#!/bin/bash
#SBATCH --job-name=deep_carto
#SBATCH --ntasks-per-node=1
#SBATCH --output=DeepCarto_%j.out
#SBATCH --error=DeepCarto_%j.err
#SBATCH --time=02:00:00
#SBATCH --qos=standard-cpu
#SBATCH --partition=normal

module purge

ml Mamba
source activate /shared/work/pnavarro/envs/deep_cartograph

export PYTHONPATH=$PYTHONPATH:/shared/work/pnavarro/repos/deep_cartograph

DEEPCARTO_PATH=/shared/work/pnavarro/repos/deep_cartograph/deep_cartograph

TRAJ_PATH=$DEEPCARTO_PATH/tests/data/input/trajectory           # Trajectories should be PLUMED and MdAnalysis compatible (dcd or xtc for example)
TOPOLOGY_PATH=$DEEPCARTO_PATH/tests/data/input/topology         # Topology should be PLUMED and MdAnalysis compatible (pdb for example)
CONFIG_PATH=config.yml                      # Configuration file - see example in the repository
OUTPUT_PATH=output                          # Output path

python $DEEPCARTO_PATH/run.py -conf $CONFIG_PATH -traj_data $TRAJ_PATH -top_data $TOPOLOGY_PATH -out $OUTPUT_PATH -v