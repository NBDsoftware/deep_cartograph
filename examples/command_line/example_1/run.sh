#!/bin/bash

conda activate deep_cartograph

TRAJ_PATH=$DEEPCARTO_PATH/tests/data/input/trajectory           # Trajectories should be PLUMED and MdAnalysis compatible (dcd or xtc for example)
TOPOLOGY_PATH=$DEEPCARTO_PATH/tests/data/input/topology         # Topology should be PLUMED and MdAnalysis compatible (pdb for example)
CONFIG_PATH=config.yml                      # Configuration file - see example in the repository
OUTPUT_PATH=output                          # Output path

deep_carto -conf $CONFIG_PATH -traj_data $TRAJ_PATH -top_data $TOPOLOGY_PATH -out $OUTPUT_PATH -v