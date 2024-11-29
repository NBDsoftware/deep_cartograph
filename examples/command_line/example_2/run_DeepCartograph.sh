#!/bin/bash

DEEPCARTO_PATH=/home/pnavarro/repos/NostrumBD/CV_learning/deep_cartograph/deep_cartograph

PARENT_INPUT_PATH=/home/pnavarro/repos/NostrumBD/CV_learning/deep_cartograph/examples/debug/compare_dynamics/input
TRAJ_DATA=$PARENT_INPUT_PATH/trajs
TOPOLOGY_DATA=$PARENT_INPUT_PATH/tops
CONFIG_PATH=config.yml                         

python $DEEPCARTO_PATH/run.py -conf $CONFIG_PATH  -traj_data $TRAJ_DATA -top_folder $TOPOLOGY_DATA -output output_1 -restart -v