DEEPCARTO_PATH=/home/pnavarro/repos/NostrumBD/CV_learning/deep_cartograph/deep_cartograph

TRAJ_PATH=$DEEPCARTO_PATH/tests/data/input/trajectory           # Trajectories should be PLUMED and MdAnalysis compatible (dcd or xtc for example)
TOPOLOGY_PATH=$DEEPCARTO_PATH/tests/data/input/topology         # Topology should be PLUMED and MdAnalysis compatible (pdb for example)
CONFIG_PATH=config.yml                      # Configuration file - see example in the repository
OUTPUT_PATH=output                          # Output path

python $DEEPCARTO_PATH/run.py -conf $CONFIG_PATH -traj_data $TRAJ_PATH -top_data $TOPOLOGY_PATH -out $OUTPUT_PATH -v