DEEPCARTO_PATH=/home/pnavarro/repos/NostrumBD/CV_learning/deep_cartograph/deep_cartograph

TRAJ_PATH=/home/pnavarro/repos/NostrumBD/CV_learning/deep_cartograph/deep_cartograph/tests/data/input/CA_trajectory.dcd    # Should be PLUMED and MdAnalysis compatible (dcd or xtc for example)
TOPOLOGY_PATH=/home/pnavarro/repos/NostrumBD/CV_learning/deep_cartograph/deep_cartograph/tests/data/input/CA_topology.pdb  # Should be PLUMED and MdAnalysis compatible (pdb for example)
CONFIG_PATH=config.yml                      # Configuration file - see example in the repository
OUTPUT_PATH=output                          # Output path

python $DEEPCARTO_PATH/run.py -conf $CONFIG_PATH -traj $TRAJ_PATH -top $TOPOLOGY_PATH -out $OUTPUT_PATH -v