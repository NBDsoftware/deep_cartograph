import deep_cartograph
from deep_cartograph.tools.compute_features import compute_features
from deep_cartograph.modules.plumed.utils import read_as_pandas
import importlib.resources as resources
from deep_cartograph import tests
import pytest 
import shutil
import yaml
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def get_config_a():
    yaml_content = """
    plumed_settings:
      traj_stride: 1
      moltype: protein
      input_name: torsions
      whole_molecule_selection: all
      features:
        dihedral_groups:
          tor:
            selection: "all"
            periodic_encoding: True
            search_mode: virtual
            atoms_format: name
    """
    return yaml.safe_load(yaml_content)


def test_compute_features():
    
    print("Testing compute_features ...")
    
    configuration = get_config_a()
        
    calpha_path = os.path.join(data_path, "calpha_traj")
    trajectory_path = os.path.join(calpha_path, "trajectory.dcd")
    topology_path = os.path.join(calpha_path, "topology.pdb")
    output_path = os.path.join(tests_path, "output_compute_features")
    
    # Check input files exist
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file {trajectory_path} does not exist.")
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topology file {topology_path} does not exist.")
        
    colvars_path = compute_features(
                    configuration=configuration,
                    trajectory=trajectory_path,
                    topology=topology_path,
                    output_folder=output_path)
    
    # Read colvars file as pandas dataframe
    colvars_df = read_as_pandas(colvars_path)
    
    # Print rows 1, 
    
    # Clean output folder
    #shutil.rmtree(output_path)
    
    assert True