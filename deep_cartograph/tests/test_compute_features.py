from deep_cartograph.tools.compute_features import compute_features
from deep_cartograph.modules.plumed.utils import read_as_pandas
import importlib.resources as resources
from deep_cartograph import tests
import shutil
import yaml
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def get_config_virtual_dihedrals():
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
  
  
def get_config_distances():
    yaml_content = """
    plumed_settings:
      traj_stride: 1
      moltype: protein
      input_name: distances
      whole_molecule_selection: all
      features:
        distance_groups:
          dist:
            first_selection: "all"
            second_selection: "all"
            first_stride: 1
            second_stride: 10
            skip_neigh_residues: False
            skip_bonded_atoms: True
            atoms_format: name
    """
    return yaml.safe_load(yaml_content)


def test_compute_features():
    
    print("Testing compute_features ...")
    
    # Inputs and reference files
    input_path = os.path.join(data_path, "input")
    trajectory_path = os.path.join(input_path, "CA_trajectory.dcd")
    topology_path = os.path.join(input_path, "CA_topology.pdb")
    reference_colvars_path = os.path.join(data_path, "reference", "compute_features", "virtual_dihedrals.dat")
    
    # Output files
    output_path = os.path.join(tests_path, "output_compute_features")
    
    print("Testing compute_features with virtual dihedrals ...")
    
    # Check input files exist
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file {trajectory_path} does not exist.")
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topology file {topology_path} does not exist.")
    if not os.path.exists(reference_colvars_path):
        raise FileNotFoundError(f"Reference colvars file {reference_colvars_path} does not exist.")
    
    # Call API
    colvars_path = compute_features(
                    configuration=get_config_virtual_dihedrals(),
                    trajectory=trajectory_path,
                    topology=topology_path,
                    output_folder=output_path)
    
    # Read colvars file as pandas dataframe
    computed_df = read_as_pandas(colvars_path)
    
    # Read reference file as pandas dataframe
    reference_df = read_as_pandas(reference_colvars_path)
    
    # Check if the computed and reference dataframes are equal
    assert computed_df.equals(reference_df)
    
    # If the test passed, clean the output folder
    shutil.rmtree(output_path)
    
    print("Testing compute_features with distances ...")
    
    # Inputs and reference files
    reference_colvars_path = os.path.join(data_path, "reference", "compute_features", "distances.dat")
    
    # Call API
    colvars_path = compute_features(
                    configuration=get_config_distances(),
                    trajectory=trajectory_path,
                    topology=topology_path,
                    output_folder=output_path)
    
    # Read colvars file as pandas dataframe
    computed_df = read_as_pandas(colvars_path)
    
    # Read reference file as pandas dataframe
    reference_df = read_as_pandas(reference_colvars_path)
    
    # Check if the computed and reference dataframes are equal
    assert computed_df.equals(reference_df)
    
    # If the test passed, clean the output folder
    shutil.rmtree(output_path)
    