from deep_cartograph.tools.traj_projection import traj_projection
import importlib.resources as resources
from deep_cartograph import tests
from pathlib import Path
import pandas as pd
import shutil
import yaml
import glob
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def test_traj_projection():

    print("Testing traj_projection...")

    # Inputs and reference files
    input_path = os.path.join(data_path, "input")
    models_folder = os.path.join(input_path, "models")
    model_paths = glob.glob(os.path.join(models_folder, "*.zip"))
    topology_path = os.path.join(input_path, "topology", "CA_example.pdb")
    colvars_path = os.path.join(data_path, "reference", "compute_features", "virtual_dihedrals.dat")
    
    # Output files
    output_path = os.path.join(tests_path, "output_traj_projection")

    # Remove output folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    # Call API
    projected_cvs_data = traj_projection(
                          configuration = {},
                          colvars_paths = [colvars_path],
                          topologies = [topology_path],
                          trajectory_names = ["CA_example"],
                          model_paths = model_paths,
                          output_folder = output_path
                          )
    
    test_passed = True
    for cv in projected_cvs_data.keys():
      
        print(f"Testing {cv}...")
        
        # Path to projected trajectory
        projected_trajectory_path = projected_cvs_data[cv]['traj_paths'][0]
        
        # Path to the reference projected trajectory
        reference_projected_trajectory_path = os.path.join(data_path, "reference", "train_colvars", f"{cv}_projected_trajectory.csv")
        
        # Check if the projected trajectory file exists
        if not os.path.exists(projected_trajectory_path):
            raise FileNotFoundError(f"Projected trajectory file {projected_trajectory_path} does not exist.")
        
        # Read the projected trajectory as pandas dataframe
        projected_trajectory_df = pd.read_csv(projected_trajectory_path)
        
        # Read the reference projected trajectory as pandas dataframe
        reference_projected_trajectory_df = pd.read_csv(reference_projected_trajectory_path)
        
        # Check if the computed and reference dataframes are equal
        test_passed = projected_trajectory_df.equals(reference_projected_trajectory_df) and test_passed

        if not test_passed:
            print(f"Test for {cv} failed.")
            break
        else:
            print(f"Test for {cv} passed.")
            
    assert test_passed
    
    # If the test passed, clean the output folder
    if test_passed:
      try:
        shutil.rmtree(output_path)
      except:
        print("Could not remove output folder.")