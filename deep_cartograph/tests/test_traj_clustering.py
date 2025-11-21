from deep_cartograph.tools.traj_cluster import traj_cluster
import importlib.resources as resources
from deep_cartograph import tests
import pandas as pd
import shutil
import glob
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def test_traj_cluster():

    print("Testing traj_cluster...")

    # Inputs and reference files
    input_path = os.path.join(data_path, "input")
    cv_trajs_folder = os.path.join(input_path, "train_colvars")
    cv_traj_paths = sorted(glob.glob(os.path.join(cv_trajs_folder, "*.csv")))
    ref_cv_trajs_folder = os.path.join(data_path, "reference", "traj_cluster")
    ref_cv_traj_paths = sorted(glob.glob(os.path.join(ref_cv_trajs_folder, "*.csv")))

    for cv_index in range(len(cv_traj_paths)):
        
        cv_traj_path = cv_traj_paths[cv_index]
        ref_cv_traj_path = ref_cv_traj_paths[cv_index]
        
        file_name = os.path.basename(cv_traj_path)
        cv_name = file_name.replace("_projected_trajectory.csv", "")
        
        print(f"Clustering trajectory for CV: {cv_name}")
        
        # Output files
        output_path = os.path.join(tests_path, f"output_traj_cluster_{cv_name}")
        
        # Remove output folder if it exists
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # Call API
        clustered_traj_data = traj_cluster(
                        configuration = {},
                        cv_traj_paths = cv_traj_path,
                        output_folder = output_path
                        )

        test_passed = True
        for traj in clustered_traj_data.keys():
            
            # Path to clustered trajectory
            clustered_trajectory_path = clustered_traj_data[traj][0]
            # Check if the clustered trajectory file exists
            if not os.path.exists(clustered_trajectory_path):
                raise FileNotFoundError(f"Clustered trajectory file {clustered_trajectory_path} does not exist.")

            # Read the clustered trajectory
            clustered_trajectory_df = pd.read_csv(clustered_trajectory_path)

            # Read the reference clustered trajectory
            reference_clustered_trajectory_df = pd.read_csv(ref_cv_traj_path)

            # Check if the computed and reference dataframes are equal
            test_passed = clustered_trajectory_df.equals(reference_clustered_trajectory_df) and test_passed

            if not test_passed:
                print(f"Test for {cv_name} failed.")
                break
            else:
                print(f"Test for {cv_name} passed.")

    assert test_passed
    
    # If the test passed, clean the output folder
    if test_passed:
      try:
        shutil.rmtree(output_path)
      except:
        print("Could not remove output folder.")