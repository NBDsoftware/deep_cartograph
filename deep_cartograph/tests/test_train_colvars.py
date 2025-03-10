from deep_cartograph.tools.train_colvars import train_colvars
import importlib.resources as resources
from deep_cartograph import tests
import pandas as pd
import shutil
import yaml
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def get_config():
    yaml_content = """
  cvs: ['pca', 'deep_tica', 'htica', 'tica', 'ae']
  common:
    dimension: 2
    num_subspaces: 10
    subspaces_dimension: 5
    input_colvars: 
      start: 0
      stop: null
      stride: 1
    architecture:
      hidden_layers: [5, 3]
      lag_time: 1                        
    training: 
      general:
        max_tries: 10
        seed: 42
        lengths: [0.8, 0.2]
        batch_size: 256
        max_epochs: 1000
        dropout: 0.1
        shuffle: False
        random_split: True
        check_val_every_n_epoch: 1
        save_check_every_n_epoch: 1
      early_stopping:
        patience: 20
        min_delta: 1.0e-05
      optimizer:
        name: Adam
        kwargs: 
          lr: 1.0e-02 
          weight_decay: 0
      lr_scheduler: null
      save_loss: True
      plot_loss: True
  ae:           
    architecture:
      hidden_layers: [5, 3]
    training:
      general:
        batch_size: 256
        max_epochs: 10000
        dropout: 0.1
      early_stopping:
        patience: 100
        min_delta: 1.0e-05
      optimizer:
        kwargs: 
          lr: 1.0e-04
          weight_decay: 0
  figures:
    fes:
      compute: True  
      save: True  
      temperature: 300
      bandwidth: 0.025
      num_bins: 200
      num_blocks: 1
      max_fes: 18
    traj_projection:
      plot: True
      num_bins: 100
      bandwidth: 0.25
      alpha: 0.6
      cmap: turbo
      use_legend: True
      marker_size: 12
  clustering:                        
    run: True                        
    algorithm: hierarchical               
    opt_num_clusters: True          
    search_interval: [5, 15]          
    num_clusters: 3                  
    linkage: complete                
    n_init: 20                       
    min_cluster_size: 50             
    min_samples: 5                  
    cluster_selection_epsilon: 0.5
    """
    return yaml.safe_load(yaml_content)


def test_train_colvars():
    
    print("Testing train_colvars...")
    
    # Inputs and reference files
    input_path = os.path.join(data_path, "input")
    trajectory_folder = os.path.join(input_path, "trajectory")
    topology_folder = os.path.join(input_path, "topology")
    trajectory_path = os.path.join(trajectory_folder, "CA_example.dcd")
    topology_path = os.path.join(topology_folder, "CA_example.pdb")
    colvars_path = os.path.join(data_path, "reference", "compute_features", "virtual_dihedrals.dat")
    filtered_features_path = os.path.join(data_path, "reference", "filter_features", "filtered_virtual_dihedrals.txt")
    
    # Output files
    output_path = os.path.join(tests_path, "output_train_colvars")
    
    # Read the filtered features into a list
    with open(filtered_features_path, 'r') as f:
        filtered_features = f.readlines()
        
    # Remove the newline characters
    filtered_features = [line.strip() for line in filtered_features]

    # Remove output folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    # Call API
    train_colvars(
        configuration = get_config(),
        colvars_paths = [colvars_path],
        feature_constraints = filtered_features,  
        output_folder = output_path,
        trajectories = [trajectory_path],
        topologies = [topology_path])
    
    test_passed = True
    for cv in get_config()['cvs']:
      
        print(f"Testing {cv}...")
        
        # Path to projected trajectory
        projected_trajectory_path = os.path.join(output_path, cv, "CA_example", "projected_trajectory.csv")
        
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
    
    assert test_passed
    
    # If the test passed, clean the output folder
    if test_passed:
      try:
        shutil.rmtree(output_path)
      except:
        print("Could not remove output folder.")