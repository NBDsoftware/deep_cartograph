from deep_cartograph.run import deep_cartograph
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
    compute_features:
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

    filter_features:
        filter_settings:
            compute_diptest: True              
            compute_entropy: False             
            compute_std: False                
            diptest_significance_level: 0.05   
            entropy_quantile: 0                
            std_quantile: 0                      
        sampling_settings:
            relaxation_time: 1  
          
    train_colvars:
      cvs: [ 'pca', 'tica', 'deep_tica', 'htica', 'ae'] 
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


def test_deep_cartograph():
    
    print("Testing deep_cartograph ...")
    
    # Inputs and reference files
    input_path = os.path.join(data_path, "input")
    trajectory_folder = os.path.join(input_path, "trajectory")
    topology_folder = os.path.join(input_path, "topology")
    reference_path = os.path.join(data_path, "reference", "train_colvars")
    
    # Output files
    output_path = os.path.join(tests_path, "output_deep_cartograph")
    
    # Check input folders
    if not os.path.exists(trajectory_folder):
        raise FileNotFoundError(f"Trajectory folder not found: {trajectory_folder}")
    if not os.path.exists(topology_folder):
        raise FileNotFoundError(f"Topology folder not found: {topology_folder}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference folder not found: {reference_path}")
            
    # Remove output folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # Call API
    deep_cartograph(configuration=get_config(),
                    trajectory_data=trajectory_folder,
                    topology_data=topology_folder,
                    ref_trajectory_data=trajectory_folder,
                    ref_topology_data=topology_folder,
                    output_folder=output_path)
    
    # Find path to train_colvars step
    train_colvars_path = os.path.join(output_path, "train_colvars")
    
    # For each CV, check if the computed and reference colvars files are equal
    test_passed = True
    for cv in get_config()['train_colvars']['cvs']:
      
      # Find paths to csv files
      reference_projection_path = os.path.join(reference_path, f"{cv}_projected_trajectory.csv")
      computed_projection_path = os.path.join(train_colvars_path, cv, "CA_example", "projected_trajectory.csv")
      computed_ref_data_path = os.path.join(train_colvars_path, cv, "reference_data", "colvars.csv")
      
      # Read csv files
      reference_df = pd.read_csv(reference_projection_path)
      computed_df = pd.read_csv(computed_projection_path)
      ref_data_df = pd.read_csv(computed_ref_data_path)
      
      # Compare them
      test_passed = test_passed and computed_df.equals(reference_df)
      for col in ref_data_df.columns:
        test_passed = test_passed and ref_data_df[col].equals(reference_df[col])
        
      print(f"{cv} test passed: {computed_df.equals(reference_df)}")

    assert test_passed

    # If the test passed, clean the output folder
    if test_passed:
      shutil.rmtree(output_path)