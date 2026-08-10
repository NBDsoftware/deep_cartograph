from deep_cartograph.tools.train_colvars import train_colvars
import importlib.resources as resources
from deep_cartograph import tests
from pathlib import Path
import pandas as pd
import shutil
import yaml
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def get_config():
    yaml_content = """
    cvs: [ 'pca', 'tica', 'deep_tica', 'htica', 'ae', 'vae'] 
    common:
      dimension: 2
      lag_time: 1 
      features_normalization: mean_std
      architecture:
        encoder: 
          layers: [16, 8]
          activation: [leaky_relu, leaky_relu]
          batchnorm: [False, False]
          dropout: [0, 0]
        decoder: 
          layers: [4, 8]
          activation: [leaky_relu, leaky_relu]
          batchnorm: [False, False]
          dropout: [0, 0]
      num_subspaces: 10
      subspaces_dimension: 5
      input_colvars: 
        start: 0
        stop: null
        stride: 1                
      training: 
        general:
          num_tries: 1
          seed: 42
          lengths: [0.8, 0.2]
          batch_size: 256
          max_epochs: 1000  
          shuffle: False
          random_split: True
          check_val_every_n_epoch: 1
          save_check_every_n_epoch: 1
        early_stopping:
          patience: 20
          min_delta: 1.0e-05
        lr_scheduler: null
        lr_scheduler_config: null
        optimizer:
          name: Adam
          kwargs: 
            lr: 1.0e-03
            weight_decay: 0
        save_loss: True
        plot_loss: True
        model_to_save: last
        kl_annealing:
          type: linear
          start_beta: 0
          max_beta: 0.001
          start_epoch: 1000
          n_epochs_anneal: 5000
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
        marker_size: 12
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
    trained_cvs_data = train_colvars(
                        configuration = get_config(),
                        train_colvars_paths = [colvars_path],
                        train_topologies = [topology_path],
                        trajectory_names = [Path(trajectory_path).stem],
                        features_list = filtered_features,
                        output_folder = output_path
                        )
    
    test_passed = True
    for cv in get_config()['cvs']:
      
        print(f"Testing {cv}...")
        
        # Path to projected trajectory
        projected_trajectory_path = trained_cvs_data[cv]['traj_paths'][0]
        
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


def get_ensemble_config(num_models: int):
    """Small and fast configuration to train an ensemble of neural network CVs."""

    configuration = get_config()

    # Only a linear and a neural network CV, to check the ensemble applies to the latter only
    configuration['cvs'] = ['pca', 'ae']
    configuration['common']['training']['general']['num_models'] = num_models
    configuration['common']['training']['general']['max_epochs'] = 10
    configuration['figures']['fes']['compute'] = False
    configuration['figures']['fes']['save'] = False
    configuration['figures']['traj_projection']['plot'] = False

    return configuration


def split_colvars(colvars_path: str, output_folder: str, num_parts: int):
    """
    Splits a colvars file into num_parts files, so that several training trajectories are
    available to build ensemble folds from. Returns the paths and the trajectory names.
    """

    lines = open(colvars_path).read().splitlines()
    header, body = lines[0], lines[1:]
    part_size = len(body) // num_parts

    os.makedirs(output_folder, exist_ok=True)

    colvars_paths, trajectory_names = [], []
    for part_index in range(num_parts):
        # Last part takes the remainder
        if part_index == num_parts - 1:
            part = body[part_index * part_size:]
        else:
            part = body[part_index * part_size:(part_index + 1) * part_size]

        part_path = os.path.join(output_folder, f"traj_{part_index}.dat")
        with open(part_path, 'w') as part_file:
            part_file.write("\n".join([header] + part) + "\n")

        colvars_paths.append(part_path)
        trajectory_names.append(f"traj_{part_index}")

    return colvars_paths, trajectory_names


def test_train_colvars_ensemble():
    """
    Trains an ensemble of neural network CVs and checks the output layout: one folder per
    ensemble member, each with its own model and a projection of every training trajectory.
    """

    print("Testing train_colvars ensemble...")

    num_models = 3

    # Inputs
    input_path = os.path.join(data_path, "input")
    topology_path = os.path.join(input_path, "topology", "CA_example.pdb")
    colvars_path = os.path.join(data_path, "reference", "compute_features", "virtual_dihedrals.dat")
    filtered_features_path = os.path.join(data_path, "reference", "filter_features", "filtered_virtual_dihedrals.txt")

    with open(filtered_features_path, 'r') as f:
        filtered_features = [line.strip() for line in f.readlines()]

    # Output files
    output_path = os.path.join(tests_path, "output_train_colvars_ensemble")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Split the input colvars file to have several training trajectories to build folds from
    split_folder = os.path.join(output_path, "split_colvars")
    colvars_paths, trajectory_names = split_colvars(colvars_path, split_folder, num_models)

    # Call API
    trained_cvs_data = train_colvars(
                        configuration = get_ensemble_config(num_models),
                        train_colvars_paths = colvars_paths,
                        train_topologies = [topology_path] * num_models,
                        trajectory_names = trajectory_names,
                        features_list = filtered_features,
                        output_folder = output_path
                        )

    # The linear CV is not ensembled, it keeps the plain cv folder
    assert os.path.isdir(os.path.join(output_path, "pca"))
    assert not os.path.exists(os.path.join(output_path, "pca_0"))
    assert len(trained_cvs_data['pca']['ensemble_model_paths']) == 1

    # The neural network CV has one folder per ensemble member
    assert not os.path.exists(os.path.join(output_path, "ae"))
    assert len(trained_cvs_data['ae']['ensemble_model_paths']) == num_models

    projections = []
    for member_index in range(num_models):
        member_folder = os.path.join(output_path, f"ae_{member_index}")

        # Each member saved its own model
        assert os.path.isfile(os.path.join(member_folder, "model.zip"))

        # Each member projected every training trajectory, also the one it did not train on
        for trajectory_name in trajectory_names:
            projection_path = os.path.join(member_folder, "traj_data", trajectory_name, "projected_trajectory.csv")
            assert os.path.isfile(projection_path)

        projections.append(pd.read_csv(
            os.path.join(member_folder, "traj_data", trajectory_names[0], "projected_trajectory.csv")))

    # The returned paths point to the first member, and the whole ensemble is listed
    assert trained_cvs_data['ae']['model_path'] == os.path.join(output_path, "ae_0", "model.zip")
    assert len(trained_cvs_data['ae']['traj_paths']) == num_models

    # Members trained on different folds must be different models
    for member_index in range(1, num_models):
        assert not projections[0].equals(projections[member_index]), \
            f"Ensemble member {member_index} is identical to member 0"

    # Clean the output folder
    try:
        shutil.rmtree(output_path)
    except:
        print("Could not remove output folder.")