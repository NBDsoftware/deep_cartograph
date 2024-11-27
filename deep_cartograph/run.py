import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Union, Tuple

# Local imports
from deep_cartograph import compute_features
from deep_cartograph import filter_features
from deep_cartograph import train_colvars
from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration, validate_configuration, read_feature_constraints
from deep_cartograph.yaml_schemas.deep_cartograph import DeepCartograph

########
# TOOL #
########

def deep_cartograph(configuration: Dict, trajectory_folder: str, topology_folder: str, ref_trajectory_folder: Union[str, None] = None, 
                    ref_topology_folder: Union[str, None] = None, label_reference: bool = False, dimension: Union[int, None] = None, 
                    cvs: Union[List[Literal['pca', 'ae', 'tica', 'deep_tica']], None] = None, 
                    restart: bool = False, output_folder: Union[str, None] = None) -> None:
    """
    Function that maps a set of trajectories onto a set of collective variables.

    Parameters
    ----------

        configuration:         Configuration dictionary (see default_config.yml for more information)
        trajectory_folder:     Path to the folder with trajectories to compute the CVs. To compare the ensembles of different trajectories on the same CVs.
        topology_folder:       Path to the folder with topology files of the trajectories. Should have the same name as the corresponding trajectory in the trajectory folder.
        ref_trajectory_folder: (Optional) Path to the folder with reference data.
        ref_topology_folder:   (Optional) Path to the folder with topology files of the reference data. Should have the same name as the corresponding reference file in the reference folder.
        label_reference:       (Optional) Use labels for reference data (names of the files in the reference folder)
        dimension:             (Optional) Dimension of the collective variables to train or compute, overwrites the value in the configuration if provided
        cvs:                   (Optional) List of collective variables to train or compute ['pca', 'ae', 'tica', 'deep_tica'], overwrites the value in the configuration if provided
        output_folder:         (Optional) Path to the output folder, if not given, a folder named 'deep_cartograph' is created
    """

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Start timer
    start_time = time.time()

    # If output folder is not given, create default
    if not output_folder:
        output_folder = 'deep_cartograph'
    
    # If restart is False, create unique output folder
    if not restart:
        output_folder = get_unique_path(output_folder)
            
    # Create output folder if it does not exist
    create_output_folder(output_folder)

    # Validate configuration
    configuration = validate_configuration(configuration, DeepCartograph, output_folder)
    
    # Check main input folders
    trajectories, topologies = check_input_folders(trajectory_folder, topology_folder)
    
    # Check reference input folders
    ref_trajectories, ref_topologies = check_ref_folders(ref_trajectory_folder, ref_topology_folder)

    # Step 1: Compute features
    step1_parent_path = os.path.join(output_folder, 'compute_features')
    
    # Step 1.1: Compute features for trajectories
    traj_colvars_paths = []
    traj_names = []
    for trajectory, topology in zip(trajectories, topologies):
        
        # Create unique output folder
        traj_name = Path(trajectory).stem
        step1_output_folder = os.path.join(step1_parent_path, traj_name)
        colvars_path = os.path.join(step1_output_folder, 'colvars.dat')
        
        # Compute features for trajectory if colvars file does not exist
        if os.path.exists(colvars_path):
            logger.info(f"Colvars file already exists for trajectory {traj_name}. Skipping computation of features.")
        else:
            compute_features(
                configuration = configuration['compute_features'], 
                trajectory = trajectory, 
                topology = topology, 
                colvars_path = colvars_path,
                output_folder = step1_output_folder)
            
        # Save path to colvars file and name of trajectory file
        traj_colvars_paths.append(colvars_path)
        traj_names.append(traj_name)

    # Step 1.2: Compute features for reference data
    ref_colvars_paths = []
    ref_names = []    
    for ref_trajectory, ref_topology in zip(ref_trajectories, ref_topologies):

        # Create unique output folder
        ref_name = Path(ref_trajectory).stem
        step1_output_folder = os.path.join(step1_parent_path, ref_name)
        colvars_path = os.path.join(step1_output_folder, 'colvars.dat')

        if os.path.exists(ref_colvars_path):
            logger.info("Colvars file already exists for reference file. Skipping computation of features.")
        else:
            ref_colvars_path = compute_features(
                configuration = configuration['compute_features'], 
                trajectory = ref_trajectory,
                topology = ref_topology,
                colvars_path = colvars_path,
                output_folder = step1_output_folder)
        
        # Save path to colvars file and name of reference file
        ref_colvars_paths.append(colvars_path)
        ref_names.append(ref_name)

    if not label_reference:
        ref_names = None

    # Step 2: Filter features
    step2_output_folder = os.path.join(output_folder, 'filter_features')
    output_features_path = os.path.join(step2_output_folder, 'filtered_features.txt')
    
    if os.path.exists(output_features_path):
        logger.info("Filtered features file already exists. Skipping filtering of features.")
    else:
        output_features_path = filter_features(
                configuration = configuration['filter_features'], 
                colvars_paths = traj_colvars_paths,
                output_features_path = output_features_path,
                output_folder = step2_output_folder)

    # Read filtered features
    filtered_features = read_feature_constraints(output_features_path) 

    # Step 3: Train colvars
    step3_output_folder = os.path.join(output_folder, 'train_colvars')
    train_colvars(
        configuration = configuration['train_colvars'],
        colvars_paths = traj_colvars_paths,
        feature_constraints = filtered_features,
        ref_colvars_paths = ref_colvars_paths,
        ref_labels = ref_names,
        dimension = dimension,
        cvs = cvs,
        trajectories = trajectories,
        topologies = topologies,
        samples_per_frame = 1/configuration['compute_features']['plumed_settings']['traj_stride'],
        output_folder = step3_output_folder)
            
    # End timer
    elapsed_time = time.time() - start_time

    # Write time to log in hours, minutes and seconds
    logger.info('Total elapsed time: %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

def check_input_folders(trajectory_folder: str, top_folder: str) -> Tuple[List[str], List[str]]:
    """
    Function that checks if the necessary input files exist.
    
    Inputs
    ------
    
        trajectory_folder (str): Path to the folder with trajectories to compute the CVs. To compare the ensembles of different trajectories on the same CVs.
        top_folder        (str): Path to the folder with topology files of the trajectories. Should have the same name as the corresponding trajectory in the trajectory folder.
    """
    
    logger = logging.getLogger("deep_cartograph")
    
    # Check if trajectory folder exists
    if not os.path.exists(trajectory_folder):
        logger.error(f"Trajectory folder not found: {trajectory_folder}")
        sys.exit(1)
    
    # List the files in the trajectory folder
    traj_file_paths = [os.path.join(trajectory_folder, f) for f in os.listdir(trajectory_folder) if os.path.isfile(os.path.join(trajectory_folder, f))]
    
    # Sort them alphabetically
    traj_file_paths.sort()
    
    # Check if there are any
    if len(traj_file_paths) == 0:
        logger.error(f"Trajectory folder is empty: {trajectory_folder}")
        sys.exit(1)
    
    # Check if topology folder exists
    if not os.path.exists(top_folder):
        logger.error(f"Topology folder not found: {top_folder}")
        sys.exit(1) 

    # List the files in the topology folder
    top_file_paths = [os.path.join(top_folder, f) for f in os.listdir(top_folder) if os.path.isfile(os.path.join(top_folder, f))]
    
    # Sort them alphabetically
    top_file_paths.sort()
    
    # Check if there are any
    if len(top_file_paths) == 0:
        logger.error(f"Topology folder is empty: {top_folder}")
        sys.exit(1)
    
    # Check if we have the same number of topology files as trajectory files
    if len(traj_file_paths) != len(top_file_paths):
        logger.error(f"Number of topology files is different from the number of trajectory files ({len(top_file_paths)} vs {len(traj_file_paths)}).")
        sys.exit(1)
        
    # Check if each trajectory file has a corresponding topology file with the same name
    for traj_file, topology_file in zip(traj_file_paths, top_file_paths):
        
        # Find name of trajectory file
        traj_name = Path(traj_file).stem
        
        # Find name of topology file
        top_name = Path(topology_file).stem
        
        # Check if they have the same name
        if traj_name != top_name:
            logger.error(f"Trajectory file does not have a corresponding topology file with the same name: {traj_name}")
            sys.exit(1)
            
    return traj_file_paths, top_file_paths

def check_ref_folders(ref_trajectory_folder: Union[str, None], ref_topology_folder: Union[str, None]) -> Tuple[List[str], List[str]]:
    """
    Function that checks if the necessary input files exist.
    
    Inputs
    ------
    
        ref_trajectory_folder (str): Path to the folder with reference data.
        ref_topology_folder   (str): Path to the folder with topology files of the reference data. Should have the same name as the corresponding reference file in the reference folder.
    """      
    
    logger = logging.getLogger("deep_cartograph")
    
    # Check if no reference folder is given
    if ref_trajectory_folder is None or ref_topology_folder is None:
        logger.debug("No reference folder given.")
        return [], []
    
    # Check if reference trajectory folder exists
    if not os.path.exists(ref_trajectory_folder):
        logger.error("Reference trajectory folder not found: %s", ref_trajectory_folder)
        sys.exit(1)
    
    # Check if reference topology folder exists
    if not os.path.exists(ref_topology_folder):
        logger.error("Reference topology folder not found: %s", ref_topology_folder)
        sys.exit(1)
        
    # List the files in the reference trajectory folder
    ref_traj_paths = [os.path.join(ref_trajectory_folder, f) for f in os.listdir(ref_trajectory_folder) if os.path.isfile(os.path.join(ref_trajectory_folder, f))]
    
    # Check if there are any
    if len(ref_traj_paths) == 0:
        logger.error("Reference trajectory folder is empty: %s", ref_trajectory_folder)
        sys.exit(1)
    
    # List the files in the reference topology folder
    ref_top_paths = [os.path.join(ref_topology_folder, f) for f in os.listdir(ref_topology_folder) if os.path.isfile(os.path.join(ref_topology_folder, f))]
    
    # Check if there are any
    if len(ref_top_paths) == 0:
        logger.error("Reference topology folder is empty: %s", ref_topology_folder)
        sys.exit(1)
    
    # Check if we have the same number of reference topology files as reference trajectory files
    if len(ref_traj_paths) != len(ref_top_paths):
        logger.error(f"Number of reference topology files is different from the number of reference trajectory files ({len(ref_top_paths)} vs {len(ref_traj_paths)}).")
        sys.exit(1)
        
    # Check if each reference trajectory has a corresponding reference topology file
    for ref_traj_path in ref_traj_paths:
            
        # Find name of reference trajectory file
        ref_traj_name = Path(ref_traj_path).stem
        
        # Check if there is a corresponding reference topology file
        if not any(ref_traj_name in ref_top for ref_top in ref_top_paths):
            logger.error("Reference topology file not found for reference trajectory: %s", ref_traj_name)
            sys.exit(1)
            
    return ref_traj_paths, ref_top_paths
    
         
def set_logger(verbose: bool):
    """
    Function that sets the logging configuration. If verbose is True, it sets the logging level to DEBUG.
    If verbose is False, it sets the logging level to INFO.

    Inputs
    ------

        verbose (bool): If True, sets the logging level to DEBUG. If False, sets the logging level to INFO.
    """
    # Issue warning if logging is already configured
    if logging.getLogger().hasHandlers():
        logging.warning("Logging has already been configured in the root logger. This may lead to unexpected behavior.")
    
    # Get the path to this file
    file_path = Path(os.path.abspath(__file__))

    # Get the path to the package
    package_path = file_path.parent

    info_config_path = os.path.join(package_path, "log_config/info_configuration.ini")
    debug_config_path = os.path.join(package_path, "log_config/debug_configuration.ini")
    
    # Check the existence of the configuration files
    if not os.path.exists(info_config_path):
        raise FileNotFoundError(f"Configuration file not found: {info_config_path}")
    
    if not os.path.exists(debug_config_path):
        raise FileNotFoundError(f"Configuration file not found: {debug_config_path}")
    
    if verbose:
        logging.config.fileConfig(debug_config_path, disable_existing_loggers=True)
    else:
        logging.config.fileConfig(info_config_path, disable_existing_loggers=True)

    logger = logging.getLogger("deep_cartograph")

    logger.info("Deep Cartograph: package for projecting and clustering trajectories using collective variables.")

########
# MAIN #
########

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Deep Cartograph", description="Map trajectories onto Collective Variables.")
    
    parser.add_argument('-conf', '-configuration', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    
    # Input files
    parser.add_argument('-traj_folder', dest='trajectory_folder', help="Path to folder with trajectories to compute the CVs. To compare the ensembles of different trajectories on the same CV space, provide different trajectories in this folder.", required=True)
    parser.add_argument('-top_folder', dest='topology_folder', help="Path to folder with topology files for the trajectories. Should have the same name as the corresponding trajectory in -traj_folder.", required=True)
    
    parser.add_argument('-ref_traj_folder', dest='ref_trajectory_folder', help="Path to folder with reference trajectories or structures. To project alongside the main trajectories.", required=False)
    parser.add_argument('-ref_top_folder', dest='ref_topology_folder', help="Path to folder with topology files of the reference data. Should have the same name as the corresponding reference file in the reference folder.", required=False)
    
    parser.add_argument('-label_reference', dest='label_reference', action='store_true', help="Use labels for reference data (names of the files in the reference folder). Do not use this option if there are many samples in the reference data.", default=False)
    
    # Options
    parser.add_argument('-restart', dest='restart', action='store_true', help="Set to restart the workflow from the last finished step. Erase those step folders that you want to repeat.", default=False)
    parser.add_argument('-dim', '-dimension', dest='dimension', type=int, help="Dimension of the CV to train or compute", required=False)
    parser.add_argument('-cvs', nargs='+', help='Collective variables to train or compute (pca, ae, tica, deep_tica)', required=False)
    parser.add_argument('-out', '-output', dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument('-v', '-verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # If output folder is not given, create default
    if args.output_folder is None:
        output_folder = 'deep_cartograph'
    else:
        output_folder = args.output_folder
          
    # If restart is False, create unique output folder
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    
    # Run tool
    deep_cartograph(
        configuration = configuration, 
        trajectory_folder = args.trajectory_folder, 
        topology_folder = args.topology_folder,
        ref_trajectory_folder = args.ref_trajectory_folder,
        ref_topology_folder = args.ref_topology_folder,
        label_reference = args.label_reference,
        dimension = args.dimension, 
        cvs = args.cvs, 
        restart = args.restart,
        output_folder = output_folder)
    
    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))