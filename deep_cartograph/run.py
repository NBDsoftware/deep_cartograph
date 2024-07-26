import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal

# Local imports
from deep_cartograph import compute_features
from deep_cartograph import filter_features
from deep_cartograph import train_colvars
from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration, validate_configuration
from deep_cartograph.yaml_schemas.deep_cartograph import DeepCartograph

########
# TOOL #
########

def deep_cartograph(configuration: Dict, trajectory: str, topology: str, reference_folder: str = None, 
                    use_reference_labels: bool = False, dimension: int = None, 
                    cvs: List[Literal['pca', 'ae', 'tica', 'dtica']] = None, 
                    output_folder: str = 'deep_cartograph') -> None:
    """
    Function that maps the trajectory onto the collective variables.

    Parameters
    ----------

        configuration:        configuration dictionary (see default_config.yml for more information)
        trajectory:           Path to the trajectory file that will be analyzed.
        topology:             Path to the topology file of the system.
        reference_folder:     Path to the folder with reference data.
        use_reference_labels: Use labels for reference data (names of the files in the reference folder)
        dimension:            Dimension of the collective variables to train or compute, if None, the value in the configuration file is used
        cvs:                  List of collective variables to train or compute (pca, ae, tica, dtica), if None, the ones in the configuration file are used
        output_folder:        Path to the output folder
    """

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    create_output_folder(output_folder)

    # Validate configuration
    configuration = validate_configuration(configuration, DeepCartograph, output_folder)
    
    # Check if trajectory file exists
    if not os.path.isfile(trajectory):
        logger.error("Trajectory file not found: %s", trajectory)
        sys.exit(1)

    # Check if topology file exists
    if not os.path.isfile(topology):
        logger.error("Topology file not found: %s", topology)
        sys.exit(1)

    # Check if reference folder exists
    if reference_folder is not None:
        if not os.path.exists(reference_folder):
            logger.error("Reference folder not found: %s", reference_folder)
            sys.exit(1)

        # List only the files in the reference folder
        ref_file_paths = [os.path.join(reference_folder, f) for f in os.listdir(reference_folder) if os.path.isfile(os.path.join(reference_folder, f))]

        # Check if there are files in the reference folder
        if len(ref_file_paths) == 0:
            logger.error("Reference folder is empty: %s", reference_folder)
            sys.exit(1)

    # Step 1: Compute features for trajectory
    step1_output_folder = os.path.join(output_folder, 'compute_features_traj')
    traj_colvars_path = compute_features(
        configuration = configuration['compute_features'], 
        trajectory = trajectory, 
        topology = topology, 
        output_folder = step1_output_folder)
    
    # Step 1.2: Compute features for each reference file
    if reference_folder is not None:
        ref_colvars_paths = []
        ref_labels = []
        for ref_file_path in ref_file_paths:

            # Create unique output folder
            ref_file_name = Path(ref_file_path).stem
            step1_output_folder = os.path.join(output_folder, f"compute_features_{ref_file_name}")

            # Compute features for reference file
            ref_colvars_path = compute_features(
                configuration = configuration['compute_features'], 
                trajectory = ref_file_path, 
                topology = topology, 
                output_folder = step1_output_folder)
            
            # Save path to colvars file and name of reference file
            ref_colvars_paths.append(ref_colvars_path)
            ref_labels.append(ref_file_name)
    else:
        ref_colvars_paths = None
        ref_labels = None

    # Step 2: Filter features
    step2_output_folder = os.path.join(output_folder, 'filter_features')
    filtered_features = filter_features(
            configuration = configuration['filter_features'], 
            colvars_path = traj_colvars_path,
            output_folder = step2_output_folder)

    if not use_reference_labels:
        ref_labels = None

    # Step 3: Train colvars
    step3_output_folder = os.path.join(output_folder, 'train_colvars')
    train_colvars(
        configuration = configuration['train_colvars'],
        colvars_path = traj_colvars_path,
        feature_constraints = filtered_features,
        ref_colvars_path = ref_colvars_paths,
        ref_labels = ref_labels,
        dimension = dimension,
        cvs = cvs,
        output_folder = step3_output_folder)
            
    # End timer
    elapsed_time = time.time() - start_time

    # Write time to log in hours, minutes and seconds
    logger.info('Total elapsed time: %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

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
    parser.add_argument('-traj', '-trajectory', dest='trajectory', help="Path to trajectory file, for which the features are computed.", required=True)
    parser.add_argument('-top', '-topology', dest='topology', help="Path to topology file.", required=True)
    parser.add_argument('-ref', '-reference', dest='reference_folder', help="Path to folder with reference data. It should contain structures or trajectories.", required=False)
    parser.add_argument('-use_rl', '-use_reference_lab', dest='use_reference_labels', action='store_true', help="Use labels for reference data (names of the files in the reference folder)", default=False)
    parser.add_argument('-dim', '-dimension', dest='dimension', type=int, help="Dimension of the CV to train or compute", required=False)
    parser.add_argument('-cvs', nargs='+', help='Collective variables to train or compute (pca, ae, tica, dtica)', required=False)
    parser.add_argument('-out', '-output', dest='output_folder', help="Path to the output folder", required=True)
    parser.add_argument('-v', '-verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Create unique output directory
    output_folder = get_unique_path(args.output_folder)
    create_output_folder(output_folder)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run tool
    deep_cartograph(
        configuration = configuration, 
        trajectory = args.trajectory,
        topology = args.topology,
        reference_folder = args.reference_folder,
        use_reference_labels = args.use_reference_labels,
        dimension = args.dimension, 
        cvs = args.cvs, 
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))