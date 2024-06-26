import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict

# Local imports
from deep_cartograph.tools.compute_features import compute_features
from deep_cartograph.tools.filter_features import filter_features
from deep_cartograph.tools.train_colvars import train_colvars
from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration, validate_configuration, files_exist
from deep_cartograph.yaml_schemas.deep_cartograph_schema import DeepCartographSchema

########
# TOOL #
########

def deep_cartograph(configuration: Dict, trajectory: str, topology: str, dimension: int = None, model: str = None, output_folder: str = 'deep_cartograph') -> None:
    """
    Function that maps the trajectory onto the collective variables.

    Parameters
    ----------

        configuration:       configuration dictionary (see default_config.yml for more information)
        trajectory:          Path to the trajectory file that will be analyzed.
        topology:            Path to the topology file of the system.
        dimension:           Dimension of the collective variables to train or compute, if None, the value in the configuration file is used
        model:               Type of collective variable model to train or compute (PCA, AE, TICA, DTICA, ALL), if None, the value in the configuration file is used
        output_folder:       Path to the output folder
    """

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Start timer
    start_time = time.time()

    # If cv dimension and type are given, update the configuration accordingly
    if dimension is not None:
        configuration['train_colvars']['cv']['dimension'] = dimension
    if model is not None:
        configuration['train_colvars']['cv']['model'] = model

    # Validate configuration
    validate_configuration(configuration, DeepCartographSchema)

    # Check if files exist
    if not files_exist(trajectory, topology):
        logger.error("One or more files do not exist. Exiting...")
        sys.exit(1)

    # Step 1: Compute features for trajectory
    step1_output_folder = os.path.join(output_folder, 'compute_features_traj')
    traj_colvars_path = compute_features(
        configuration = configuration['compute_features'], 
        trajectory = trajectory, 
        topology = topology, 
        output_folder = step1_output_folder)
    
    # Step 1.2: Compute features for topology
    step1_output_folder = os.path.join(output_folder, 'compute_features_top')
    top_colvars_path = compute_features(
        configuration = configuration['compute_features'], 
        trajectory = topology, 
        topology = topology, 
        output_folder = step1_output_folder)

    # Step 2: Filter features
    step2_output_folder = os.path.join(output_folder, 'filter_features')
    filtered_features = filter_features(
            configuration = configuration['filter_features'], 
            colvars_path = traj_colvars_path,
            output_folder = step2_output_folder)

    # Step 3: Train colvars
    step3_output_folder = os.path.join(output_folder, 'train_colvars')
    train_colvars(
        configuration = configuration['train_colvars'],
        colvars_path = traj_colvars_path,
        feature_constraints = filtered_features,
        ref_colvars_path = top_colvars_path,
        dimension = dimension,
        model = model,
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
    logger.info("===============================================================================================")

########
# MAIN #
########

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Deep Cartograph", description="Map trajectories onto Collective Variables.")
    
    parser.add_argument('-conf', '-configuration', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    parser.add_argument('-traj', '-trajectory', dest='trajectory', help="Path to trajectory file, for which the features are computed.", required=True)
    parser.add_argument('-top', '-topology', dest='topology', help="Path to topology file.", required=True)
    parser.add_argument('-dim', '-dimension', dest='dimension', type=int, help="Dimension of the CV to train or compute", required=False)
    parser.add_argument('-m', '-model', dest='model', type=str, help="Type of CV model to train or compute (PCA, AE, TICA, DTICA, ALL)", required=False)
    parser.add_argument('-out', '-output', dest='output_folder', help="Path to the output folder", required=True)
    parser.add_argument('-v', '-verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Create unique output directory
    output_folder = get_unique_path(args.output_folder)
    create_output_folder(output_folder)

    # Read configuration
    configuration = read_configuration(args.configuration_path, output_folder)

    # Run tool
    deep_cartograph(
        configuration = configuration, 
        trajectory = args.trajectory,
        topology = args.topology,
        dimension = args.dimension, 
        model = args.model, 
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))