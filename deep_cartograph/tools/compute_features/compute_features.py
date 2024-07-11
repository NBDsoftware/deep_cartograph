import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict

# Local imports
from deep_cartograph.modules.plumed import utils as plumed_utils
from deep_cartograph.modules.plumed.input_file import input_file as plumed_input
from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration, validate_configuration, files_exist
from deep_cartograph.yaml_schemas.compute_features_schema import ComputeFeaturesSchema

########
# TOOL #
########

def compute_features(configuration: Dict, trajectory: str, topology: str, colvars_path: str = None, output_folder: str = 'compute_features') -> str:
    """
    Function that 

    Parameters
    ----------

        configuration:       configuration dictionary (see default_config.yml for more information)
        trajectory:          Path to the trajectory file that will be analyzed.
        topology:            Path to the topology file of the system.
        colvars_path:        Path to the output colvars file with the time series of the features.
        output_folder:       Path to the output folder
    """

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================")
    logger.info("Compute features")
    logger.info("================")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    create_output_folder(output_folder)

    # Validate configuration
    configuration = validate_configuration(configuration, ComputeFeaturesSchema, output_folder)

    # Check if files exist
    if not files_exist(trajectory):
        logger.error(f"Trajectory file {trajectory} does not exist. Exiting...")
        sys.exit(1)
    else:
        logger.info(f"Trajectory file: {trajectory}")
        
    if not files_exist(topology):
        logger.error(f"Topology file {topology} does not exist. Exiting...")
        sys.exit(1)

    # Create colvars path if not provided
    if colvars_path is None:
        colvars_path = os.path.join(output_folder, 'colvars.dat')

    # Build PLUMED input
    plumed_input_path, plumed_topology = plumed_input.track_features(configuration['plumed_settings'], topology, colvars_path, output_folder)

    # Construct plumed command
    plumed_command = plumed_utils.get_driver_command(plumed_input_path, trajectory, plumed_topology)

    # Execute plumed command
    plumed_utils.run_driver_command(plumed_command, configuration['plumed_environment'])

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Compute features): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return colvars_path

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
    tool_path = file_path.parent
    all_tools_path = tool_path.parent
    package_path = all_tools_path.parent

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

    parser = argparse.ArgumentParser("Deep Cartograph: Compute features", description="Compute features from a trajectory using PLUMED.")
    
    parser.add_argument('-conf', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    parser.add_argument('-trajectory', dest='trajectory', help="Path to trajectory file, for which the features are computed.", required=True)
    parser.add_argument('-topology', dest='topology', help="Path to topology file.", required=True)
    parser.add_argument('-colvars', dest='colvars_path', help="Path to the output colvars file that the PLUMED input will produce", required=True)
    parser.add_argument('-output', dest='output_folder', help="Path to the output folder", required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Create unique output directory
    output_folder = get_unique_path(args.output_folder)
    create_output_folder(output_folder)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run tool
    _ = compute_features(
        configuration = configuration, 
        trajectory = args.trajectory,
        topology = args.topology,
        colvars_path = args.colvars_path,
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))