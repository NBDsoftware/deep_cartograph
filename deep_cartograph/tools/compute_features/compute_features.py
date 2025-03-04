import os
import sys
import time
import shutil
import logging.config
from pathlib import Path
from typing import Dict, Union

########
# TOOL #
########

def compute_features(configuration: Dict, trajectory: str, topology: str, colvars_path: Union[str, None] = None, 
                     output_folder: str = 'compute_features') -> str:
    """
    Function that computes features from a trajectory using PLUMED.

    Parameters
    ----------

        configuration:
            A configuration dictionary (see default_config.yml for more information)
            
        trajectory:          
            Path to the trajectory file that will be analyzed.
            
        topology:            
            Path to the topology file of the trajectory.
            
        colvars_path:        
            (Optional) Path to the output colvars file with the time series of the features.
            
        output_folder:       
            (Optional) Path to the output folder
        
    Returns
    -------

        colvars_path:        
            Path to the output colvars file with the time series of the features.
    """

    from deep_cartograph.modules.common import create_output_folder, validate_configuration, files_exist
    from deep_cartograph.yaml_schemas.compute_features import ComputeFeaturesSchema
    import deep_cartograph.modules.plumed as plumed 
    import deep_cartograph.modules.md as md

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
        
    # Set plumed input path
    plumed_input_path = os.path.join(output_folder, 'plumed_input.dat')
    
    # Set plumed topology path
    plumed_topology_path = os.path.abspath(os.path.join(output_folder, 'plumed_topology.pdb'))
    
     # Create new topology file
    md.create_pdb(topology, plumed_topology_path)
    
    # Find list of features to compute with PLUMED
    features_list = md.get_features_list(configuration['plumed_settings']['features'], plumed_topology_path)

    # Create the corresponding PLUMED Builder
    plumed_builder = plumed.input.TrackFeaturesBuilder(plumed_input_path, plumed_topology_path, features_list, configuration['plumed_settings'])
    plumed_builder.build(colvars_path)

    # Construct plumed driver command
    plumed_command = plumed.cli.get_driver_command(plumed_input_path, trajectory, md.get_number_atoms(topology), output_folder)

    # Execute plumed command
    plumed.cli.run_plumed(plumed_command, configuration['plumed_environment'], configuration['plumed_settings']['timeout'])
    
    # Check output file
    plumed.colvars.check(colvars_path)

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

def main():
    
    import argparse
    from deep_cartograph.modules.common import get_unique_path, read_configuration

    parser = argparse.ArgumentParser("Deep Cartograph: Compute features", description="Compute features from a trajectory using PLUMED.")
    
    parser.add_argument('-conf', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    parser.add_argument('-trajectory', dest='trajectory', help="Path to trajectory file, for which the features are computed.", required=True)
    parser.add_argument('-topology', dest='topology', help="Path to topology file.", required=True)
    parser.add_argument('-colvars', dest='colvars_path', help="Path to the output colvars file that the PLUMED input will produce", required=True)
    parser.add_argument('-output', dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)
    
    # Give value to output_folder
    if args.output_folder is None:
        output_folder = 'compute_features'
    else:
        output_folder = args.output_folder
        
    # Create unique output directory
    output_folder = get_unique_path(output_folder)

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

if __name__ == "__main__":

    main()