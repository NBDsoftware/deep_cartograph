import os
import sys
import time
import shutil
import logging.config
from pathlib import Path
from typing import Dict, Union, List

########
# TOOL #
########

def compute_features(configuration: Dict, trajectories: Union[List[str], str], topologies: Union[List[str], str], 
                     output_folder: str = 'compute_features') -> List[str]:
    """
    Function that computes features from a trajectory using PLUMED.

    Parameters
    ----------

        configuration:
            A configuration dictionary (see default_config.yml for more information)
            
        trajectories:          
            Paths to the trajectory files that will be analyzed.
            
        topologies:            
            Paths to the topology files of the trajectories.
            
        output_folder:       
            (Optional) Path to the output folder
        
    Returns
    -------

        colvars_paths:        
            Paths to the output colvars files with the time series of the features.
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
    
    if isinstance(trajectories, str):
        trajectories = [trajectories]
    if isinstance(topologies, str):
        topologies = [topologies]
        
    # Check if files exist
    if not files_exist(*trajectories):
        logger.error(f"Trajectory file missing. Exiting...")
        sys.exit(1)
        
    if not files_exist(*topologies):
        logger.error(f"Topology file missing. Exiting...")
        sys.exit(1)
        
    colvars_paths = []
        
    for trajectory, topology in zip(trajectories, topologies):

        traj_name = Path(trajectory).stem
        traj_output_folder = os.path.join(output_folder, traj_name)
        plumed_input_path = os.path.join(traj_output_folder, 'plumed_input.dat')
        plumed_topology_path = os.path.abspath(os.path.join(traj_output_folder, 'plumed_topology.pdb'))
        colvars_path = os.path.join(traj_output_folder, 'colvars.dat')
        colvars_paths.append(colvars_path)
        
        # Skip if colvars file already exists
        if os.path.exists(colvars_path):
            logger.info(f"Skipping {traj_name}. Colvars file already exists.")
            continue
        
        # Create trajectory output folder
        create_output_folder(traj_output_folder)
    
        # Create new topology file
        md.create_pdb(topology, plumed_topology_path)
    
        # Find list of features to compute with PLUMED
        features_list = md.get_features_list(configuration['plumed_settings']['features'], plumed_topology_path)

        # Create the plumed input builder
        builder_args = {
            'input_path': plumed_input_path,
            'topology_path': plumed_topology_path,
            'feature_list': features_list,
            'traj_stride': configuration['plumed_settings']['traj_stride']
        }
        plumed_builder = plumed.input.builder.ComputeFeaturesBuilder(**builder_args)
        plumed_builder.build(colvars_path)

        # Construct plumed driver command
        driver_command_args = {
            'plumed_input': plumed_input_path,
            'traj_path': trajectory,
            'num_atoms':  md.get_number_atoms(topology),
            'output_path': traj_output_folder
        }
        plumed_command = plumed.cli.get_driver_command(**driver_command_args)

        # Execute command
        run_args = {
            'plumed_command': plumed_command,
            'plumed_settings': configuration['plumed_environment'],
            'plumed_timeout': configuration['plumed_settings']['timeout']   
        }
        plumed.cli.run_plumed(**run_args)
    
        # Check output file
        plumed.colvars.check(colvars_path)

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Compute features): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return colvars_paths

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