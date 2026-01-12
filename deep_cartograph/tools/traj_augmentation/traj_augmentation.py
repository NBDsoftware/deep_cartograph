import os
import sys
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, Union, List

from deep_cartograph.yaml_schemas.traj_augmentation import TrajAugmentationSchema
import deep_cartograph.modules.md as md
from deep_cartograph.modules.common import ( 
    get_unique_path, 
    read_configuration,
    validate_configuration, 
    files_exist
)

########
# TOOL #
########

def traj_augmentation(
    configuration: Dict,
    trajectories: Union[List[str], str],
    topologies: Union[List[str], str],  
    output_folder: str = "traj_augmentation",
) -> List[str]:
    """
    Augments trajectory samples interpolating the existing frames.

    Args:
        configuration (Dict): 
            Configuration dictionary (see `default_config.yml` for more details).
        
        trajectories (Union[List[str], str]): 
            Path(s) to the trajectory files to be augmented.
            If a single path is provided as a string, it will be converted to a list.
        
        topologies (Union[List[str], str]): 
            Path(s) to the topology files corresponding to the trajectories.
            Must be in the same order as `trajectories`.
            If a single path is provided as a string, it will be converted to a list.
        
        output_folder (str, optional): 
            Path to the output folder where the augmented trajectories will be saved.
            Default: `"traj_augmentation"`.

    Returns:
        List[str]: Paths to the augmented trajectory files.
    """

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("=======================")
    logger.info("Trajectory Augmentation")
    logger.info("=======================")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Validate configuration
    configuration = validate_configuration(configuration, TrajAugmentationSchema, output_folder)
    
    if isinstance(trajectories, str):
        trajectories = [trajectories]
    if isinstance(topologies, str):
        topologies = [topologies]
        
    # Check the number of trajectories and topologies is the same
    if len(trajectories) != len(topologies):
        logger.error(f"Number of trajectories ({len(trajectories)}) and topologies ({len(topologies)}) do not match. Exiting...")
        sys.exit(1) 
        
    # Check if files exist
    if not files_exist(*trajectories):
        logger.error(f"Trajectory file missing. Exiting...")
        sys.exit(1)
    if not files_exist(*topologies):
        logger.error(f"Topology file missing. Exiting...")
        sys.exit(1)

    # For each trajectory, perform augmentation and relaxation
    augmented_trajectories = []
    augmented_topologies = []
    for traj_path, top_path in zip(trajectories, topologies):
        traj_name = Path(traj_path).stem
        logger.info(f"Processing trajectory: {traj_name}")

        # Trajectory augmentation
        new_traj_path, new_top_path = md.interpolate_trajectory(
                                        topology_file=top_path,
                                        trajectory_file=traj_path,
                                        num_frames=configuration['num_frames'],
                                        keep_original_frames=configuration['keep_original_frames'],
                                        interpolation_method=configuration['interpolation_method'],
                                        noise_std=configuration['noise_std'],
                                        atom_selection=configuration['atom_selection'],
                                        traj_format=configuration['traj_format'],
                                        output_path=output_folder)
        augmented_trajectories.append(new_traj_path)
        augmented_topologies.append(new_top_path)
    
    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Trajectory Augmentation): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return augmented_trajectories, augmented_topologies

def set_logger(verbose: bool, log_path: str):
    """
    Configures logging for Deep Cartograph. 
    
    If `verbose` is `True`, sets the logging level to DEBUG.
    Otherwise, sets it to INFO.

    Inputs
    ------

    Args:
        verbose (bool): If `True`, logging level is set to DEBUG. 
                        If `False`, logging level is set to INFO.
        log_path (str): Path to the log file where logs will be saved.
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
    
    # Pass the log_path to the fileConfig using the 'defaults' parameter
    config_path = debug_config_path if verbose else info_config_path
    logging.config.fileConfig(
        config_path,
        defaults={'log_path': log_path},
        disable_existing_loggers=True
    )

    logger = logging.getLogger("deep_cartograph")
    logger.info("Deep Cartograph: package for analyzing MD simulations using collective variables.")
    
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="Deep Cartograph: Trajectory Augmentation",
        description="Trajectory Augmentation Tool: Augments trajectory samples interpolating the existing frames."
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-trajectory', dest='trajectory', type=str, required=True,
        help="Path to trajectory file, for which the features are computed."
    )
    parser.add_argument(
        '-topology', dest='topology', type=str, required=True,
        help="Path to topology file."
    )
    
    parser.add_argument(
        '-output', dest='output_folder', type=str, required=False,
        help="Path to the output folder."
    )
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', default=False,
        help="Set the logging level to DEBUG."
    )
    
    return parser.parse_args()

########
# MAIN #
########

def main():

    args = parse_arguments()

    # Determine output folder, if restart is False, create a unique output folder
    output_folder = args.output_folder if args.output_folder else 'traj_augmentation'
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run traj_augmentation
    _ = traj_augmentation(
        configuration = configuration, 
        trajectory = args.trajectory,
        topology = args.topology,
        output_folder = output_folder)

if __name__ == "__main__":
    main()