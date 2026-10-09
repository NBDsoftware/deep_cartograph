"""
Trajectory augmentation tool: creates new trajectories with more frames by
interpolating between existing frames (optionally adding noise).
"""
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
    check_data,
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
    trajectory_data: Union[List[str], str],
    topology_data: Union[List[str], str],  
    num_replicas: int = 1,
    output_folder: str = "traj_augmentation",
) -> List[str]:
    """
    Create new trajectories with more frames by interpolating between the existing frames.

    Useful when only a few structures are available (e.g. a short path between two states).
    Gaussian noise can also be added to the new frames (see `default_config.yml`).

    Parameters
    ----------
    configuration : Dict
        Configuration dictionary (see `default_config.yml` for more details).

    trajectory_data : str or List[str]
        Trajectory file(s) or folder with trajectories to augment.
        Accepted formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`.

    topology_data : str or List[str]
        Topology file(s) or folder with topologies for the trajectories.
        A single topology is used for all trajectories; otherwise each topology must have
        the same name as its trajectory. Accepted format: `.pdb`.

    num_replicas : int, optional
        Number of augmented trajectories to create from each input trajectory. Each replica
        uses a different random seed, so replicas only differ if `noise_std` is set. Default: `1`.

    output_folder : str, optional
        Path to the output folder. Default: `"traj_augmentation"`.

    Returns
    -------
    augmented_trajectories : List[str]
        Paths to the augmented trajectory files.

    augmented_topologies : List[str]
        Paths to the topology files of the augmented trajectories (same order).
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
    
    # Check main data
    trajectories, topologies = check_data(trajectory_data, topology_data)
        
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

    # For each trajectory, perform augmentation across all replicas
    augmented_trajectories = []
    augmented_topologies = []
    base_seed = configuration['random_seed']
    for traj_path, top_path in zip(trajectories, topologies):
        traj_name = Path(traj_path).stem
        logger.info(f"Processing trajectory: {traj_name}")

        for replica in range(num_replicas):
            replica_seed = base_seed + replica
            suffix = f"_rep{replica}" if num_replicas > 1 else ""
            logger.info(f"  Replica {replica} (seed={replica_seed})")

            # Trajectory augmentation
            new_traj_path, new_top_path = md.interpolate_trajectory(
                                            topology_file=top_path,
                                            trajectory_file=traj_path,
                                            num_frames=configuration['num_frames'],
                                            keep_original_frames=configuration['keep_original_frames'],
                                            interpolation_method=configuration['interpolation_method'],
                                            noise_std=configuration['noise_std'],
                                            random_seed=replica_seed,
                                            atom_selection=configuration['atom_selection'],
                                            traj_format=configuration['traj_format'],
                                            prepare_trajectory=configuration['prepare_trajectory'],
                                            output_path=output_folder,
                                            suffix=suffix)
            augmented_trajectories.append(new_traj_path)
            augmented_topologies.append(new_top_path)
    
    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Trajectory Augmentation): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return augmented_trajectories, augmented_topologies

def set_logger(verbose: bool, log_path: str):
    """
    Set up logging for Deep Cartograph.

    Parameters
    ----------
    verbose : bool
        If True, log at DEBUG level. Otherwise, log at INFO level.

    log_path : str
        Path to the log file.

    Raises
    ------
    FileNotFoundError
        If the logging configuration files in `log_config/` are missing.
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
    """Parse the command-line arguments of the trajectory augmentation command."""
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
        '-traj_data', dest='trajectory_data', required=False, nargs='+',
        help=(
            "List of trajectory paths or path to folder with trajectories. "
            "These trajectories will be augmented. "
            "Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd."
        )
    )
    parser.add_argument(
        '-top_data', dest='topology_data', required=False, nargs='+',
        help=(
            "List of topology paths or path to folder with topologies for the trajectories. "
            "If a folder is provided, each topology should have the same name as the "
            "corresponding trajectory in -traj_data. If a single topology file is provided, it will be used for all trajectories. "
            "Accepted format: .pdb."
        )
    )
    parser.add_argument(
        '-n', '-num_replicas', dest='num_replicas', type=int, default=1, required=False,
        help="Number of replicas to generate for each trajectory. Replicas only differ if noise_std is not null. Default: 1."
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
    """Entry point of the trajectory augmentation command: read the arguments and configuration, then run the tool."""

    args = parse_arguments()

    # Create output folder if it does not exist
    output_folder = args.output_folder if args.output_folder else 'traj_augmentation'
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run traj_augmentation
    _ = traj_augmentation(
        configuration = configuration, 
        trajectory_data = args.trajectory_data,
        topology_data = args.topology_data,
        num_replicas = args.num_replicas,
        output_folder = output_folder)

if __name__ == "__main__":
    main()