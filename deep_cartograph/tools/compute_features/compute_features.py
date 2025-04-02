import os
import sys
import time
import shutil
import argparse
import numpy as np
import logging.config
from pathlib import Path
from typing import Dict, Union, List, Optional

from deep_cartograph.yaml_schemas.compute_features import ComputeFeaturesSchema
import deep_cartograph.modules.plumed as plumed 
import deep_cartograph.modules.md as md
from deep_cartograph.modules.common import ( 
    get_unique_path, 
    read_configuration,
    create_output_folder, 
    validate_configuration, 
    files_exist
)

########
# TOOL #
########

def compute_features(
    configuration: Dict,
    trajectories: Union[List[str], str],
    topologies: Union[List[str], str],
    reference_topology: Optional[str] = None,
    output_folder: str = "compute_features",
) -> List[str]:
    """
    Computes features from a trajectory using PLUMED.

    Args:
        configuration (Dict): 
            Configuration dictionary (see `default_config.yml` for more details).
        
        trajectories (Union[List[str], str]): 
            Path(s) to the trajectory files to be analyzed.
            If a single path is provided as a string, it will be converted to a list.
        
        topologies (Union[List[str], str]): 
            Path(s) to the topology files corresponding to the trajectories.
            Must be in the same order as `trajectories`.
            If a single path is provided as a string, it will be converted to a list.
        
        reference_topology (Optional[str]): 
            Path to the reference topology file.
            Used to extract features from user selections.
            Defaults to the first topology in `topologies`.
            Accepted format: `.pdb`.
        
        output_folder (str, optional): 
            Path to the output folder where computed features will be stored.
            Default: `"compute_features"`.

    Returns:
        List[str]: Paths to the output colvars files containing the time series of the computed features.
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
        
    # Set reference topology
    if not reference_topology:
        reference_topology = topologies[0]
    elif not os.path.exists(reference_topology):
        logger.error(f"Reference topology file missing. Exiting...")
        sys.exit(1)
        
    # Create a reference plumed topology file
    ref_plumed_topology = os.path.join(output_folder, 'ref_topology.pdb')
    md.create_pdb(reference_topology, ref_plumed_topology)
    
    # Find list of features to compute from reference topology - user selection of features refers to this topology
    ref_feature_list = md.get_features_list(configuration['plumed_settings']['features'], ref_plumed_topology)
 
    # For each topology
    features_lists = []
    for topology in topologies:
        
        # Find top name
        top_name = Path(topology).stem
    
        # Create output folder
        top_output_folder = os.path.join(output_folder, top_name)
        create_output_folder(top_output_folder)
        
        # Create new topology file
        plumed_topology = os.path.join(top_output_folder, 'plumed_topology.pdb')
        md.create_pdb(topology, plumed_topology)
        
        # Translate features to new topology
        features_list = plumed.features.FeatureTranslator(ref_plumed_topology, plumed_topology, ref_feature_list).run()
        features_lists.append(features_list)

    # Keep just the features available in all topologies
    masks = np.array([[x is not None for x in lst] for lst in features_lists])
    mask =  masks.all(axis=0)
    common_features_lists = [[lst[i] for i in range(len(lst)) if mask[i]] for lst in features_lists]
        
    # Compute the features for each traj and topology
    colvars_paths = []
    for i in range(len(topologies)):

        topology = topologies[i]
        top_name = Path(topology).stem
        trajectory = trajectories[i]
        features_list = common_features_lists[i]
        
        logger.info(f"Computing features for {top_name} using {Path(trajectory).stem}")

        top_output_folder = os.path.join(output_folder, top_name)

        plumed_input_path = os.path.join(top_output_folder, 'plumed_input.dat')
        plumed_topology_path = os.path.abspath(os.path.join(top_output_folder, 'plumed_topology.pdb'))
        colvars_path = os.path.join(top_output_folder, 'colvars.dat')
        colvars_paths.append(colvars_path)
        
        # Skip if colvars file already exists
        if os.path.exists(colvars_path):
            logger.info(f"Skipping {top_name}. Colvars file already exists.")
            continue

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
            'output_path': top_output_folder
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

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="Deep Cartograph: Compute features",
        description="Compute features from a trajectory using PLUMED."
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
        '-colvars', dest='colvars_path', type=str, required=True,
        help="Path to the output colvars file that the PLUMED input will produce."
    )
    
    # Optional arguments
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