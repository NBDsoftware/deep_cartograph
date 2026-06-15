import os
import sys
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, Union, List, Optional

from deep_cartograph.yaml_schemas.compute_features import ComputeFeaturesSchema
from deep_cartograph.modules.features.common import find_common_features
import deep_cartograph.modules.features as features
import deep_cartograph.modules.plumed as plumed 
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

def compute_features(
    configuration: Dict,
    trajectories: Union[List[str], str],
    topologies: Union[List[str], str],
    reference_topology: Optional[str] = None,
    reference_features: Optional[List[str]] = None,
    traj_stride: Optional[int] = None,   
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
            Defaults to the first topology in `topologies` if not provided. 
            Accepted format: `.pdb`.
            
        reference_features (Optional[List[str]]):
            List of features to compute in the reference topology
            If not given they will be extracted from the reference topology using the features configuration
            
        traj_stride (Optional[int]):
            Stride for reading the trajectory. Default: 1 (read all frames).
            Note: This parameter is also specified in the configuration file.
            If both are provided, the function argument takes precedence.
        
        output_folder (Optional[str]):
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
    
    # If the output exists already, skip the step
    skip_step = True
    colvars_paths = [os.path.join(os.path.join(output_folder, Path(traj).stem), 'colvars.dat') for traj in trajectories]
    for colvars_path in colvars_paths:
        if not os.path.exists(colvars_path):
            skip_step = False
            break
    if skip_step:
        logger.info(f"Colvars files already exist in {output_folder}. Skipping feature computation.")
        return colvars_paths

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Validate configuration
    configuration = validate_configuration(configuration, ComputeFeaturesSchema, output_folder)
    
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
    
    if reference_topology is None:
        reference_topology = topologies[0]
        logger.info(f"No reference topology provided. Using the first topology as reference: {reference_topology}")

    if not os.path.exists(reference_topology):
        logger.error(f"Reference topology file missing. Exiting...")
        sys.exit(1)
        
    if reference_features is None:
        args = { 
            'features_configuration': configuration['plumed_settings']['features'],
            'topologies': topologies,
            'reference_topology': reference_topology,
            'output_folder': os.path.join(output_folder, 'common_features') 
        }
        reference_features = find_common_features(**args)
            
        
    # Enforce trajectory stride from function argument
    if traj_stride:
        configuration['plumed_settings']['traj_stride'] = traj_stride
        
    # Create a reference plumed topology file
    ref_plumed_topology = os.path.join(output_folder, 'ref_topology.pdb')
    md.create_pdb(reference_topology, ref_plumed_topology)

    # Compute the features for each traj and topology
    colvars_paths = []
    for topology, trajectory in zip(topologies, trajectories):
        
        # Find top and traj names
        top_name = Path(topology).stem
        traj_name = Path(trajectory).stem
        
        traj_output_folder   = os.path.join(output_folder, traj_name)
        os.makedirs(traj_output_folder, exist_ok=True)
        plumed_input_path    = os.path.join(traj_output_folder, 'plumed_input.dat')
        plumed_topology_path = os.path.abspath(os.path.join(traj_output_folder, 'plumed_topology.pdb'))
        colvars_path         = os.path.join(traj_output_folder, 'colvars.dat')
        colvars_paths.append(colvars_path)
        
        # Skip if colvars file already exists
        if os.path.exists(colvars_path):
            logger.info(f"Skipping {top_name}. Colvars file already exists.")
            continue
        
        md.create_pdb(topology, plumed_topology_path)
        logger.debug(f"Translating features from reference topology to topology {Path(topology).name}")
        features_list = features.Translator(ref_plumed_topology, plumed_topology_path, reference_features).run()
        
        # Check there are no empty features
        if None in features_list:
            raise ValueError(f"Some common reference features could not be translated to topology {top_name}. This should not happen, please check the common features and this topology. Exiting...")
        
        logger.info(f"Computing features for {traj_name} with topology {top_name}...")

        # If features contain coordinates, we need to fit the structure to the reference topology
        need_fit_template = any(feat.startswith("coord") for feat in features_list)
        if need_fit_template:
            fit_template_path = os.path.join(traj_output_folder, "fit_template.pdb")
            md.create_plumed_rmsd_template(topology, fit_template_path)
            logger.warning(f"""This CV is using coordinates as features, 
                           make sure the topology and trajectory for {top_name} are 
                           fitted to the rest of topologies.
                           Otherwise the features computed won't have the same meaning between systems.""")
        else:
            fit_template_path = None

        # Create the plumed input builder
        builder_args = {
            'plumed_input_path': plumed_input_path,
            'topology_path': plumed_topology_path,
            'features_list': features_list,
            'traj_stride': configuration['plumed_settings']['traj_stride'],
            'fit_template_path': fit_template_path
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
        prog="Deep Cartograph: Compute features",
        description="Compute features from a trajectory using PLUMED."
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-trajectory', dest='trajectory', type=str, required=True, nargs='+',
        help="Path to trajectory files, for which the features are computed."
    )
    parser.add_argument(
        '-topology', dest='topology', type=str, required=True, nargs='+',
        help="Path to topology files."
    )
    
    # Optional arguments
    parser.add_argument(
        '-traj_stride', dest='traj_stride', type=int, required=False,
        help="Stride for reading the trajectory. Default: 1 (read all frames)."
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
    output_folder = args.output_folder if args.output_folder else 'compute_features'
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run Compute Features tool
    _ = compute_features(
        configuration = configuration, 
        trajectories = args.trajectory,
        topologies = args.topology,
        traj_stride = args.traj_stride,
        output_folder = output_folder)

if __name__ == "__main__":
    main()