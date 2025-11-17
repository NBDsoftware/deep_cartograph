import os
import sys
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Optional

from deep_cartograph.yaml_schemas.deep_cartograph import DeepCartograph
from deep_cartograph.tools import (
    analyze_geometry,
    compute_features,
    filter_features,
    train_colvars,
    traj_projection,
    traj_cluster
)
from deep_cartograph.modules.common import (
    check_data,
    get_unique_path,
    validate_configuration,
    read_features_list,
    read_configuration
)

########
# TOOL #
########

def deep_cartograph(
    configuration: Dict,
    trajectory_data: str,
    topology_data: str,
    supplementary_traj_data: Optional[str] = None,
    supplementary_top_data: Optional[str] = None,
    reference_topology: Optional[str] = None,
    dimension: Optional[int] = None,
    cvs: Optional[List[Literal["pca", "ae", "tica", "htica", "deep_tica"]]] = None,
    restart: bool = False,
    output_folder: Optional[str] = None,
) -> None:
    """
    Main API for the Deep Cartograph workflow.

    Args:
        configuration (Dict): 
            Configuration dictionary (refer to `default_config.yml` for details).
        
        trajectory_data (str): 
            Path to a trajectory file or directory containing multiple trajectories.
            These will be used to compute the collective variables.
            Accepted formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`.
        
        topology_data (str): 
            Path to a topology file or directory with topology files for trajectories.
            - If a single topology file is provided, it is used for all trajectories.
            - If a directory is provided, each topology file must match a trajectory filename.
            Accepted format: `.pdb`.
        
        supplementary_traj_data (Optional[str]): 
            Path to a supplementary trajectory file or directory.
            These trajectories will be projected onto the CV but not used for computing CVs.
            Example: experimental structures, coarse-grained simulations.
            Default: `None`.
            Accepted formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`.
        
        supplementary_top_data (Optional[str]): 
            Path to a supplementary topology file or directory.
            - If a single topology file is provided, it is used for all supplementary trajectories.
            - If a directory is provided, each supplementary trajectory must have a matching topology file.
            Default: `None`.
            Accepted format: `.pdb`.
        
        reference_topology (Optional[str]): 
            Path to a reference topology file used to determine features from user selections.
            Default: first topology file in `topology_data`.
            Accepted format: `.pdb`.
        
        dimension (Optional[int]): 
            Number of dimensions for the collective variables.
            If provided, this overrides the value in the configuration file.
            Default: `None`.
        
        cvs (Optional[List[Literal["pca", "ae", "tica", "htica", "deep_tica"]]]): 
            List of collective variables to train or compute.
            If provided, this overrides the configuration file settings.
            Default: `None`.
        
        restart (bool): 
            If `True`, restarts the workflow from the last completed step.
            Deletes step folders that need to be recomputed.
            Default: `False`.
        
        output_folder (Optional[str]): 
            Path to the output directory.
            Default: `"deep_cartograph"`.

    Returns:
        None
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
    os.makedirs(output_folder, exist_ok=True)

    # Validate configuration
    configuration = validate_configuration(configuration, DeepCartograph, output_folder)
    
    # Check main input folders
    trajectories, topologies = check_data(trajectory_data, topology_data)
    trajectory_names = [Path(traj).stem for traj in trajectories]
    
    # Set reference topology
    if not reference_topology:
        reference_topology = topologies[0]
    elif not os.path.exists(reference_topology):
        logger.error(f"Reference topology file missing: {reference_topology}")
        sys.exit(1)
    
    # STEP 0: Analyze geometry
    # ------------------------
    args = {
        'configuration': configuration['analyze_geometry'],
        'trajectories': trajectories,
        'topologies': topologies,
        'output_folder': os.path.join(output_folder, 'analyze_geometry')
    }
    analyze_geometry(**args)

    # STEP 1: Compute features
    # ------------------------
    # Compute features for all trajectories
    args = {
        'configuration': configuration['compute_features'], 
        'trajectories': trajectories, 
        'topologies': topologies, 
        'reference_topology': reference_topology,
        'output_folder': os.path.join(output_folder, 'compute_features')
    }
    traj_colvars_paths = compute_features(**args)
    
    # Compute features for supplementary data
    if supplementary_traj_data:
        supplementary_trajs, supplementary_tops = check_data(supplementary_traj_data, supplementary_top_data)
        sup_trajectory_names = [Path(traj).stem for traj in supplementary_trajs]
        args = {
            'configuration': configuration['compute_features'], 
            'trajectories': supplementary_trajs, 
            'topologies': supplementary_tops, 
            'reference_topology': reference_topology,
            'traj_stride': 1,  # Use all frames for supplementary data - typically smaller datasets
            'output_folder': os.path.join(output_folder, 'compute_ref_features')
        }
        supplementary_colvars_paths = compute_features(**args)
    else:
        supplementary_trajs, supplementary_tops = None, None
        sup_trajectory_names = None
        supplementary_colvars_paths = None

    # STEP 2: Filter features
    # ------------------------
    args = {
        'configuration': configuration['filter_features'], 
        'colvars_paths': traj_colvars_paths,
        'topologies': topologies,
        'reference_topology': reference_topology,
        'output_folder': os.path.join(output_folder, 'filter_features')
    }
    output_features_path = filter_features(**args)

    # Read filtered features
    filtered_features = read_features_list(output_features_path)

    # STEP 3: Train colvars
    # ---------------------
    args = {
        'configuration': configuration['train_colvars'],
        'train_colvars_paths': traj_colvars_paths,
        'train_topologies': topologies,
        'trajectory_names': trajectory_names,
        'reference_topology': reference_topology,
        'features_list': filtered_features,
        'dimension': dimension,
        'cvs': cvs,
        'frames_per_sample': configuration['compute_features']['plumed_settings']['traj_stride'],
        'output_folder': os.path.join(output_folder, 'train_colvars')
    }
    trained_cvs_data = train_colvars(**args)
    
    # STEP 4: Trajectory projection
    # -----------------------------
    if supplementary_trajs:
        args = { 
            'configuration' : configuration['traj_projection'],
            'colvars_paths': supplementary_colvars_paths,
            'topologies': supplementary_tops,
            'trajectory_names': sup_trajectory_names,
            'model_paths': [trained_cvs_data[cv]['model_path'] for cv in trained_cvs_data.keys()],
            'model_traj_paths': [trained_cvs_data[cv]['traj_paths'] for cv in trained_cvs_data.keys()],
            'output_folder': os.path.join(output_folder, 'traj_projection')
        }
        sup_cvs_data = traj_projection(**args)
    else:
        sup_cvs_data = {}

    # STEP 5: Trajectory clustering
    # -----------------------------
    for cv in trained_cvs_data.keys():

        logger.info(f"Clustering trajectories in CV space: {cv}")

        args = {
            'configuration': configuration['traj_cluster'],
            'cv_traj_paths': trained_cvs_data[cv]['traj_paths'],
            'trajectories': trajectories,
            'topologies': topologies,
            'sup_cv_traj_paths': sup_cvs_data.get(cv, {}).get('traj_paths', None),
            'sup_trajectories': supplementary_trajs,
            'sup_topologies': supplementary_tops,
            'frames_per_sample': configuration['compute_features']['plumed_settings']['traj_stride'],
            'output_folder': os.path.join(output_folder, 'traj_cluster', cv)
        }
        traj_cluster(**args)
    
    # End timer
    elapsed_time = time.time() - start_time

    # Write time to log in hours, minutes and seconds
    logger.info('Total elapsed time: %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))   

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
    package_path = file_path.parent

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
        prog="Deep Cartograph",
        description="Map trajectories onto Collective Variables."
    )

    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-traj_data', dest='trajectory_data', required=True,
        help=(
            "Path to trajectory or folder with trajectories to analyze. "
            "Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd."
        )
    )
    parser.add_argument(
        '-top_data', dest='topology_data', required=True,
        help=(
            "Path to topology or folder with topology files for the trajectories. "
            "If a folder is provided, each topology should have the same name as the "
            "corresponding trajectory in -traj_data. Accepted format: .pdb."
        )
    )

    # Optional input files
    parser.add_argument(
        '-sup_traj_data', dest='supplementary_traj_data', required=False,
        help=(
            "Path to supplementary trajectory or folder with supplementary trajectories. "
            "Used to project onto the CV alongside 'trajectory_data' but not used for computing CVs."
        )
    )
    parser.add_argument(
        '-sup_top_data', dest='supplementary_top_data', required=False,
        help=(
            "Path to supplementary topology or folder with supplementary topologies. "
            "If a folder is provided, each topology should match the corresponding "
            "supplementary trajectory in -sup_traj_data."
        )
    )
    parser.add_argument(
        '-ref_top', dest='reference_topology', required=False,
        help=(
            "Path to reference topology file. Used to find features from user selections. "
            "Defaults to the first topology in topology_data. Accepted format: .pdb."
        )
    )

    # Options
    parser.add_argument(
        '-restart', dest='restart', action='store_true', default=False,
        help="Restart workflow from the last finished step."
    )
    parser.add_argument(
        '-dim', '-dimension', dest='dimension', type=int, required=False,
        help="Dimension of the CV to train or compute. Overrides the configuration input YML."
    )
    parser.add_argument(
        '-cvs', nargs='+', required=False,
        help="Collective variables to train or compute (pca, ae, tica, htica, vae, deep_tica). "
             "Overrides the configuration input YML."
    )
    parser.add_argument(
        '-out', '-output', dest='output_folder', required=False,
        help="Path to the output folder."
    )
    parser.add_argument(
        '-v', '-verbose', dest='verbose', action='store_true', default=False,
        help="Set logging level to DEBUG."
    )

    return parser.parse_args()

########
# MAIN #
########


def main():
    """Main function to execute Deep Cartograph workflow."""
    
    args = parse_arguments()

    # Determine output folder, if restart is False, create a unique output folder
    output_folder = args.output_folder if args.output_folder else 'deep_cartograph'
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run Deep Cartograph tool
    deep_cartograph(
        configuration=configuration,
        trajectory_data=args.trajectory_data,
        topology_data=args.topology_data,
        supplementary_traj_data=args.supplementary_traj_data,
        supplementary_top_data=args.supplementary_top_data,
        reference_topology=args.reference_topology,
        dimension=args.dimension,
        cvs=args.cvs,
        restart=args.restart,
        output_folder=output_folder
    )


if __name__ == "__main__":
    main()