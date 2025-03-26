import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Union

########
# TOOL #
########

def deep_cartograph(configuration: Dict, trajectory_data: str, topology_data: str, validation_traj_data: Union[str, None] = None, 
                    validation_top_data: Union[str, None] = None, reference_topology: Union[str, None] = None, dimension: Union[int, None] = None, 
                    cvs: Union[List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']], None] = None, 
                    restart: bool = False, output_folder: Union[str, None] = None) -> None:
    """
    Main API of the Deep Cartograph workflow.
    
    NOTE: Currently we request same number of trajectories and topologies.
          In the future we might want to provide individual topologies as data 
          or produce plumed enhanced sampling files for specific topologies.

    Parameters
    ----------

        configuration         
            Configuration dictionary (see default_config.yml for more information)
            
        trajectory_data       
            Path to trajectory or folder with trajectories to analyze. 
            They will be used to compute the collective variables. 
            Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd  
            
        topology_data         
            Path to topology or folder with topology files for the trajectories. 
            If a single topology file is provided, it is used for all trajectories. 
            If a folder is given, each trajectory should have a corresponding topology file with the same name.
            Accepted formats: .pdb
            
        validation_traj_data (Optional)
            Path to validation trajectory or folder with validation trajectories. 
            To project onto the CV alongside 'trajectory_data' but not used to compute the CVs. 
            Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd 
        
        validation_top_data (Optional)    
            Path to validation topology or folder with validation topologies. 
            If a single topology file is provided, it is used for all validation trajectories.
            If a folder is given, each validation trajectory should have a corresponding topology file with the same name.
            Accepted formats: .pdb
        
        reference_topology (Optional)
            Path to reference topology file. The reference topology is used to find the features from the user selections.
            Default is the first topology in topology_data. Accepted formats: .pdb
        
        dimension (Optional)          
            Dimension of the collective variables to train or compute, overwrites the value in the configuration if provided
        
        cvs (Optional)                    
            List of collective variables to train or compute ['pca', 'ae', 'tica', 'htica', 'deep_tica'], overwrites the value in the configuration if provided
            
        restart (Optional)
            Set to restart the workflow from the last finished step. Erase those step folders that you want to repeat. Default is False.
        
        output_folder (Optional)      
            Path to the output folder. Default is 'deep_cartograph'
    """
    from deep_cartograph.modules.common import check_data, check_validation_data, create_output_folder, get_unique_path, validate_configuration, read_feature_constraints
    from deep_cartograph.yaml_schemas.deep_cartograph import DeepCartograph
    
    from deep_cartograph.tools import analyze_geometry
    from deep_cartograph.tools import compute_features
    from deep_cartograph.tools import filter_features
    from deep_cartograph.tools import train_colvars
    
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
    create_output_folder(output_folder)

    # Validate configuration
    configuration = validate_configuration(configuration, DeepCartograph, output_folder)
    
    # Check main input folders
    trajectories, topologies = check_data(trajectory_data, topology_data)
    
    # Set reference topology
    if not reference_topology:
        reference_topology = topologies[0]
    elif not os.path.exists(reference_topology):
        logger.error(f"Reference topology file missing. Exiting...")
        sys.exit(1)
        
    # Check validation input folders
    validation_trajs, validation_tops = check_validation_data(validation_traj_data, validation_top_data)
    
    # Step 0: Analyze geometry
    # ------------------------
    
    args = {
        'configuration': configuration['analyze_geometry'],
        'trajectories': trajectories,
        'topologies': topologies,
        'output_folder': os.path.join(output_folder, 'analyze_geometry')
    }
    analyze_geometry(**args)

    # Step 1: Compute features
    # ------------------------
    
    # Compute features for all trajectories
    args = {
        'configuration': configuration['compute_features'], 
        'trajectories': trajectories, 
        'topologies': topologies, 
        'output_folder': os.path.join(output_folder, 'compute_features')
    }
    traj_colvars_paths = compute_features(**args)
        
    # Compute features for validation data
    args = {
        'configuration': configuration['compute_features'], 
        'trajectories': validation_trajs, 
        'topologies': validation_tops, 
        'output_folder': os.path.join(output_folder, 'compute_ref_features')
    }
    validation_colvars_paths = compute_features(**args)
        
    # If there are less than 10 validation trajectories, use their names as labels
    if len(validation_trajs) < 10: 
        validation_labels = [Path(val_trajectory).stem for val_trajectory in validation_trajs]
    else:
        validation_labels = None

    ## Step 2: Filter features
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
    filtered_features = read_feature_constraints(output_features_path) 

    # Step 3: Train colvars
    # ---------------------
    
    args = {
        'configuration': configuration['train_colvars'],
        'colvars_paths': traj_colvars_paths,
        'feature_constraints': filtered_features,
        'validation_colvars_paths': validation_colvars_paths,
        'validation_labels': validation_labels,
        'dimension': dimension,
        'cvs': cvs,
        'trajectories': trajectories,
        'topologies': topologies,
        'samples_per_frame': 1/configuration['compute_features']['plumed_settings']['traj_stride'],
        'output_folder': os.path.join(output_folder, 'train_colvars')
    }
    train_colvars(**args)
            
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

def main():
    
    parser = argparse.ArgumentParser("Deep Cartograph", description="Map trajectories onto Collective Variables.")
    
    # Required input files
    parser.add_argument('-conf', '-configuration', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    parser.add_argument('-traj_data', dest='trajectory_data', help="Path to trajectory or folder with trajectories to analyze. Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd ", required=True)
    parser.add_argument('-top_data', dest='topology_data', help="Path to topology or folder with topology files for the trajectories. If a folder is provided, each topology should have the same name as the corresponding trajectory in -traj_data. Accepted formats: .pdb", required=True)
    
    # Optional input files
    parser.add_argument('-val_traj_data', dest='validation_traj_data', help="Path to validation trajectory or folder with validation trajectories. To project onto the CV alongside 'trajectory_data' but not used to compute the CVs.", required=False)
    parser.add_argument('-val_topology_data', dest='validation_top_data', help="Path to validation topology or folder with validation topologies. If a folder is provided, each topology should have the same name as the corresponding validation trajectory in -ref_traj_data.", required=False)
    parser.add_argument('-ref_top', dest='reference_topology', help="Path to reference topology file. The reference topology is used to find the features from the user selections. Default is the first topology in topology_data. Accepted formats: .pdb", required=False)
    
    # Options
    parser.add_argument('-restart', dest='restart', action='store_true', help="Set to restart the workflow from the last finished step. Erase those step folders that you want to repeat.", default=False)
    parser.add_argument('-dim', '-dimension', dest='dimension', type=int, help="Dimension of the CV to train or compute, overwrites the configuration input YML.", required=False)
    parser.add_argument('-cvs', nargs='+', help='Collective variables to train or compute (pca, ae, tica, htica, deep_tica), overwrites the configuration input YML.', required=False)
    parser.add_argument('-out', '-output', dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument('-v', '-verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()
    
    from deep_cartograph.modules.common import read_configuration, get_unique_path

    # Set logger
    set_logger(verbose=args.verbose)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # If output folder is not given, create default
    if args.output_folder is None:
        output_folder = 'deep_cartograph'
    else:
        output_folder = args.output_folder
          
    # If restart is False, create unique output folder
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    
    # Run tool
    deep_cartograph(
        configuration = configuration, 
        trajectory_data = args.trajectory_data, 
        topology_data = args.topology_data,
        validation_traj_data = args.validation_traj_data,
        validation_top_data = args.validation_top_data,
        reference_topology = args.reference_topology,
        dimension = args.dimension, 
        cvs = args.cvs, 
        restart = args.restart,
        output_folder = output_folder)
    
    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))
    
    
if __name__ == "__main__":
    
    main()