import os
import time
import shutil
import numpy as np
import logging.config
from pathlib import Path
from typing import Dict, List

########
# TOOL #
########

def analyze_geometry(configuration: Dict, trajectories: List[str], topologies: List[str], 
                     output_folder: str = 'analyze_geometry') -> str:
    """
    Function that performs different geometrical analysis of a trajectory using MDAnalysis

    Parameters
    ----------

        configuration:
            A configuration dictionary (see default_config.yml for more information)
            
        trajectories:          
            Path to trajectories that will be analyzed.
            
        topology_data:            
            Path to topology files of the trajectories.
            
        output_folder:       
            (Optional) Path to the output folder
    """

    from deep_cartograph.modules.common import validate_configuration, save_data
    from deep_cartograph.modules.figures import plot_data
    from deep_cartograph.modules.md import RMSD, RMSF
    
    from deep_cartograph.yaml_schemas.analyze_geometry import AnalyzeGeometrySchema

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================")
    logger.info("Analyze geometry")
    logger.info("================")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Validate configuration
    configuration = validate_configuration(configuration, AnalyzeGeometrySchema, output_folder)
    
    if not configuration['run']:
        logger.info("Skipping Analyze Geometry step.")
        return output_folder
    
    # Get time step per frame in ns
    dt_per_frame = float(configuration['dt_per_frame'])* 1e-3 

    # For each type of analysis in the configuration
    for category, analyses in configuration['analysis'].items():
        
        # If there are analyses of this type
        if analyses:
            
            logger.info(f"Analyzing {category}...")
        
            # For each analysis of this type
            for name, params in analyses.items():
                
                logger.info(f" - {name}")
                
                title = params['title']
                y_label = f"{category} (A)"
                
                # Save here analysis results for each trajectory
                y_data = {}
                x_data = {}
                
                # Analyze each trajectory
                for trajectory, topology in zip(trajectories, topologies):
                    
                    # Get trajectory name
                    trajectory_name = Path(trajectory).stem
                    
                    # Get selections
                    selection = params['selection']
                    fit_selection = params['fit_selection']
                    
                    # Execute analysis
                    if category == 'RMSD':
                        y_data[trajectory_name] = RMSD(trajectory, topology, selection, fit_selection)
                        x_data[trajectory_name] = np.arange(0, len(y_data[trajectory_name])) * dt_per_frame
                        x_label = 'Time (ns)'
                    elif category == 'RMSF':
                        y_data[trajectory_name], x_data[trajectory_name] = RMSF(trajectory, topology, selection, fit_selection)
                        x_label = 'Residue'
                
                # Create figure path
                figure_path = os.path.join(output_folder, f'{name}_{category}.png')

                # Save figure with results
                plot_data(y_data, x_data, title, y_label, x_label, figure_path)
                
                # Save csv with results
                save_data(y_data, x_data, y_label, x_label, output_folder)

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Analyze geometry): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return

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
    from deep_cartograph.modules.common import get_unique_path, read_configuration, check_data

    parser = argparse.ArgumentParser("Deep Cartograph: Analyze geometry", description="Analyze geometry from a trajectory using PLUMED.")
    
    parser.add_argument('-conf', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    
    parser.add_argument('-traj_data', dest='trajectory_data', help="Path to trajectory or folder with trajectories to analyze.", required=True)
    parser.add_argument('-top_data', dest='topology_data', help="Path to topology or folder with topology files for the trajectories. If a folder is provided, each topology should have the same name as the corresponding trajectory in -traj_data.", required=True)
    
    parser.add_argument('-output', dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Determine output folder, if restart is False, create a unique output folder
    output_folder = args.output_folder if args.output_folder else 'analyze_geometry'
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)
    
    # Read configuration
    configuration = read_configuration(args.configuration_path)
    
    # Check main input folders
    trajectories, topologies = check_data(args.trajectory_data, args.topology_data)

    # Run Analyze Geometry tool
    _ = analyze_geometry(
        configuration = configuration, 
        trajectories = trajectories,
        topologies = topologies,
        output_folder = output_folder)

if __name__ == "__main__":

    main()