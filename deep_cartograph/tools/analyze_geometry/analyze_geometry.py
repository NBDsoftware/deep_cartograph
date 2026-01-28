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
                     ref_topologies: List[str], output_folder: str = 'analyze_geometry') -> str:
    """
    Function that performs different geometrical analysis of a trajectory using MDAnalysis.
    
    - RMSD: Root Mean Square Deviation -> with respect to the first frame or a listed reference structure
    - RMSF: Root Mean Square Fluctuation

    Parameters
    ----------

        configuration:
            A configuration dictionary (see default_config.yml for more information)
            
        trajectories:          
            Path to trajectories that will be analyzed.
            
        topology_data:            
            Path to topology files of the trajectories.
            
        ref_topologies:
            (Optional) List of paths to reference topology files to compute RMSD against.
            
        output_folder:       
            (Optional) Path to the output folder
    """

    from deep_cartograph.modules.common import validate_configuration, save_data
    from deep_cartograph.modules.figures import plot_data
    from deep_cartograph.modules.md import RMSD, RMSF, dRMSD
    
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

    # For each type of analysis
    for category, analyses in configuration['analysis'].items():
        
        # If there is any requested
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
                
                # For each trajectory and topology
                for trajectory, topology in zip(trajectories, topologies):
                    trajectory_name = Path(trajectory).stem
                    selection = params['selection']
                    fit_selection = params.get('fit_selection')
                    selection_stride = params.get('selection_stride', 1)
                    
                    if category == 'RMSD':
                        # Use ref_topologies if they exist, otherwise use the first frame
                        refs_to_run = ref_topologies if ref_topologies else [None]
                        for ref_pdb in refs_to_run:
                            logger.info(f"   - Processing trajectory: {trajectory_name} with reference: {ref_pdb if ref_pdb else 'first frame'}")
                            ref_label = f"_to_{Path(ref_pdb).stem}" if ref_pdb else "first_frame"
                            traj_key = trajectory_name + ref_label
                            
                            y_data[traj_key] = RMSD(trajectory, topology, selection, fit_selection, ref_pdb)
                            x_data[traj_key] = np.arange(0, len(y_data[traj_key])) * dt_per_frame
                            x_label = 'Time (ns)'
                    elif category == 'RMSF':
                        y_data[trajectory_name], x_data[trajectory_name] = RMSF(trajectory, topology, selection, fit_selection)
                        x_label = 'Residue'
                    elif category == 'dRMSD':
                        # Use ref_topologies if they exist, otherwise use the topology
                        refs_to_run = ref_topologies if ref_topologies else [topology]
                        for ref_pdb in refs_to_run:
                            logger.info(f"   - Processing trajectory: {trajectory_name} with reference: {ref_pdb}")
                            ref_label = f"_to_{Path(ref_pdb).stem}"
                            traj_key = trajectory_name + ref_label
                            
                            dRMSD_output_folder = os.path.join(output_folder, f'dRMSD_temp_{traj_key}')
                            os.makedirs(dRMSD_output_folder, exist_ok=True)
                            
                            y_data[traj_key] = dRMSD(trajectory, topology, selection, selection_stride, ref_pdb, dRMSD_output_folder)
                            x_data[traj_key] = np.arange(0, len(y_data[traj_key])) * dt_per_frame
                            x_label = 'Time (ns)'
                    else:
                        logger.error(f"Unknown analysis category: {category}")
                        continue
                
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
    
    parser.add_argument('-ref_top_data', dest='ref_topology_data', help="(Optional) Path to reference topology or folder with reference topology files to compute RMSD against.", required=False, default=None)
    parser.add_argument('-output', dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Determine output folder
    output_folder = args.output_folder if args.output_folder else 'analyze_geometry'
    os.makedirs(output_folder, exist_ok=True)

    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)
    
    # Read configuration
    configuration = read_configuration(args.configuration_path)
    
    # Check main input folders
    trajectories, topologies = check_data(args.trajectory_data, args.topology_data)
    
    # If reference topology data is provided
    ref_topologies = None
    if args.ref_topology_data:
        ref_topologies, _ = check_data(args.ref_topology_data, args.ref_topology_data)

    # Run Analyze Geometry tool
    _ = analyze_geometry(
        configuration = configuration, 
        trajectories = trajectories,
        topologies = topologies,
        ref_topologies = ref_topologies,
        output_folder = output_folder)

if __name__ == "__main__":

    main()