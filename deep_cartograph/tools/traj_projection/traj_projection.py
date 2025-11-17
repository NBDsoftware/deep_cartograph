# Import modules
import os
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Optional

from deep_cartograph.tools.traj_projection.traj_projection_workflow import TrajProjectionWorkflow
from deep_cartograph.modules.common import (
    get_unique_path, 
    read_configuration
)

########
# TOOL #
########

def traj_projection(
    configuration: Dict,
    colvars_paths: List[str],
    topologies: List[str] = None,
    trajectory_names: List[str] = None,
    model_paths: List[str] = None,
    model_traj_paths: Optional[List[List[str]]] = None,
    output_folder: Optional[str] = 'traj_projection'
) -> Dict[str, List[str]]:
    """
    Projection of trajectories onto pre-trained collective variables (CVs).
    The trajectories are provided as colvars files containing the time series of features.
    The CVs are pre-trained models that can be loaded from files.

    Parameters
    ----------
    configuration : Dict
        Configuration dictionary (see `default_config.yml` for more information).

    colvars_paths : List[str]
        List of paths to the colvars files containing the input data from new trajectories to project (samples of features).

    topologies : List[str]
        List of paths to topologies of new trajectories to project
        
    trajectory_names : List[str]
        List of trajectory names corresponding to the input colvars files.
    
    model_paths : List[str]
        List of paths to the pre-trained collective variable model files.
    
    model_traj_paths : Optional[List[List[str]]]
        List of paths to the projected trajectory data used to train the collective variable model(s). These will
        be used to compute the background FES.
    
    output_folder : Optional[str]
        Path to the output folder where results will be saved.
        
    Returns
    -------

    Dict[str, List[str]]
        A dictionary where keys are the names of the cv models from model_paths and values are lists of paths to
        the trajectories in the CV space for each colvars file.
    """
    
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================================================================")
    logger.info("Projection of trajectories onto pre-trained collective variables.")
    logger.info("================================================================")

    # Start timer
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a TrajProjectionWorkflow object 
    workflow = TrajProjectionWorkflow(
        configuration=configuration,
        colvars_paths=colvars_paths,
        topologies=topologies,
        trajectory_names=trajectory_names,
        model_paths=model_paths,
        model_traj_paths=model_traj_paths,
        output_folder=output_folder
    )

    # Run the workflow
    output_paths = workflow.run()

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Project trajectories): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return output_paths

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
        prog="Deep Cartograph:  Trajectory Projection",
        description=("Projection of trajectories onto pre-trained collective variables."
        )
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-colvars', '-colvars_files', dest='colvars_path', type=str, nargs='*', default=None, required=True,
        help="Path to the colvars file(s) containing the input data (samples of features)."
    )
    parser.add_argument(
        '-top', '-topology', dest='topologies', type=str, nargs='*', default=None, required=False,
        help="Path to the topology file(s) corresponding to the input colvars file(s)."
    )
    parser.add_argument(
        '-names', '-trajectory_names', dest='trajectory_names', type=str, nargs='*', default=None, required=False,
        help="Names of the trajectories corresponding to the input colvars file(s)."
    )
    parser.add_argument(
        '-models', '-cvs_models', dest='model_paths', type=str, nargs='*', default=None, required=True,
        help="Path to the pre-trained collective variable model file(s)."
    )
    parser.add_argument(
        '-models_traj', '-cvs_models_traj', dest='model_traj_paths', type=str, nargs='*', default=None, required=True,
        help="Path to the projected data used for training the collective variable model(s). These will be used to compute the background FES."
    )
    
    # Optional args
    parser.add_argument(
        '-out', '-output', dest='output_folder', required=False,
        help="Path to the output folder"
    )
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', required=False,
        help="Set the logging level to DEBUG."
    )

    return parser.parse_args()

########
# MAIN #
########

def main():

    args = parse_arguments()

    # Create new output folder
    output_folder = args.output_folder if args.output_folder else 'traj_projection'
    output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run Traj Projection tool
    traj_projection(
        configuration = configuration,
        colvars_paths = args.colvars_path,
        topologies = args.topologies,
        trajectory_names = args.trajectory_names,
        model_paths = args.model_paths,
        model_traj_paths = args.model_traj_paths,
        output_folder = output_folder)
    
if __name__ == "__main__":

    main()
    