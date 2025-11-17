# Import modules
import os
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Union, Optional

from deep_cartograph.tools.traj_cluster.traj_cluster_workflow import TrajClusterWorkflow
from deep_cartograph.modules.common import (
    get_unique_path, 
    read_configuration
)

########
# TOOL #
########
def traj_cluster(
    configuration: Dict,
    cv_traj_paths: Union[str, List[str]],
    trajectories: Optional[List[str]] = None,
    topologies: Optional[List[str]] = None,
    sup_cv_traj_paths: Optional[List[str]] = None,
    sup_trajectories: Optional[List[str]] = None,
    sup_topologies: Optional[List[str]] = None,
    frames_per_sample: Optional[int] = 1,
    output_folder: str = 'traj_cluster'
) -> None:
    """
    Clusters frames from the trajectories based on the value of the collective variable. 
    Clusters are found using all the frames from the trajectories provided in `cv_traj_paths`.
    Optionally, supplementary trajectories can be projected onto the same clusters using `sup_cv_traj_paths`.
    
    It is assumed that each trajectory in `cv_traj_paths` has a corresponding trajectory in `trajectories` and `topologies`.
    The same applies to `sup_cv_traj_paths`, `sup_trajectories`, and `sup_topologies`.

    Parameters
    ----------
    configuration : Dict
        Configuration dictionary (see `default_config.yml` for more information).
        
    cv_traj_paths : str or List[str]
        Path or list of paths to csv files containing the collective variable trajectories.

    trajectories : Optional[List[str]], default=None
        Path to the trajectory files corresponding to the collective variable trajectories (same order as cv_traj_paths).

    topologies : Optional[List[str]], default=None
        Path to the topology files corresponding to the trajectory files (same order as trajectories).

    sup_cv_traj_paths : Optional[List[str]], default=None
        List of paths to csv files containing the collective variable supplementary trajectories.
        If `None`, no supplementary data is used.

    sup_trajectories : Optional[List[str]], default=None
        List of paths to trajectory files corresponding to the supplementary collective variable trajectories (same order as sup_cv_traj_paths).

    sup_topologies : Optional[List[str]], default=None
        List of paths to topology files corresponding to the supplementary trajectory files (same order as sup_trajectories).

    frames_per_sample : Optional[float], default=1
        Frames in the collective variable trajectory file for each frame in the trajectory file.

    output_folder : str, default='traj_cluster'
        Path to the output folder where the output files will be saved.  
        If not provided, a folder named 'traj_cluster' is created.

    Returns
    -------
    None
        This function does not return a value. It saves output files to the specified folder.
    """
    
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("=====================================")
    logger.info("Trajectory clustering on the CV space")
    logger.info("=====================================")
    logger.info("Clustering of trajectories using sklearn.")

    # Start timer
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    if isinstance(cv_traj_paths, str):
        cv_traj_paths = [cv_traj_paths]

    # Create a TrajClusterWorkflow object
    workflow = TrajClusterWorkflow(
        configuration=configuration,
        cv_traj_paths=cv_traj_paths,
        trajectories=trajectories,
        topologies=topologies,
        sup_cv_traj_paths=sup_cv_traj_paths,
        sup_trajectories=sup_trajectories,
        sup_topologies=sup_topologies,
        frames_per_sample=frames_per_sample,
        output_folder=output_folder
    )
        
    # Run the workflow if clustering is enabled in the configuration
    output_paths = workflow.run()
    
    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Train colvars): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

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
    logger.info("Deep Cartograph: package for projecting and clustering trajectories using collective variables.")
    
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="Deep Cartograph:  Trajectory clustering on the CV space",
        description=("Clusters frames from the trajectories based on the value of the collective variable."
        )
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-cv_traj', '-cv_trajectory', dest='cv_traj_path', type=str, required=True,
        help="Path to the input collective variable trajectory file."
    )
    
    # Optional arguments
    parser.add_argument(
        '-trajectory', dest='trajectory', type=str, required=False,
        help=("Path to trajectory file corresponding to the collective variable trajectory file." 
              "The values of the collective variable must correspond to frames of this trajectory." 
              "Used to create structure clusters."
        )
    )
    parser.add_argument(
        '-topology', dest='topology', type=str, required=False,
        help="Path to topology file of the trajectory."
    )
    parser.add_argument(
        '-sup_cv_traj', '-sup_cv_trajectory', dest='sup_cv_traj_path', type=str, required=True,
        help="Path to the input supplementary cv trajectory file."
    )
    parser.add_argument(
        '-sup_trajectory', dest='sup_trajectory_path', type=str, required=False,
        help="Path to trajectory file of the supplementary cv trajectory file."
    )
    parser.add_argument(
        '-sup_topology', dest='sup_topology_path', type=str, required=False,
        help="Path to topology file of the supplementary cv trajectory file."
    )
    parser.add_argument(
        '-frames_per_sample', dest='frames_per_sample', type=int, required=False,
        help="Frames in the trajectory for each sample in the cv trajectory."
    )
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

    # Create a new output folder
    output_folder = args.output_folder if args.output_folder else 'traj_cluster'
    output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Trajectories should be list or None - see traj_cluster API
    trajectories = None
    if args.trajectory:
        trajectories = [args.trajectory]
        
    # Topologies should be list or None - see traj_cluster API
    topologies = None
    if args.topology:
        topologies = [args.topology]
        
    # Supplementary cv trajectories should be list or None - see traj_cluster API
    sup_cv_traj_paths = None
    if args.sup_cv_traj_path:
        sup_cv_traj_paths = [args.sup_cv_traj_path]
    
    # Supplementary trajectories should be list or None - see traj_cluster API
    sup_trajectories = None
    if args.sup_trajectory_path:
        sup_trajectories = [args.sup_trajectory_path]   
    
    # Supplementary topologies should be list or None - see traj_cluster API
    sup_topologies = None
    if args.sup_topology_path:
        sup_topologies = [args.sup_topology_path]

    # Run Trajectory Clustering tool
    traj_cluster(
        configuration = configuration,
        trajectories = trajectories,
        topologies = topologies,
        sup_cv_traj_paths = sup_cv_traj_paths,
        sup_trajectories = sup_trajectories,
        sup_topologies = sup_topologies,
        frames_per_sample = args.frames_per_sample,
        output_folder = output_folder)
    
if __name__ == "__main__":

    main()
    