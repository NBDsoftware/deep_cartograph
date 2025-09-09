# Import modules
import os
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional

from deep_cartograph.tools.train_colvars.train_colvars_workflow import TrainColvarsWorkflow
from deep_cartograph.modules.common import (
    get_unique_path, 
    read_configuration, 
    read_feature_constraints
)

########
# TOOL #
########

def train_colvars(
    configuration: Dict,
    colvars_paths: Union[str, List[str]],
    trajectories: Optional[List[str]] = None,
    topologies: Optional[List[str]] = None,
    reference_topology: Optional[str] = None,
    feature_constraints: Optional[Union[List[str], str]] = None,
    sup_colvars_paths: Optional[List[str]] = None,
    sup_topology_paths: Optional[List[str]] = None,
    sup_labels: Optional[List[str]] = None,
    dimension: Optional[int] = None,
    cvs: Optional[List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']]] = None,
    samples_per_frame: Optional[float] = 1,
    output_folder: str = 'train_colvars'
) -> None:
    """
    Trains collective variables using the mlcolvar library and computes a Free Energy Surface (FES) 
    along the CVs from the trajectory data. 

    Supported collective variables (CVs):
        - pca (Principal Component Analysis)
        - ae (Autoencoder)
        - tica (Time Independent Component Analysis)
        - htica (Hierarchical Time Independent Component Analysis)
        - deep_tica (Deep Time Independent Component Analysis)

    Parameters
    ----------
    configuration : Dict
        Configuration dictionary (see `default_config.yml` for more information).

    colvars_paths : str or List[str]
        Path or list of paths to colvars files containing the input data (samples of features).

    trajectories : Optional[List[str]], default=None
        Path to the trajectory files corresponding to the colvars files (same order as colvars_paths).

    topologies : Optional[List[str]], default=None
        Path to the topology files corresponding to the trajectory files (same order as trajectories).

    reference_topology : Optional[str], default=None
        Path to the reference topology file. If `None`, the first topology file is used as reference.

    feature_constraints : Optional[Union[List[str], str]], default=None
        List of features to use for training, or a string with regex to filter feature names.  
        If `None`, all features except `*labels`, `time`, `*bias`, and `*walker` are used.

    sup_colvars_paths : Optional[List[str]], default=None
        List of paths to supplementary colvars files (e.g., experimental structures).  
        If `None`, no supplementary data is used.

    sup_topology_paths : Optional[List[str]], default=None
        List of paths to topology files corresponding to the supplementary colvars files.
        
    sup_labels : Optional[List[str]], default=None
        List of labels to identify the supplementary data.  
        If `None`, the supplementary data is identified as 'supplementary data i'.

    dimension : Optional[int], default=None
        Dimension of the CVs to train or compute. If `None`, the value in the configuration is used.

    cvs : Optional[List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']]], default=None
        List of collective variables to train or compute. If `None`, the ones in the configuration are used.

    samples_per_frame : Optional[float], default=1
        Number of samples in the colvars file for each frame in the trajectory file.  
        Calculated with: `samples_per_frame = (trajectory saving frequency) / (colvars saving frequency)`.

    output_folder : str, default='train_colvars'
        Path to the output folder where the output files will be saved.  
        If not provided, a folder named 'train_colvars' is created.

    Returns
    -------
    None
        This function does not return a value. It saves output files to the specified folder.
    """
    
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================================")
    logger.info("Training of Collective Variables")
    logger.info("================================")
    logger.info("Training of collective variables using the mlcolvar library.")

    # Start timer
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]
    
    # Create a TrainColvarsWorkflow object 
    workflow = TrainColvarsWorkflow(
        configuration=configuration,
        train_colvars_paths=colvars_paths,
        train_trajectory_paths=trajectories,
        train_topology_paths=topologies,
        ref_topology_path=reference_topology,
        feature_constraints=feature_constraints,
        sup_colvars_paths=sup_colvars_paths,
        sup_topology_paths=sup_topology_paths,
        sup_labels=sup_labels,
        cv_dimension=dimension,
        cvs=cvs,
        samples_per_frame=samples_per_frame,
        output_folder=output_folder
    )
        
    # Run the workflow
    workflow.run()
    
    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Train colvars): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

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
        prog="Deep Cartograph:  Train Collective Variables",
        description=("Train collective variables using the mlcolvar library."
        )
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-colvars', dest='colvars_path', type=str, required=True,
        help="Path to the input colvars file."
    )
    
    # Optional arguments
    parser.add_argument(
        '-trajectory', dest='trajectory', type=str, required=False,
        help=("Path to trajectory file corresponding to the colvars file." 
              "The feature samples in the colvars file must correspond to frames of this trajectory." 
              "Used to create structure clusters."
        )
    )
    parser.add_argument(
        '-topology', dest='topology', type=str, required=False,
        help="Path to topology file of the trajectory."
    )
    parser.add_argument(
        '-reference_topology', dest='reference_topology', type=str, required=False,
        help="Path to reference topology file. If None, the first topology file is used as reference."
    )
    parser.add_argument(
        '-samples_per_frame', dest='samples_per_frame', type=float, required=False,
        help=("Samples in the colvars file for each frame in the trajectory file." 
              "Calculated with: samples_per_frame = (trajectory saving frequency)/(colvars saving frequency)."
        )
    )
    parser.add_argument(
        '-sup_colvars', dest='sup_colvars_path', type=str, required=False,
        help=("Path to colvars file with supplementary data to project alongside" 
              "the FES of the training data (e.g. experimental structures). If None, no supplementary data is used"
        )
    )
    parser.add_argument(
        '-sup_topology', dest='sup_topology_path', type=str, required=False,
        help="Path to topology file of the supplementary colvars file."
    )
    parser.add_argument(
        '-features_path', type=str, required=False,
        help="Path to a file containing the list of features that should be used (these are used if the path is given)"
    )
    parser.add_argument(
        '-features_regex', type=str, required=False,
        help="Regex to filter the features (features_path is prioritized over this, mutually exclusive)"
    )
    parser.add_argument(
        '-dim', '-dimension', dest='dimension', type=int, required=False,
        help="Dimension of the CV to train or compute"
    )
    parser.add_argument(
        '-cvs', nargs='+', required=False,
        help="Collective variables to train or compute (pca, ae, tica, htica, deep_tica)"
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

    # Create new output folder
    output_folder = args.output_folder if args.output_folder else 'train_colvars'
    output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Read features to use
    feature_constraints = read_feature_constraints(args.features_path, args.features_regex)

    # Reference data should be list or None - see train_colvars API
    sup_labels = None
    sup_colvars_paths = None
    sup_topology_paths = None
    if args.sup_colvars_path:
        sup_labels = [Path(args.sup_colvars_path).stem]
        sup_colvars_paths = [args.sup_colvars_path]
    if args.sup_topology_path:
        sup_topology_paths = [args.sup_topology_path]

    # Trajectories should be list or None - see train_colvars API
    trajectories = None
    if args.trajectory:
        trajectories = [args.trajectory]
        
    # Topologies should be list or None - see train_colvars API
    topologies = None
    if args.topology:
        topologies = [args.topology]
        
    # Run Train Colvars tool
    train_colvars(
        configuration = configuration,
        colvars_paths = args.colvars_path,
        trajectories = trajectories,
        topologies = topologies,
        reference_topology = args.reference_topology,
        feature_constraints = feature_constraints,
        sup_colvars_paths = sup_colvars_paths,
        sup_topology_paths = sup_topology_paths,
        sup_labels = sup_labels,
        dimension = args.dimension,
        cvs = args.cvs,
        samples_per_frame = args.samples_per_frame,
        output_folder = output_folder)
    
if __name__ == "__main__":

    main()
    