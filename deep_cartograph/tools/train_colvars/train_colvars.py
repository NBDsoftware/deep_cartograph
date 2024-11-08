# Import modules
import os
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Union

from deep_cartograph.tools.train_colvars.train_colvars_workflow import TrainColvarsWorkflow

########
# TOOL #
########

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

def train_colvars(configuration: Dict, colvars_path: str, feature_constraints: Union[List[str], str, None] = None, 
                  ref_colvars_path: Union[List[str], None] = None, ref_labels: Union[List[str], None] = None, dimension: Union[int, None] = None, 
                  cvs: Union[List[Literal['pca', 'ae', 'tica', 'dtica']], None] = None, trajectory: Union[str, None] = None, 
                  topology: Union[str, None] = None, samples_per_frame: Union[float, None] = 1, output_folder: str = 'train_colvars'):
    """
    Function that trains collective variables using the mlcolvar library. 

    The following CVs can be computed: 

        - pca (Principal Component Analysis) 
        - ae (Autoencoder)
        - tica (Time Independent Component Analysis)
        - deep_tica (Deep Time Independent Component Analysis)

    It also plots an estimate of the Free Energy Surface (FES) along the CVs from the trajectory data.

    Parameters
    ----------

        configuration:       configuration dictionary (see default_config.yml for more information)
        colvars_path:        path to the colvars file with the input data (samples of features)
        feature_constraints: list with the features to use for the training | str with regex to filter feature names. If None, all features but *labels, time, *bias and *walker are used from the colvars file
        ref_colvars_path:    list of paths to colvars files with reference data. If None, no reference data is used
        ref_labels:          list of labels to identify the reference data. If None, the reference data is identified as 'reference data i'
        cv_dimension:        dimension of the CVs to train or compute, if None, the value in the configuration file is used
        cvs:                 List of collective variables to train or compute (pca, ae, tica, dtica), if None, the ones in the configuration file are used
        trajectory_path:     path to the trajectory file that will be clustered
        topology_path:       path to the topology file of the system
        samples_per_frame:   samples in the colvars file for each frame in the trajectory file. Calculated with: samples_per_frame = (trajectory saving frequency)/(colvars saving frequency)
        output_folder:       path to folder where the output files are saved, if not given, a folder named 'output' is created
    """
    
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================================")
    logger.info("Training of Collective Variables")
    logger.info("================================")
    logger.info("Training of collective variables using the mlcolvar library.")

    # Start timer
    start_time = time.time()
    
    # Create a TrainColvarsWorkflow object 
    workflow = TrainColvarsWorkflow(
        configuration=configuration,
        colvars_path=colvars_path,
        feature_constraints=feature_constraints,
        ref_colvars_path=ref_colvars_path,
        ref_labels=ref_labels,
        cv_dimension=dimension,
        cvs=cvs,
        trajectory_path=trajectory,
        topology_path=topology,
        samples_per_frame=samples_per_frame,
        output_folder=output_folder
    )
        
    # Run the workflow
    workflow.run()
    
    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Train colvars): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

if __name__ == "__main__":

    from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration, read_feature_constraints

    parser = argparse.ArgumentParser("Deep Cartograph: Train Collective Variables", description="Train collective variables using the mlcolvar library.")

    parser.add_argument('-conf', '-configuration', dest='configuration_path', type=str, help='Path to configuration file (.yml)', required=True)
    parser.add_argument('-colvars', dest='colvars_path', type=str, help='Path to the colvars file with feature samples.', required=True)
    parser.add_argument('-trajectory', dest='trajectory', help="""Path to trajectory file corresponding to the colvars file. The feature samples in the 
                        colvars file must correspond to frames of this trajectory. Used to create structure clusters.""", required=False)
    parser.add_argument('-topology', dest='topology', help="Path to topology file of the trajectory.", required=False)
    parser.add_argument('-samples_per_frame', dest='samples_per_frame', type=float, help="""Samples in the colvars file for each frame in the trajectory file. 
                        Calculated with: samples_per_frame = (trajectory saving frequency)/(colvars saving frequency).""", required=False)
    parser.add_argument('-ref_colvars', dest='ref_colvars_path', type=str, help='Path to the colvars file with the reference data', required=False)
    parser.add_argument('-use_rl', '-use_ref_labels', dest='use_reference_labels', action='store_true', help="Use labels for reference data (names of the files in the reference folder)", default=False)
    parser.add_argument('-features_path', type=str, help='Path to a file containing the list of features that should be used (these are used if the path is given)', required=False)
    parser.add_argument('-features_regex', type=str, help='Regex to filter the features (features_path is prioritized over this, mutually exclusive)', required=False)
    parser.add_argument('-dim', '-dimension', type=int, help='Dimension of the CV to train or compute', required=False)
    parser.add_argument('-cvs', nargs='+', help='Collective variables to train or compute (pca, ae, tica, deep_tica)', required=False)
    parser.add_argument('-out', '-output', dest='output_folder', help='Path to the output folder', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Set the logging level to DEBUG', default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Create unique output directory
    output_folder = get_unique_path(args.output_folder)
    create_output_folder(output_folder)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Read features to use
    feature_constraints = read_feature_constraints(args.features_path, args.features_regex)

    # Reference data should be list or None - see train_colvars API
    ref_labels = None
    if args.ref_colvars_path:
        ref_colvars_path = [args.ref_colvars_path]
        if args.use_reference_labels:
            ref_labels = [Path(args.ref_colvars_path).stem]
            
    # Create a TrainColvarsWorkflow object and run the workflow
    train_colvars(
        configuration = configuration,
        colvars_path = args.colvars_path,
        feature_constraints = feature_constraints,
        ref_colvars_path = ref_colvars_path,
        ref_labels = ref_labels,
        cv_dimension = args.dimension,
        cvs = args.cvs,
        trajectory_path = args.trajectory,
        topology_path = args.topology,
        samples_per_frame = args.samples_per_frame,
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))