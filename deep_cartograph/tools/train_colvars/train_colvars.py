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

def train_colvars(configuration: Dict, colvars_paths: Union[str, List[str]], feature_constraints: Union[List[str], str, None] = None, 
                  sup_colvars_paths: Union[List[str], None] = None, sup_labels: Union[List[str], None] = None, dimension: Union[int, None] = None, 
                  cvs: Union[List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']], None] = None, trajectories: Union[List[str], None] = None, 
                  topologies: Union[List[str], None] = None, reference_topology: Union[str, None] = None, samples_per_frame: Union[float, None] = 1, 
                  output_folder: str = 'train_colvars'):
    """
    Function that trains collective variables using the mlcolvar library. 

    The following CVs can be computed: 

        - pca (Principal Component Analysis) 
        - ae (Autoencoder)
        - tica (Time Independent Component Analysis)
        - htica (Hierarchical Time Independent Component Analysis)
        - deep_tica (Deep Time Independent Component Analysis)

    It also plots an estimate of the Free Energy Surface (FES) along the CVs from the trajectory data.

    Parameters
    ----------

        configuration       
            Configuration dictionary (see default_config.yml for more information)
            
        colvars_paths       
            Path or list of paths to the colvars files with the input data (samples of features)
            
        feature_constraints (Optional)
            List with the features to use for the training | str with regex to filter feature names. If None, all features but *labels, time, *bias and *walker are used from the colvars file
            
        sup_colvars_paths (Optional)   
            List of paths to colvars files with supplementary data to project alongside the FES of the training data (e.g. experimental structures). If None, no supplementary data is used

        sup_labels (Optional)          
            List of labels to identify the reference data. If None, the reference data is identified as 'reference data i'
            
        cv_dimension (Optional)        
            Dimension of the CVs to train or compute, if None, the value in the configuration file is used
            
        cvs (Optional)                 
            List of collective variables to train or compute (pca, ae, tica, htica, deep_tica), if None, the ones in the configuration file are used
            
        trajectories (Optional)        
            Path to the trajectory files corresponding to the colvars files (same order).
            
        topologies (Optional)          
            Path to the topology files corresponding to the trajectory files (same order).
            
        reference_topology (Optional)
            Path to the reference topology file. If None, the first topology file is used as reference topology
            
        samples_per_frame (Optional)   
            Samples in the colvars file for each frame in the trajectory file. Calculated with: samples_per_frame = (trajectory saving frequency)/(colvars saving frequency)
            
        output_folder (Optional)       
            Path to folder where the output files are saved, if not given, a folder named 'output' is created
    """
    
    from deep_cartograph.modules.common import create_output_folder
    
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================================")
    logger.info("Training of Collective Variables")
    logger.info("================================")
    logger.info("Training of collective variables using the mlcolvar library.")

    # Start timer
    start_time = time.time()
    
    # Create output directory
    create_output_folder(output_folder)

    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]
    
    # Create a TrainColvarsWorkflow object 
    workflow = TrainColvarsWorkflow(
        configuration=configuration,
        training_colvars_paths=colvars_paths,
        feature_constraints=feature_constraints,
        sup_colvars_paths=sup_colvars_paths,
        sup_labels=sup_labels,
        cv_dimension=dimension,
        cvs=cvs,
        trajectory_paths=trajectories,
        topology_paths=topologies,
        ref_topology_path=reference_topology,
        samples_per_frame=samples_per_frame,
        output_folder=output_folder
    )
        
    # Run the workflow
    workflow.run()
    
    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Train colvars): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

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
    
    from deep_cartograph.modules.common import get_unique_path, read_configuration, read_feature_constraints

    parser = argparse.ArgumentParser("Deep Cartograph: Train Collective Variables", description="Train collective variables using the mlcolvar library.")

    parser.add_argument('-conf', '-configuration', dest='configuration_path', type=str, help='Path to configuration file (.yml)', required=True)
    parser.add_argument('-colvars', dest='colvars_path', type=str, help='Path to the colvars file with feature samples.', required=True)
    parser.add_argument('-trajectory', dest='trajectory', help="""Path to trajectory file corresponding to the colvars file. The feature samples in the 
                        colvars file must correspond to frames of this trajectory. Used to create structure clusters.""", required=False)
    parser.add_argument('-topology', dest='topology', help="Path to topology file of the trajectory.", required=False)
    parser.add_argument('-samples_per_frame', dest='samples_per_frame', type=float, help="""Samples in the colvars file for each frame in the trajectory file. 
                        Calculated with: samples_per_frame = (trajectory saving frequency)/(colvars saving frequency).""", required=False)
    parser.add_argument('-sup_colvars', dest='sup_colvars_paths', type=str, help="""List of paths to colvars files with supplementary data to project alongside 
                        the FES of the training data (e.g. experimental structures). If None, no supplementary data is used""", required=False)
    parser.add_argument('-features_path', type=str, help='Path to a file containing the list of features that should be used (these are used if the path is given)', required=False)
    parser.add_argument('-features_regex', type=str, help='Regex to filter the features (features_path is prioritized over this, mutually exclusive)', required=False)
    parser.add_argument('-dim', '-dimension', dest='dimension', type=int, help='Dimension of the CV to train or compute', required=False)
    parser.add_argument('-cvs', nargs='+', help='Collective variables to train or compute (pca, ae, tica, htica, deep_tica)', required=False)
    parser.add_argument('-out', '-output', dest='output_folder', help='Path to the output folder', required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Set the logging level to DEBUG', default=False)

    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Read features to use
    feature_constraints = read_feature_constraints(args.features_path, args.features_regex)

    # Reference data should be list or None - see train_colvars API
    sup_labels = None
    sup_colvars_paths = None
    if args.sup_colvars_paths:
        sup_labels = [Path(args.sup_colvars_paths).stem]
        sup_colvars_paths = [args.sup_colvars_paths]
            
    # Trajectories should be list or None - see train_colvars API
    trajectories = None
    if args.trajectory:
        trajectories = [args.trajectory]
        
    # Topologies should be list or None - see train_colvars API
    topologies = None
    if args.topology:
        topologies = [args.topology]
    
    # Give value to output_folder
    if args.output_folder is None:
        output_folder = 'train_colvars'
    else:
        output_folder = args.output_folder
    
    # If called through command line, create a unique output folder
    output_folder = get_unique_path(output_folder)
        
    # Create a TrainColvarsWorkflow object and run the workflow
    train_colvars(
        configuration = configuration,
        colvars_paths = args.colvars_path,
        feature_constraints = feature_constraints,
        sup_colvars_paths = sup_colvars_paths,
        sup_labels = sup_labels,
        dimension = args.dimension,
        cvs = args.cvs,
        trajectories = trajectories,
        topologies = topologies,
        samples_per_frame = args.samples_per_frame,
        output_folder = output_folder)
    
    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))
    
if __name__ == "__main__":

    main()
    