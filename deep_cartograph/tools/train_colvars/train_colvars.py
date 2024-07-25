# Import modules
import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Union
from mlcolvar.utils.io import  create_dataset_from_files  

########
# TOOL #
########

def train_colvars(configuration: Dict, colvars_path: str, feature_constraints: Union[List[str], str], 
                  ref_colvars_path: List[str] = None, ref_labels: List[str] = None, 
                  dimension: int = None, cvs: List[Literal['pca', 'ae', 'tica', 'dtica']] = None, 
                  output_folder: str = 'train_colvars'):
    """
    Function that trains collective variables using the mlcolvar library. 

    The following CVs can be computed: 

        - pca (Principal Component Analysis) 
        - ae (Autoencoder)
        - tica (Time Independent Component Analysis)
        - dtica (Deep Time Independent Component Analysis)

    It also plots an estimate of the Free Energy Surface (FES) along the CVs from the trajectory data.

    Parameters
    ----------

        configuration:       configuration dictionary (see default_config.yml for more information)
        colvars_path:        path to the colvars file with the input data (samples of features)
        feature_constraints: list with the features to use for the training | str with regex to filter feature names. If None, all features but *labels, time, *bias and *walker are used from the colvars file
        ref_colvars_path:    list of paths to colvars files with reference data. If None, no reference data is used
        ref_labels:          list of labels to identify the reference data. If None, the reference data is identified as 'reference data i'
        dimension:           dimension of the CVs to train or compute, if None, the value in the configuration file is used
        cvs:                 List of collective variables to train or compute (pca, ae, tica, dtica), if None, the ones in the configuration file are used
        output_folder:       path to folder where the output files are saved, if not given, a folder named 'output' is created
    """

    from deep_cartograph.modules.common import create_output_folder, validate_configuration, files_exist, get_filter_dict, merge_configurations
    from deep_cartograph.tools.train_colvars.utils import compute_pca, compute_ae, compute_tica, compute_deep_tica
    from deep_cartograph.yaml_schemas.train_colvars import TrainColvars

    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================================")
    logger.info("Training of Collective Variables")
    logger.info("================================")
    logger.info("Training of collective variables using the mlcolvar library.")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    create_output_folder(output_folder)
    
    # Validate configuration
    configuration = validate_configuration(configuration, TrainColvars, output_folder)

    # Check if files exist
    if not files_exist(colvars_path):
        logger.error(f"Colvars file {colvars_path} does not exist. Exiting...")
        sys.exit(1)
    if ref_colvars_path is not None:
        if not files_exist(*ref_colvars_path):
            logger.error(f"Reference colvars file {ref_colvars_path} does not exist. Exiting...")
            sys.exit(1)


    ###############
    # PREPARATION #
    ###############

    # Enforce CLI arguments if any
    if cvs:
        configuration['cvs']= cvs
    if dimension:
        configuration['common']['dimension'] = dimension

    logger.info('Creating datasets from colvars...')

    # Create feature filter dictionary from feature constraints
    filter_dict = get_filter_dict(feature_constraints)

    # Check if the colvars file exists
    if not files_exist(colvars_path):
        return 

    # Build dataset from colvars file with the selected features
    features_dataset, colvars_dataframe = create_dataset_from_files(file_names=[colvars_path], filter_args=filter_dict, verbose = False, return_dataframe=True)  

    # Filter dataframe to keep just the selected features
    features_dataframe = colvars_dataframe.filter(**filter_dict)

    # If reference data is given
    if ref_colvars_path:

        ref_features_dataset = []
        ref_features_dataframe = []

        for path in ref_colvars_path:

            # Build dataset from colvars file with the selected features
            ref_features_dataset_i, ref_colvars_dataframe_i = create_dataset_from_files(file_names=[path], filter_args=filter_dict, verbose = False, return_dataframe=True)     
            # Filter dataframe to keep just the selected features
            ref_features_dataframe_i = ref_colvars_dataframe_i.filter(**filter_dict)

            ref_features_dataset.append(ref_features_dataset_i)
            ref_features_dataframe.append(ref_features_dataframe_i)
    else:
        ref_features_dataset = None
        ref_features_dataframe = None

    # Log number of features and samples
    logger.info(f'Number of samples: {features_dataframe.shape[0]}')
    logger.info(f'Number of features: {features_dataframe.shape[1]}')

    ###########
    # CV: PCA #
    ###########

    if 'pca' in configuration['cvs']:
        pca_output_path = os.path.join(output_folder, 'pca')

        # Merge common and pca configurations
        pca_configuration = merge_configurations(configuration['common'], configuration.get('pca',{}))

        compute_pca(features_dataframe = features_dataframe, 
                    ref_features_dataframe = ref_features_dataframe,
                    ref_labels = ref_labels,
                    cv_settings = pca_configuration, 
                    figures_settings = configuration['figures'], 
                    clustering_settings = configuration['clustering'],
                    output_path = pca_output_path)

    ###################
    # CV: Autoencoder #
    ###################

    if 'ae' in configuration['cvs']:
        ae_output_path = os.path.join(output_folder, 'ae')

        # Merge common and ae configurations
        ae_configuration = merge_configurations(configuration['common'], configuration.get('ae',{}))

        compute_ae(features_dataset = features_dataset, 
                   ref_features_dataset = ref_features_dataset,
                   ref_labels = ref_labels,
                   cv_settings = ae_configuration,
                   figures_settings = configuration['figures'],
                   clustering_settings = configuration['clustering'],
                   output_path = ae_output_path)

    ############
    # CV: TICA #
    ############

    if 'tica' in configuration['cvs']:
        tica_output_path = os.path.join(output_folder, 'tica')

        # Merge common and tica configurations
        tica_configuration = merge_configurations(configuration['common'], configuration.get('tica',{}))

        compute_tica(features_dataframe = features_dataframe,
                     ref_features_dataframe = ref_features_dataframe,
                     ref_labels = ref_labels,
                     cv_settings = tica_configuration,
                     figures_settings = configuration['figures'],
                     clustering_settings = configuration['clustering'],
                     output_path = tica_output_path)
    
    #################
    # CV: Deep-TICA #
    #################

    if 'dtica' in configuration['cvs']:
        deep_tica_output_path = os.path.join(output_folder, 'deep_tica')

        # Merge common and deep tica configurations
        deep_tica_configuration = merge_configurations(configuration['common'], configuration.get('dtica',{}))

        compute_deep_tica(features_dataframe = features_dataframe,
                          ref_features_dataframe = ref_features_dataframe,
                          ref_labels = ref_labels,
                          cv_settings = deep_tica_configuration,
                          figures_settings = configuration['figures'],
                          clustering_settings = configuration['clustering'],
                          output_path = deep_tica_output_path)
    
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

if __name__ == "__main__":

    from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration, read_feature_constraints

    parser = argparse.ArgumentParser("Deep Cartograph: Train Collective Variables", description="Train collective variables using the mlcolvar library.")

    parser.add_argument('-conf', '-configuration', dest='configuration_path', type=str, help='Path to configuration file (.yml)', required=True)
    parser.add_argument('-colvars', dest='colvars_path', type=str, help='Path to the colvars file', required=True)
    parser.add_argument('-ref_colvars', dest='ref_colvars_path', type=str, help='Path to the colvars file with the reference data', required=False)
    parser.add_argument('-use_rl', '-use_reference_lab', dest='use_reference_labels', action='store_true', help="Use labels for reference data (names of the files in the reference folder)", default=False)
    parser.add_argument('-features_path', type=str, help='Path to a file containing the list of features that should be used (these are used if the path is given)', required=False)
    parser.add_argument('-features_regex', type=str, help='Regex to filter the features (features_path is prioritized over this, mutually exclusive)', required=False)
    parser.add_argument('-dim', '-dimension', type=int, help='Dimension of the CV to train or compute', required=False)
    parser.add_argument('-cvs', nargs='+', help='Collective variables to train or compute (pca, ae, tica, dtica)', required=False)
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
            
    # Run tool
    train_colvars(
        configuration = configuration, 
        colvars_path = args.colvars_path, 
        feature_constraints = feature_constraints, 
        ref_colvars_path = ref_colvars_path,
        ref_labels = ref_labels,
        dimension = args.dimension, 
        cvs = args.cvs, 
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))