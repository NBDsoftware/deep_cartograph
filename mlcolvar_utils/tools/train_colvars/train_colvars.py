import os
import time
import argparse
import logging.config
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger

#logger = CSVLogger(save_dir="experiments",   # directory where to save file
#                    name='myCV',             # name of experiment
#                    version=None             # version number (if None it will be automatically assigned)
#                    )

# assign callback to trainer
#trainer = lightning.Trainer(callbacks=[logger])

# Import mlcolvars 
from mlcolvar.utils.io import  create_dataset_from_files    

# Import local modules
from mlcolvar_utils.modules.common import common
from mlcolvar_utils.tools.train_colvars.utils import compute_pca, compute_ae, compute_tica, compute_deep_tica


########
# MAIN #
########

def train_cvs(configuration_path: str, colvars_path: str, features_path: str, cv_dimension: int, cv_type: str, output_folder: str):
    """
    Function that trains collective variables using the mlcolvar library. 

    The following CVs can be computed: 

        - PCA (Principal Component Analysis) 
        - AE (Autoencoder)
        - TICA (Time Independent Component Analysis)
        - DTICA (Deep Time Independent Component Analysis)

    It also plots an estimate of the Free Energy Surface (FES) along the CVs from the trajectory data.

    Inputs
    ------

        configuration_path: path to the configuration file
        colvars_path:  path to the colvars file with the input data
        features_path: path to a file containing a list of features that should be used (if no path is given, all features are used)
        cv_dimension:  dimension of the CVs to train
        cv_type:       type of CV to train (PCA, AE, TICA, DTICA, ALL)
        output_path:   path where the output files are saved
    """

    # Title
    logger.info("Training of Collective Variables\n")

    # Start timer
    start_time = time.time()

    # Create output directory
    common.create_output_folder(output_folder)

    ###############
    # PREPARATION #
    ###############

    # Get parameters and paths
    global_parameters = common.get_global_parameters(configuration_path)

    # Enforce CLI arguments if any
    if cv_type:
        global_parameters['cv']['type'] = cv_type
    if cv_dimension:
        global_parameters['cv']['dimension'] = cv_dimension

    logger.info('Creating datasets from colvars...')

    # Get filter dictionary to select input features for the training
    filter_dict = common.get_filter_dict(features_path, global_parameters['cv']['features_regex'])

    # Build dataset from colvars file with the selected features
    features_dataset, colvars_dataframe = create_dataset_from_files(file_names=[colvars_path], filter_args=filter_dict, verbose = False, return_dataframe=True)         

    # Filter dataframe to keep just the selected features
    features_dataframe = colvars_dataframe.filter(**filter_dict)

    # Log number of features and samples
    logger.info(f' Number of samples: {features_dataframe.shape[0]}')
    logger.info(f' Number of features: {features_dataframe.shape[1]}')

    ###########
    # CV: PCA #
    ###########

    if global_parameters['cv']['type'] in ('PCA', 'ALL'):
        print("Computing PCA")
        compute_pca(features_dataframe = features_dataframe, 
                    cv_dimension = global_parameters['cv']['dimension'], 
                    figures_settings = global_parameters['figures'], 
                    output_folder = output_folder)

    ###################
    # CV: Autoencoder #
    ###################

    if global_parameters['cv']['type'] in ('AE', 'ALL'):
        print("Computing AE")
        compute_ae(features_dataset = features_dataset, 
                   cv_dimension = global_parameters['cv']['dimension'],
                   figures_settings = global_parameters['figures'], 
                   training_settings = global_parameters['cv']['trainings'],
                   output_folder = output_folder)

    ############
    # CV: TICA #
    ############

    if global_parameters['cv']['type'] in ('TICA', 'ALL'):
        print("Computing TICA")
        compute_tica(features_dataframe = features_dataframe,
                     cv_dimension = global_parameters['cv']['dimension'],
                     figures_settings = global_parameters['figures'],
                     output_folder = output_folder)
    
    #################
    # CV: Deep-TICA #
    #################

    if global_parameters['cv']['type'] in ('DTICA', 'ALL'):
        print("Computing DTICA")
        compute_deep_tica(features_dataframe = features_dataframe,
                          cv_dimension = global_parameters['cv']['dimension'],
                          figures_settings = global_parameters['figures'],
                          training_settings = global_parameters['cv']['trainings'],
                          output_folder = output_folder)

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time: %.1f minutes' % (elapsed_time/60))

def set_logger(verbose: bool):
    """
    Function that sets the logging configuration. If verbose is True, it sets the logging level to DEBUG.
    If verbose is False, it sets the logging level to INFO.

    Inputs
    ------

        verbose (bool): If True, sets the logging level to DEBUG. If False, sets the logging level to INFO.
    """

    # Get the path to this file
    file_path = Path(os.path.abspath(__file__))

    # Get the path to the parent directory
    tool_path = file_path.parent

    # Get the path to the parent directory
    all_tools_path = tool_path.parent

    # Get the path to the parent directory
    mlcolvars_utils_path = all_tools_path.parent

    info_config_path = os.path.join(mlcolvars_utils_path, "configurations/log_file/info_configuration.ini")
    debug_config_path = os.path.join(mlcolvars_utils_path, "configurations/log_file/debug_configuration.ini")

    if verbose:
        logging.config.fileConfig(debug_config_path)
    else:
        logging.config.fileConfig(info_config_path)

    logger = logging.getLogger("mlcolvar_utils")

    logger.info("MLCOLVAR Utils: Tool to extract CVs from simulations data")
    logger.info("========================================================= \n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("CV trainer with mlcolvar")

    parser.add_argument('-conf', dest='configuration_path', type=str, help="Path to configuration file (.yml)", required=True)
    parser.add_argument("-colvars", dest='colvars_path', type=str, help="Path to the colvars file", required=True)
    parser.add_argument("-features", dest='features_path', type=str, help="Path to a file containing the features that should be used (these are used if the path is given)", required=False)
    parser.add_argument("-cv_dimension", type=int, help="Dimension of the CVs", required=False)
    parser.add_argument("-cv_type", type=str, help="Type of CV to train (PCA, AE, TICA, DTICA, ALL)", required=False)
    parser.add_argument("-output", dest='output_folder', type=str, default='output', help="Output folder", required=False)
    
    args = parser.parse_args()

    # Set logger
    set_logger(verbose=True)
    logger = logging.getLogger("mlcolvar_utils")

    # Run tool
    train_cvs(args.configuration_path, args.colvars_path, args.features_path, args.cv_dimension, args.cv_type, args.output_folder)