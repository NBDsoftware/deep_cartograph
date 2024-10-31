import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict

########
# TOOL #
########

def filter_features(configuration: Dict, colvars_path: str, csv_summary: bool = False,
                    output_folder: str = 'filter_features'):
    """
    Function that filters the features in the colvars file using different algorithms to select a subset that contains 
    the most information about the system.

    Parameters
    ----------

        configuration:             Configuration dictionary (see default_config.yml for more information)
        colvars_path:              Path to the input colvars file with the time series of features.
        csv_summary:               (Optional) If True, saves a CSV summary with the filter values for each collective variable
        output_folder:             (Optional) Path to the output folder, if not given, a folder named 'filter_features' is created

    Returns
    -------

        filtered_features:   
    """

    from deep_cartograph.modules.amino import amino
    from deep_cartograph.tools.filter_features.filtering import Filter
    from deep_cartograph.modules.common import create_output_folder, validate_configuration, save_list, find_feature_names, files_exist
    from deep_cartograph.yaml_schemas.filter_features import FilterFeatures

    logger = logging.getLogger("deep_cartograph")
    
    logger.info("==================")
    logger.info("Filtering features")
    logger.info("==================")
    logger.info("Finding the features that contains the most information about the transitions or conformational changes.")
    logger.info("The following algorithms are available:")
    logger.info("- Hartigan's dip test filter. Keeps features that are not unimodal.")
    logger.info("- Shannon entropy filter. Keeps features with entropy greater than a threshold.")
    logger.info("- Standard deviation filter. Keeps features with standard deviation greater than a threshold.")
    logger.info("- Final Mutual information clustering (AMINO). Clusters filtered features according to a mutual information based distance and selects one feature per cluster minimizing the distorsion.")
    logger.info("Note that the all features must be in the same units to apply the entropy and standard deviation filters meaningfully.")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    create_output_folder(output_folder)

    # Validate configuration
    configuration = validate_configuration(configuration, FilterFeatures, output_folder)

    # Check the colvars file exists
    if not files_exist(colvars_path):
        logger.error(f"Colvars file {colvars_path} does not exist. Exiting...")
        sys.exit(1)

    # Initialize the list of features
    initial_features = find_feature_names(colvars_path)

    logger.info(f'Initial size of features set: {len(initial_features)}.')
    save_list(initial_features, os.path.join(output_folder, 'all_features.txt'))

    # Create a Filter object
    features_filter = Filter(colvars_path, initial_features, output_folder, configuration['filter_settings'])

    # Filter the features
    filtered_features = features_filter.run(csv_summary)

    # Apply AMINO to the filtered subset of features
    filtered_features = amino(filtered_features, colvars_path, output_folder, configuration['amino_settings'], configuration['sampling_settings'])

    # Save the filtered features
    filtered_features_path = os.path.join(output_folder, 'filtered_features.txt')
    save_list(filtered_features, filtered_features_path)

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Filter features): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))
            
    return filtered_features

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




if __name__ == "__main__":

    from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration

    parser = argparse.ArgumentParser("Deep Cartograph: Filter Features", description="Filter the features in the colvar file using different algorithms to select a subset of features that contains the most information about the system.")
    
    # Inputs
    parser.add_argument("-conf", dest='configuration_path', help="Path to the YAML configuration file with the settings of the filtering task", required=True)
    parser.add_argument("-colvars", dest='colvars_path', type=str, help="Path to the input colvars file", required=True)
    parser.add_argument("-output", dest='output_folder', help="Path to the output folder", required=True)
    parser.add_argument("-csv_summary", action='store_true', help="Save a CSV summary with the values of the different metrics for each feature", required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)
    
    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Create unique output directory
    output_folder = get_unique_path(args.output_folder)
    create_output_folder(output_folder)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Filter colvars file 
    _ = filter_features(
        configuration = configuration,
        colvars_path = args.colvars_path,
        csv_summary = args.csv_summary,
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))

        
    
    