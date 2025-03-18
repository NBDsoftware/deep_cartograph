import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Union

########
# TOOL #
########

def filter_features(configuration: Dict, colvars_paths: Union[str, List[str]], csv_summary: bool = True, 
                    output_folder: str = 'filter_features'):
    """
    Function that filters the features in the colvars file/s using different algorithms to select a subset that contains 
    the most information about the system.
    
    The API is prepared to handle multiple large colvars files. It will incur in many opening and closing operations on the files to avoid memory issues.
    
    NOTE: add a quick version that loads all data into memory for small datasets? Depending on the number of samples/files vs number of features ratio.

    Parameters
    ----------

        configuration:             
            Configuration dictionary (see default_config.yml for more information)
            
        colvars_paths:             
            Path or list of paths to the input colvars file/s with the time series of features to filter. If more than one file is given, they should have the same features.
            
        csv_summary:               
            (Optional) If True, saves a CSV summary with the filter values for each collective variable
            
        output_folder:             
            (Optional) Path to the output folder, if not given, a folder named 'filter_features' is created

    Returns
    -------

        output_features_path:      Path to the output file with the filtered features.
    """

    from deep_cartograph.tools.filter_features.filtering import Filter
    from deep_cartograph.modules.common import create_output_folder, validate_configuration, save_list, find_feature_names
    from deep_cartograph.yaml_schemas.filter_features import FilterFeaturesSchema

    logger = logging.getLogger("deep_cartograph")
    
    logger.info("==================")
    logger.info("Filtering features")
    logger.info("==================")
    logger.info("Finding the features that contains the most information about the transitions or conformational changes.")
    logger.info("The following algorithms are available:")
    logger.info("- Hartigan's dip test filter. Keeps features that are not unimodal.")
    logger.info("- Shannon entropy filter. Keeps features with entropy greater than a threshold.")
    logger.info("- Standard deviation filter. Keeps features with standard deviation greater than a threshold.")
    logger.info("Note that the all features must be in the same units to apply the entropy and standard deviation filters meaningfully.")

    # Start timer
    start_time = time.time()
    
    # Set output file path
    output_features_path = os.path.join(output_folder, 'filtered_features.txt')
    
    # If the file exists already, skip the filtering
    if os.path.exists(output_features_path):
        logger.info(f"Filtered features file already exists: {output_features_path}. Skipping filtering.")
        return output_features_path

    # Create output folder if it does not exist
    create_output_folder(output_folder)

    # Validate configuration
    configuration = validate_configuration(configuration, FilterFeaturesSchema, output_folder)
    
    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]

    # Check the colvars file exists
    check_colvars(colvars_paths)

    # Initialize the list of features
    initial_features = find_feature_names(colvars_paths)

    logger.info(f'Initial size of features set: {len(initial_features)}.')
    save_list(initial_features, os.path.join(output_folder, 'all_features.txt'))

    # Create a Filter object
    features_filter = Filter(colvars_paths, initial_features, output_folder, configuration['filter_settings'])

    # Filter the features
    filtered_features = features_filter.run(csv_summary)

    # Save the filtered features
    save_list(filtered_features, output_features_path)

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Filter features): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))
            
    return output_features_path

def check_colvars(colvars_paths: List[str]):
    """
    Function that checks the existence of the colvars files.

    Parameters
    ----------

        colvars_paths: List of paths to the input colvars files with the time series of features to filter.
    """

    for path in colvars_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Colvars file not found: {path}")
    
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

    from deep_cartograph.modules.common import get_unique_path, create_output_folder, read_configuration

    parser = argparse.ArgumentParser("Deep Cartograph: Filter Features", description="Filter the features in the colvar file using different algorithms to select a subset of features that contains the most information about the system.")
    
    # Inputs
    parser.add_argument("-conf", dest='configuration_path', help="Path to the YAML configuration file with the settings of the filtering task", required=True)
    parser.add_argument("-colvars", dest='colvars_paths', type=str, help="Path to the input colvars file", required=True)
    parser.add_argument("-output", dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument("-csv_summary", action='store_true', help="Save a CSV summary with the values of the different metrics for each feature", required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)
    
    args = parser.parse_args()

    # Set logger
    set_logger(verbose=args.verbose)

    # Give value to output_folder
    if args.output_folder is None:
        output_folder = 'filter_features'
    else:
        output_folder = args.output_folder
        
    # Create unique output directory
    output_folder = get_unique_path(output_folder)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Filter colvars file 
    _ = filter_features(
        configuration = configuration,
        colvars_paths = args.colvars_paths,
        csv_summary = args.csv_summary,
        output_folder = output_folder)

    # Move log file to output folder
    shutil.move('deep_cartograph.log', os.path.join(output_folder, 'deep_cartograph.log'))

if __name__ == "__main__":

    main()
    