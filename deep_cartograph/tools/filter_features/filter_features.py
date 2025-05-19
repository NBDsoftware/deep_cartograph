import os
import sys
import time
import shutil
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Union, Optional

from deep_cartograph.yaml_schemas.filter_features import FilterFeaturesSchema
from deep_cartograph.tools.filter_features.filtering import Filter
from deep_cartograph.modules.common import (
    create_output_folder, 
    validate_configuration, 
    save_list,
    get_unique_path, 
    create_output_folder, 
    read_configuration
)

########
# TOOL #
########

def filter_features(
    configuration: Dict,
    colvars_paths: Union[str, List[str]],
    csv_summary: bool = True,
    topologies: Optional[List[str]] = None,
    reference_topology: Optional[str] = None,
    output_folder: str = "filter_features"
) -> str:
    """
    Filters features from colvars files using various algorithms to select a subset that retains 
    the most information about the system.

    This function is optimized to handle large colvars files efficiently by performing multiple 
    open/close operations to minimize memory usage.

    **NOTE**:  
    - If `topologies` and `reference_topology` are not provided, it is assumed that all colvars files 
      have the same feature names.  
    - This assumption allows easy CLI usage.

    Parameters
    ----------
    configuration : Dict
        Configuration dictionary (see `default_config.yml` for details).
            
    colvars_paths : str or List[str]
        Path or list of paths to colvars file(s) containing the time series of features to filter.  
        If multiple files are provided, they must have the same feature set.
            
    csv_summary : bool, optional (default: True)
        If `True`, saves a CSV summary with filter values for each collective variable.

    topologies : List[str], optional (default: None)
        Topologies corresponding to the colvars files.  
        If provided, they are used to translate feature names to the reference topology.

    reference_topology : str, optional (default: None)
        Reference topology for feature name translation.  
        If `None`, the first topology in `topologies` is used as a reference.
            
    output_folder : str, optional (default: "filter_features")
        Path to the output folder.  
        If not specified, a folder named `"filter_features"` is created.

    Returns
    -------
    output_features_path : str
        Path to the output file containing the filtered features.
    """

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

    if topologies:
        if reference_topology is None:
            reference_topology = topologies[0]
        elif not os.path.exists(reference_topology):
            logger.error(f"Reference topology file missing: {reference_topology}")
            sys.exit(1)

    # Filter the features
    args = {
        'colvars_paths': colvars_paths,
        'topologies': topologies,
        'reference_topology': reference_topology,
        'settings': configuration['filter_settings'],
        'output_dir': output_folder
    }
    filtered_features = Filter(**args).run(csv_summary)

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

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="Deep Cartograph: Filter Features",
        description=("Filter the features in the colvar file using different" 
                     "algorithms to select a subset of features that contains"
                     "the most information about the system."
        )
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-colvars', dest='colvars_paths', type=str, required=True,
        help="Path to the input colvars file."
    )
    
    # Optional arguments
    parser.add_argument(
        '-output', dest='output_folder', type=str, required=False,
        help="Path to the output folder."
    )
    parser.add_argument(
        '-csv_summary', action='store_true', required=False,
        help="Save a CSV summary with the values of the different metrics for each feature."
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
    