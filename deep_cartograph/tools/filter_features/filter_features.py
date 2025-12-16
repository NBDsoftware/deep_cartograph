import os
import sys
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Union, Optional

from deep_cartograph.yaml_schemas.filter_features import FilterFeaturesSchema
from deep_cartograph.modules.features import Filter
from deep_cartograph.modules.common import (
    validate_configuration, 
    save_list,
    get_unique_path, 
    read_configuration
)

########
# TOOL #
########

def filter_features(
    configuration: Dict,
    colvars_paths: Union[str, List[str]],
    waypoint_colvars_paths: Optional[List[str]] = None,
    csv_summary: bool = True,
    topologies: Optional[List[str]] = None,
    waypoint_topologies: Optional[List[str]] = None,
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
    
    waypoint_colvars_paths : List[str], optional (default: None)
        List of paths to colvars files containing the value of the features for intermediate 
        conformations that define the transition of interest.  
        If given, features that do not change their value across these structures will be filtered out.

    csv_summary : bool, optional (default: True)
        If `True`, saves a CSV summary with filter values for each collective variable.

    waypoint_topologies : List[str], optional (default: None)
        List of topologies corresponding to the waypoint colvars files.

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
    
    # If the output exists already, skip the step
    if os.path.exists(output_features_path):
        logger.info(f"Filtered features file already exists: {output_features_path}. Skipping filtering.")
        return output_features_path

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Validate configuration
    configuration = validate_configuration(configuration, FilterFeaturesSchema, output_folder)
    
    # Check colvars files
    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]
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
        'settings': configuration['filter_settings'],
        'waypoint_colvars_paths': waypoint_colvars_paths,
        'topologies': topologies,
        'waypoint_topologies': waypoint_topologies,
        'reference_topology': reference_topology,
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
        '-waypoint_colvars', dest='waypoint_colvars', type=List[str], nargs='+', required=False,
        help="""List of paths to colvars files containing the value of the features for intermediate 
        conformations that define the transition of interest. If given, features that do not change their value
        across these structures will be filtered out."""
    )
    parser.add_argument(
        '-topologies', dest='topologies', type=List[str], nargs='+', required=False,
        help="List of topologies corresponding to the colvars files."
    )
    parser.add_argument(
        '-waypoint_topologies', dest='waypoint_topologies', type=List[str], nargs='+', required=False,
        help="List of topologies corresponding to the waypoint colvars files."
    )
    parser.add_argument(
        '-ref_topology', dest='reference_topology', type=str, required=False,
        help="Path to the reference topology for feature name translation."
    )
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

    # Determine output folder, if restart is False, create a unique output folder
    output_folder = args.output_folder if args.output_folder else 'filter_features'
    if not args.restart:
        output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Run Filter Features tool
    _ = filter_features(
        configuration = configuration,
        colvars_paths = args.colvars_paths,
        waypoint_colvars_paths = args.waypoint_colvars,
        topologies = args.topologies,
        waypoint_topologies = args.waypoint_topologies,
        reference_topology = args.reference_topology,
        csv_summary = args.csv_summary,
        output_folder = output_folder)
    
if __name__ == "__main__":

    main()
    