import os
import sys
import yaml
import math
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path, PurePath
from typing import Any, Dict, List, Union, Tuple
from mlcolvar.data import DictDataset
from pydantic import ValidationError
from pydantic import BaseModel

# Set logger
logger = logging.getLogger(__name__)

# Constants
default_regex = "^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)"


# General utils
def create_output_folder(output_path: str) -> None:
    """
    Creates the output path if it does not exist.

    Parameters
    ----------

    output_path : str
        Path of the output folder
    """

    # Create parent output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return

def files_exist(*file_path):
    '''
    Returns true if all files exist.
    Inputs
    ------

        file_path  (str): variable number of paths to files including filename
    
    Output
    ------

        exist (bool): True if they all exist. False if any of them doesn't exist
    '''

    all_exist = True

    for path in file_path:
        
        this_file_exist = os.path.isfile(path)
        all_exist = all_exist and this_file_exist
            
    return all_exist


# Related to configuration
def read_configuration(configuration_path: str) -> Dict[str, Any]:
    """
    Function to read the YAML configuration file. Exits if configuration file is not found.

    Inputs
    ------

        configuration_path       (str): Path to YAML configuration file

    Outputs
    -------
        
        configuration (dict): Dictionary with configuration
    """

    # Read configuration file
    if files_exist(configuration_path):
        with open(configuration_path) as config_file:
            configuration = yaml.load(config_file, Loader = yaml.FullLoader)
    else:
        logger.error(f"Configuration file {configuration_path} not found")
        sys.exit(1)
    
    return configuration

def validate_configuration(configuration: Dict[str, Any], schema: BaseModel, output_folder: str) -> Dict[str, Any]:
    """
    Validate the configuration dictionary with the given schema and dump the validated configuration to
    the output folder.

    Parameters
    ----------

    configuration : Dict[str, Any]
        Configuration dictionary
    schema : BaseModel
        Pydantic schema to validate the configuration
    output_folder : str
        Path to the output folder

    Returns
    -------

    validated_configuration : Dict[str, Any]
        Validated configuration dictionary
    """

    try:
        validated_configuration = schema(**configuration).model_dump()
    except ValidationError as e:
        logger.error(f"Configuration file is not valid: {e}")
        sys.exit(1)
    
    # Dump the validated configuration to the output folder
    if output_folder is not None:
        bck_configuration_path = os.path.join(output_folder, "configuration.yml")
        with open(bck_configuration_path, 'w') as config_file:
            yaml.dump(validated_configuration, config_file)

    return validated_configuration

def merge_configurations(common_config: Dict, specific_config: Union[Dict, None]) -> Dict:
        """
        Merge the common configuration with the cv-specific configuration recursively.

        It preserves all key and value pairs in the common configuration that are not in 
        the cv-specific configuration 
        
        Returns
        -------

        merged_config : Dict
            Merged configuration dictionary
        """
        
        merged_config = common_config.copy()

        if specific_config:
            for key, value in specific_config.items():
                if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                    # if both values are dictionaries, merge them recursively
                    merged_config[key] = merge_configurations(merged_config[key], value)
                else:
                    # otherwise, use the value from the specific configuration
                    merged_config[key] = value
        else:
            merged_config = common_config
            
        return merged_config


# Features utils
def find_feature_names(colvars_paths: List[str]) -> List[str]:
    """
    Find feature names present in colvars_paths. Feature names are read from the first line. 
    Only the common features present in all colvars files will be returned as a list.

    Inputs
    ------

        colvars_path: Paths to the colvars files with the time series data of the features
    
    Outputs
    -------

        common_features: List of common feature names present in all colvars files
    """
    
    common_features = set()
    for path in colvars_paths:
        
        # Read the first line of the colvars file
        with open(path, 'r') as colvar_file:
            first_line = colvar_file.readline().split()

        # Read feature names. Skip first 3 elements, they are: #! FIELDS time
        feature_names = set()
        for index in range(3, len(first_line)):
            feature_names.add(first_line[index])
            
        if len(common_features) == 0:
            common_features = feature_names
        else:
            common_features = common_features.intersection(feature_names)

    # Check if there are any features
    if len(common_features) == 0:
        logger.error(f'No common features found in the colvars files: {colvars_paths}')
        sys.exit(1)
    
    # Return sorted list to ensure consistency
    return sorted(common_features)

def read_feature_constraints(features_path: Union[str, None], features_regex: Union[str, None] = None) -> Union[List[str], str]:
    """
    Read the feature constraints from the configuration file. Either a list of features or a regex.
    If both are given, the list of features is used.

    Parameters
    ----------

    features_path : str
        Path to the file with the list of features

    features_regex : str
        Regular expression to select the features
    
    Returns
    -------

    feature_constraints : Union[List[str], str]
        List of features to use or regex to select the features
    """

    if features_path is not None:
        logger.info(f' Using features in {features_path}')
        
        # Features path is given, load the list of features 
        feature_constraints = np.loadtxt(features_path, dtype=str)
        
        # Remove any trailing newline characters
        feature_constraints = [feature.strip() for feature in feature_constraints]
        
        return feature_constraints
    
    if features_regex is not None:
        # Regex is given, use it to select the features
        logger.info(f' Using regex to select features: {features_regex}')
        return features_regex
    
    # No features path or regex is given, use default filter
    logger.info(' Using all features except time, *labels, *walker and *bias columns')
    return default_regex



# Input validation
def check_data(trajectory_data: str, topology_data: str) -> Tuple[List[str], List[str]]: # NOTE merge with check_ref_data and add argument to make existence optional
    """
    Function that checks the existence of the necessary input data files.
    
    Inputs
    ------
    
        trajectory_data    (str): Path to trajectory or folder with trajectories.
        topology_data      (str): Path to topology or folder with topology files for the trajectories. 
                                  If a folder is provided, each topology should have the same name as the corresponding trajectory in trajectory_data.
                                  If a single topology file is provided, it is used for all trajectories.
    """
    
    logger = logging.getLogger("deep_cartograph")
    
    if os.path.isdir(trajectory_data):
        # List the files in the trajectory folder
        traj_file_paths = [os.path.join(trajectory_data, f) for f in os.listdir(trajectory_data) if os.path.isfile(os.path.join(trajectory_data, f))]
    elif os.path.isfile(trajectory_data):
        # Single trajectory file
        traj_file_paths = [trajectory_data]
    elif not os.path.exists(trajectory_data):
        logger.error(f"Trajectory data not found: {trajectory_data}")
        sys.exit(1)
    else:
        logger.error(f"Trajectory data should be a file or a folder: {trajectory_data}")
        sys.exit(1)
    
    # Sort them alphabetically 
    traj_file_paths.sort()
    
    # Check if there are any
    if len(traj_file_paths) == 0:
        logger.error(f"Trajectory data folder is empty: {trajectory_data}")
        sys.exit(1)
    
    if os.path.isdir(topology_data):
        # List the files in the topology folder
        top_file_paths = [os.path.join(topology_data, f) for f in os.listdir(topology_data) if os.path.isfile(os.path.join(topology_data, f))]
    elif os.path.isfile(topology_data):
        # Single topology file
        top_file_paths = [topology_data]
    elif not os.path.exists(topology_data):
        logger.error(f"Topology data not found: {topology_data}")
        sys.exit(1)
    else:
        logger.error(f"Topology data should be a file or a folder: {topology_data}")
        sys.exit(1)
        
    # Sort them alphabetically
    top_file_paths.sort()
    
    # Check if there are any
    if len(top_file_paths) == 0:
        logger.error(f"Topology folder is empty: {topology_data}")
        sys.exit(1)
    
    if len(top_file_paths) > 1:
        
        # Check if each trajectory file has a corresponding topology file with the same name
        for traj_file, topology_file in zip(traj_file_paths, top_file_paths):
            
            # Find name of trajectory file
            traj_name = Path(traj_file).stem
            
            # Find name of topology file
            top_name = Path(topology_file).stem
            
            # Check if they have the same name
            if traj_name != top_name:
                logger.error(f"Trajectory file does not have a corresponding topology file with the same name: {traj_name}")
                sys.exit(1)
                
    # If we have a single topology file, we use it for all trajectories
    if len(top_file_paths) == 1 and len(traj_file_paths) > 1:
        top_file_paths = top_file_paths * len(traj_file_paths)
    
    # Check if we have the same number of topology files as trajectory files
    if len(traj_file_paths) != len(top_file_paths):
        logger.error(f"Number of topology files is different from the number of trajectory files ({len(top_file_paths)} vs {len(traj_file_paths)}).")
        sys.exit(1)
            
    return traj_file_paths, top_file_paths

def check_ref_data(ref_trajectory_data: Union[str, None], ref_topology_data: Union[str, None]) -> Tuple[List[str], List[str]]:
    """
    Function that checks (if given) the existence of the optional reference data files.
    
    Inputs
    ------
    
        ref_trajectory_data (str): Path to the folder with reference data.
        ref_topology_data   (str): Path to the folder with topology files of the reference data. Should have the same name as the corresponding reference file in the reference folder.
    """      
    
    logger = logging.getLogger("deep_cartograph")
    
    if ref_trajectory_data is None and ref_topology_data is not None:
        logger.error("Reference topology data provided without reference trajectory data.")
        sys.exit(1)
    elif ref_trajectory_data is not None and ref_topology_data is None:
        logger.error("Reference trajectory data provided without reference topology data.")
        sys.exit(1)
    elif ref_trajectory_data is None and ref_topology_data is None:
        logger.debug("No reference data provided.")
        return [], []
    
    # Both reference trajectory and topology data are provided
    
    # Check reference data 
    if os.path.isdir(ref_trajectory_data):
        # List the files in the reference trajectory folder
        ref_traj_paths = [os.path.join(ref_trajectory_data, f) for f in os.listdir(ref_trajectory_data) if os.path.isfile(os.path.join(ref_trajectory_data, f))]
    elif os.path.isfile(ref_trajectory_data):
        # Single reference trajectory file
        ref_traj_paths = [ref_trajectory_data]
    elif not os.path.exists(ref_trajectory_data):
        logger.error(f"Reference trajectory data not found: {ref_trajectory_data}")
        sys.exit(1)
    else:
        logger.error(f"Reference trajectory data should be a file or a folder: {ref_trajectory_data}")
        sys.exit(1)
        
    # Sort them alphabetically
    ref_traj_paths.sort()
    
    # Check if there are any
    if len(ref_traj_paths) == 0:
        logger.error("Reference trajectory folder is empty: %s", ref_trajectory_data)
        sys.exit(1)
        
    # Check reference topology data
    if os.path.isdir(ref_topology_data):
        # List the files in the reference topology folder
        ref_top_paths = [os.path.join(ref_topology_data, f) for f in os.listdir(ref_topology_data) if os.path.isfile(os.path.join(ref_topology_data, f))]
    elif os.path.isfile(ref_topology_data):
        # Single reference topology file
        ref_top_paths = [ref_topology_data]
    elif not os.path.exists(ref_topology_data):
        logger.error(f"Reference topology data not found: {ref_topology_data}")
        sys.exit(1)
    else:
        logger.error(f"Reference topology data should be a file or a folder: {ref_topology_data}")
        sys.exit(1)
        
    # Sort them alphabetically
    ref_top_paths.sort()

    # Check if there are any
    if len(ref_top_paths) == 0:
        logger.error("Reference topology folder is empty: %s", ref_topology_data)
        sys.exit(1)
    
    # Check if we have the same number of reference topology files as reference trajectory files
    if len(ref_traj_paths) != len(ref_top_paths):
        logger.error(f"Number of reference topology files is different from the number of reference trajectory files ({len(ref_top_paths)} vs {len(ref_traj_paths)}).")
        sys.exit(1)
        
    # Check if each reference trajectory has a corresponding reference topology file
    for ref_traj_path, ref_top_path in zip(ref_traj_paths, ref_top_paths):
            
        # Find name of reference trajectory file
        ref_traj_name = Path(ref_traj_path).stem
        
        # Find name of reference topology file
        ref_top_name = Path(ref_top_path).stem
        
        # Check if they have the same name
        if ref_traj_name != ref_top_name:
            logger.error(f"Reference trajectory file does not have a corresponding reference topology file with the same name: {ref_traj_name}")
            sys.exit(1)
            
    return ref_traj_paths, ref_top_paths

# Related to i/o
def create_dataset_from_dataframe(df: pd.DataFrame, filter_args: dict = None, verbose: bool = True):
    """
    Initialize a dataset from a dataframe. Suitable for supervised/unsupervised tasks.

    Parameters
    ----------
    filter_args: dict, optional
        Dictionary of arguments which are passed to df.filter() to select descriptors (keys: items, like, regex), by default None
        Note that 'time' and '*.bias' columns are always discarded.
    verbose : bool, optional
        Print info on the datasets, by default True
    create_labels: bool, optional
        Assign a label to each file, default True if more than a file is given, otherwise False
    kwargs : optional
        args passed to mlcolvar.utils.io.load_dataframe

    Returns
    -------
    torch.Dataset
        Torch labeled dataset of the given data
    """

    # filter inputs
    df_data = df.filter(**filter_args) if filter_args is not None else df.copy()
    df_data = df_data.filter(regex="^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)")

    if verbose:
        logger.debug(f"\n - Loaded dataframe {df.shape}:", list(df.columns))
        logger.debug(f" - Descriptors {df_data.shape}:", list(df_data.columns))

    # create DictDataset
    dictionary = {"data": torch.Tensor(df_data.values)}

    dataset = DictDataset(dictionary)

    return dataset

def save_list(list_to_save: list, path_to_save: str) -> None:
    """
    Function that saves a list to a file.

    Inputs
    ------

        list_to_save: List to save
        path_to_save: Path to the file where the list will be saved
    """

    # Open the file
    with open(path_to_save, 'w') as file_to_save:

        # Save the list
        for element in list_to_save:
            file_to_save.write(f'{element}\n')

    return

def read_list(path_to_read: str) -> list:
    """
    Function that reads a list from a file.

    Inputs
    ------

        path_to_read: Path to the file where the list is saved
    
    Returns
    -------

        list_read: List read from the file
    """

    # Open the file
    with open(path_to_read, 'r') as file_to_read:

        # Read the lines
        list_read = file_to_read.readlines()

    return list_read

def get_unique_path(path: str):
    """
    Returns a unique path. 
    If it exists, it creates a new path with a suffix number.
    If it does not exist, it returns the original path.
    If the path is a file, it returns the path to the file with a suffix number, respecting the extension.

    Inputs
    ----------

        path           (str): original path
    
    Returns
    -------
        
        unique_path    (str): unique path
    """

    # Convert to PurePath object
    pure_path = PurePath(path)

    # If path exists
    if os.path.exists(path):

        # Get parent path
        parent_path = pure_path.parent

        # If path is a file
        if os.path.isfile(path):

            # Find the suffix and the stem
            stem = pure_path.stem
            suffix = pure_path.suffix
            
            # While the path exists
            i = 1
            while os.path.exists(path):

                # Create new path
                path = os.path.join(parent_path, stem + "_" + str(i) + suffix)
                i += 1
            
            # Return unique path
            return path
        
        # If path is a folder
        elif os.path.isdir(path):

            # Find the name
            name = pure_path.name

            # While the path exists
            i = 1
            while os.path.exists(path):

                # Create new path
                path = os.path.join(parent_path, name + "_" + str(i))
                i += 1
            
            # Return unique path
            return path
    
    # If path does not exist
    else:

        # Return original path
        return path
    
def read_colvars(colvars_paths: Union[List[str], str], feature_names: List[str], stratified_samples: Union[List[int], None] = None ) -> pd.DataFrame:
    """ 
    Read the data of the features in the feature_names list from the colvars file.
    If stratified_samples is not None, only read the samples corresponding to the indices in stratified_samples list.
    
    Inputs
    ------

        colvars_paths:           List of paths to the colvar files with the time series data of the features
        feature_names:          List of names feature names to read, should be present in all the colvars files
        stratified_samples:     List of indices of the samples to use starting at 1
    
    Outputs
    -------

        features_df:            Dataframe with the time series data of the features
    """
    
    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]
    
    merged_df = pd.DataFrame()
    for path in colvars_paths:
        
        # Check if the file exists
        if not os.path.exists(path):
            logger.error(f"Colvars file not found: {path}")
            sys.exit(1)

        # Read first line of colvars file excluding "#! FIELDS"
        with open(path, 'r') as file:
            column_names = file.readline().split()[2:]
            
        # Check if there are any features
        if len(column_names) == 0:
            logger.error(f'No features found in the colvars file: {path}')
            sys.exit(1)

        if stratified_samples is None:
            # Read colvar file using pandas, read only the columns of the features to analyze
            colvars_df = pd.read_csv(path, sep='\s+', dtype=np.float32, comment='#', header=0, usecols=feature_names, names=column_names)
        else:
            # Read colvar file using pandas, read only the columns of the features to analyze and only the rows in stratified_samples
            colvars_df = pd.read_csv(path, sep='\s+', dtype=np.float32, comment='#', header=0, usecols=feature_names, skiprows= lambda x: x not in stratified_samples, names=column_names)

        # Concatenate this dataframe to the merged dataframe
        merged_df = pd.concat([merged_df, colvars_df], ignore_index=True)

    return merged_df

# Related to training
def closest_power_of_two(n: int) -> int:
    """
    Returns the closest power of two that is less than n.

    Inputs
    ------

        n (int): Number
    
    Returns
    -------

        closest_power (int): Closest power of two that is less than n
    """

    # Get the power of two
    closest_power = 2**math.floor(math.log2(n))

    if closest_power == n:
        closest_power = 2**(math.floor(math.log2(n)) - 1)
        
    return closest_power