import os
import sys
import yaml
import math
import logging
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path, PurePath
from typing import Any, Dict, List, Union, Tuple, Optional
from pydantic import ValidationError
from pydantic import BaseModel

# Set logger
logger = logging.getLogger(__name__)

# General utils
def package_is_installed(*package_name: str) -> bool:
    """
    Check if some packages are installed.
    
    Parameters
    ----------
    
    package_name : str
        Variable number of package names to check
    
    Returns
    -------
    
    bool
        True if all packages are installed, False otherwise
    """
    
    for package in package_name:
        if importlib.util.find_spec(package) is None:
            logger.debug(f"Package {package} is not installed")
            return False
    return True

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
        
        if not this_file_exist:
            logger.error(f"File not found {path}")
            
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

def merge_configurations(common_config: Dict, specific_config: Optional[Dict]) -> Dict:
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
            
        return merged_config


# Features utils
def read_features_list(features_path: Optional[str]) -> Union[List[str], str]:
    """
    Read the feature list to use

    Parameters
    ----------

    features_path : str
        Path to the file with the list of features
    
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
    
    return None


# Input validation
def check_data(trajectory_data: str, topology_data: str) -> Tuple[List[str], List[str]]:
    """
    Function that checks the existence of the necessary input data files.
    
    Inputs
    ------
    
    trajectory_data
        Path to trajectory or folder with trajectories.
        
    topology_data
        Path to topology or folder with topology files for the trajectories. 
        If a single topology file is provided, it is used for all trajectories.
        If a folder is given, each trajectory should have a corresponding topology file with the same name.
        
    Returns
    -------
    
    traj_file_paths
        List of trajectory file paths.
    
    top_file_paths
        List of topology file paths.
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
        logger.error(f"Trajectory data should be a path to a file or a folder: {trajectory_data}")
        sys.exit(1)
        
    # Remove any hidden files
    traj_file_paths = [f for f in traj_file_paths if not Path(f).name.startswith('.')]
    
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
        
    # Remove any hidden files
    top_file_paths = [f for f in top_file_paths if not Path(f).name.startswith('.')]
        
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
            
    # Log the found files
    for traj_file, top_file in zip(traj_file_paths, top_file_paths):
        logger.debug(f"Found trajectory file: {Path(traj_file).name}")
        logger.debug(f"Corresponding topology file: {Path(top_file).name}")
    
    return traj_file_paths, top_file_paths

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
    
    import torch
    from mlcolvar.data import DictDataset

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

def save_data(y_data: Dict[str, np.array], x_data: Dict[str, np.array], y_label: str, x_label: str, folder_path: str):
    """
    Save the data to files.

    Parameters
    ----------

    y_data : Dict[str, np.array]
        Dictionary with the y data to save
    x_data : Dict[str, np.array]
        Dictionary with the corresponding x data
    y_label : str
        Label for the y data
    x_label : str
        Label for the x data
    folder_path : str
        Path to the folder where the data will be saved
    """
    
    # For each key in y_data
    for key in y_data.keys():
        
        # Find the corresponding x data
        x_array = x_data.get(key)
        if x_array is None:
            raise ValueError(f"No x values provided for {key}")
        
        # File path
        file_path = os.path.join(folder_path, f"{key}.csv")
        
        # Save the data
        np.savetxt(file_path, np.column_stack((x_array, y_data[key])), delimiter=",", header=f"{x_label},{y_label}", comments="")
    
def write_as_csv(dataframe, path):
    """
    Writes a pandas DataFrame to a CSV file keeping the same column names but in PLUMED format.
    Note that the time column is assumed to be in ns and will be converted to ps!

    If the file already exists, it will append the new data to the existing file.

    Inputs
    ------

        dataframe   (pandas DataFrame):    DataFrame to be written
        path                     (str):    path to the CSV file including the file name

    """

    # Convert time from ns to ps
    dataframe["time"] = dataframe["time"] * 1000

    # Check if file already exists
    if not os.path.isfile(path):

        # Create csv file with header line
        header_line = "#! FIELDS " + " ".join(dataframe.columns)
        with open(path, 'w') as csv_file:
            csv_file.write(header_line + "\n")
    
    else:
        # Erase first dataframe row and add time offset
        # Find last time in csv file
        with open(path, 'r') as csv_file:
            last_line = csv_file.readlines()[-1]
        last_time = float(last_line.split()[0])

        # Erase first row - repeated sample for same initial conditions 
        dataframe = dataframe.drop(dataframe.index[0])

        # Add time offset to dataframe
        dataframe["time"] = dataframe["time"] + last_time

    # Close csv file
    csv_file.close()

    # Append data to csv file
    dataframe.to_csv(path, mode='a', header=False, index=False, sep=' ', float_format='%.6f')

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