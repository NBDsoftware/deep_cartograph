import os
import sys
import yaml
import math
import torch
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from pathlib import PurePath
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
        validated_configuration = schema(**configuration).dict()
    except ValidationError as e:
        logger.error(f"Configuration file is not valid: {e}")
        sys.exit(1)
    
    # Dump the validated configuration to the output folder
    if output_folder is not None:
        bck_configuration_path = get_unique_path(os.path.join(output_folder, "configuration.yml"))
        with open(bck_configuration_path, 'w') as config_file:
            yaml.dump(validated_configuration, config_file)

    return validated_configuration

# Features utils
def find_feature_names(colvars_path: str) -> list:
    """
    Find feature names from first colvars line and save them in a list

    Inputs
    ------

        colvars_path: Path to the colvars file with the time series data of the features / collective variables
    
    Outputs
    -------

        feature_names: List of feature names
    """

    # Open the colvar file
    with open(colvars_path, 'r') as colvar_file:

        # Read the first line and split it
        first_line = colvar_file.readline().split()

    feature_names = []

    # Loop over the elements of the first line, starting from the 4th element
    for op_index in range(3, len(first_line)):
        feature_names.append(first_line[op_index])

    # Check if there are any features
    if len(feature_names) == 0:
        logger.error(f'No features found in the colvars file {colvars_path}')
        sys.exit(1)
        
    return feature_names

def read_feature_constraints(features_path: str, features_regex: str) -> Union[List[str], str]:
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
        # Features path is given, load the list of features
        feature_constraints = np.loadtxt(features_path, dtype=str)
        logger.info(f' Using features in {features_path}')
        return feature_constraints
    
    if features_regex is not None:
        # Regex is given, use it to select the features
        logger.info(f' Using regex to select features: {features_regex}')
        return features_regex
    
    # No features path or regex is given, use default filter
    logger.info(' Using all features except time, *labels, *walker and *bias columns')
    return default_regex

def get_filter_dict(feature_constraints: Union[List[str], str]) -> Dict:
    """
    Create the filter dictionary to select the features to use from the feature constraints.

    Parameters
    ----------

    feature_constraints: Union[List[str], str]
        List of features to use or regex to select the features
    
    Returns
    -------
    
    filter_dict : dict
        Dictionary with the filter to select the features
    """

    if isinstance(feature_constraints, list):
        # List of features is given
        filter_dict = dict(items=feature_constraints)

    elif isinstance(feature_constraints, str):
        # Regex is given
        filter_dict = dict(regex=feature_constraints)
        
    else:
        # No constraints are given
        filter_dict = dict(regex=default_regex)
    
    return filter_dict


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
        print(f"\n - Loaded dataframe {df.shape}:", list(df.columns))
        print(f" - Descriptors {df_data.shape}:", list(df_data.columns))

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
    
def read_colvars_pandas(colvars_path: str, feature_names: list, stratified_samples: list = None ) -> Dict:
    """ 
    Read the data of the features in the feature_names list from the colvars file.
    If stratified_samples is not None, only read the samples corresponding to the indices in stratified_samples list.
    
    Inputs
    ------

        colvars_path:           Path to the colvar file with the time series data of the features / collective variables
        feature_names:          List of names feature names to read
        stratified_samples:     List of indices of the samples to use starting at 1
    
    Outputs
    -------

        ops_data:               Dictionary with the time series data of the features in the feature_names list
    """

    # Read first line of COLVARS file
    with open(colvars_path, 'r') as colvars_file:
        first_line = colvars_file.readline()
    
    # Close COLVARS file
    colvars_file.close()

    # Separate first line by spaces
    first_line = first_line.split()

    # The first element is "#!" and the second is "FIELDS", remove them
    column_names = first_line[2:]

    if stratified_samples is None:
        # Read colvar file using pandas, read only the columns of the features to analyze
        colvars_df = pd.read_csv(colvars_path, sep='\s+', dtype=np.float32, comment='#', header=0, usecols=feature_names, names=column_names)
    else:
        # Read colvar file using pandas, read only the columns of the features to analyze and only the rows in stratified_samples
        colvars_df = pd.read_csv(colvars_path, sep='\s+', dtype=np.float32, comment='#', header=0, usecols=feature_names, skiprows= lambda x: x not in stratified_samples, names=column_names)

    # Convert the dataframe to a dictionary
    features_data = colvars_df.to_dict('list')

    return features_data

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