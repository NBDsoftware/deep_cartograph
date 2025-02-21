import os
import io 
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Union

# Set logger
logger = logging.getLogger(__name__)

# I/O CSV Handlers
# ------------
#
# Functions to handle PLUMED CSV files
def read_as_pandas(colvars_path: str) -> pd.DataFrame:
    '''
    Function that reads a COLVARS file and returns a pandas DataFrame with the same column names as in the COLVARS file.    
    The time column in ps will be converted to ns.

    If the logger level is set to DEBUG, information about the column names will be printed.

    Inputs
    ------

        colvars_path    (str):          COLVARS file path

    Outputs
    -------

        colvars_df      (pandas DataFrame):      COLVARS data
    '''

    # Read column names
    column_names = read_column_names(colvars_path)

    # Read COLVARS file
    colvars_df = pd.read_csv(colvars_path, sep='\s+', dtype=np.float32, comment='#', header=None, names=column_names)

    # Convert time from ps to ns - working with integers to avoid rounding errors
    colvars_df["time"] = colvars_df["time"] * 1000 / 1000000

    # Show info of traj_df
    writtable_info = io.StringIO()
    colvars_df.info(buf=writtable_info)
    logger.debug(f"{writtable_info.getvalue()}")

    return colvars_df

def read_column_names(colvars_path: str) -> list:
    '''
    Reads the column names from a COLVARS file. 

    Inputs
    ------

        colvars_path    (str):          COLVARS file path

    Outputs
    -------

        column_names    (list of str):  list with the column names
    '''

    # Read first line of COLVARS file
    with open(colvars_path, 'r') as colvars_file:
        first_line = colvars_file.readline()

    # Separate first line by spaces
    first_line = first_line.split()

    # The first element is "#!" and the second is "FIELDS" - remove them
    column_names = first_line[2:]

    return column_names

def read(colvars_paths: Union[List[str], str], feature_names: List[str], stratified_samples: Union[List[int], None] = None ) -> pd.DataFrame:
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

def check(colvars_path: str):
    ''' 
    Check colvars file content.

        - Check the file is not empty
        - Check the file doesn't contain NaN values
    
    Inputs
    ------

        colvars_path    (str):          COLVARS file path
    '''
    # Check that the file exists
    if not os.path.exists(colvars_path):
        logger.error(f"COLVARS file not found: {colvars_path}")
        sys.exit(1)
    
    # Read file
    colvars_df = pd.read_csv(colvars_path, sep='\s+', dtype=np.float32, comment='#', header=None)
    
    # Check if the file is empty
    if colvars_df.empty:
        logger.error(f"COLVARS file is empty: {colvars_path}")
        sys.exit(1)

    # Check if the file contains NaN values
    if colvars_df.isnull().values.any():
        logger.error(f"COLVARS file contains NaN values: {colvars_path}")
        sys.exit(1)