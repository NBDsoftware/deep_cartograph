import os
import re
import io 
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional

# Set logger
logger = logging.getLogger(__name__)

# I/O CSV Handlers
# ------------
#
# Functions to handle PLUMED CSV files
def read_colvars(colvars_path: str,
                 **kwargs) -> pd.DataFrame:
    '''
    Function that reads a COLVARS file and returns a pandas DataFrame with the same column 
    names as in the COLVARS file. The time column in ps will be converted to ns.

    If the logger level is set to DEBUG, information about the column names will be printed.

    Inputs
    ------

        colvars_path    (str):          COLVARS file path
        
        kwargs          (dict):         keyword arguments passed to pd.read_csv function

    Outputs
    -------

        colvars_df      (pandas DataFrame):      COLVARS data
    '''

    # Read column names
    column_names = read_column_names(colvars_path)

    # Read COLVARS file
    colvars_df = pd.read_csv(
        colvars_path, 
        sep='\s+', 
        dtype=np.float32, 
        comment='#', 
        header=None, 
        names=column_names, 
        **kwargs
    )

    # Convert time from ps to ns - working with integers to avoid rounding errors
    colvars_df["time"] = colvars_df["time"] * 1000 / 1000000

    # Show info of traj_df
    writtable_info = io.StringIO()
    colvars_df.info(buf=writtable_info)
    logger.debug(f"{writtable_info.getvalue()}")

    return colvars_df

def read_column_names(colvars_path: str, features_only: bool = False) -> List[str]:
    '''
    Reads the column names from a COLVARS file. 

    Inputs
    ------

        colvars_path:          
            COLVARS file path
            
        features_only:
            If True, only the features are read. If False, all the columns are read.

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
    
    if features_only:
        # Additional regex used by create_dataset_from_files()
        default_regex = "^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)"
        
        # Filter the features based on the default regex
        column_names = [name for name in column_names if re.search(default_regex, name)]

    return column_names

def read_features(colvars_paths: Union[List[str], str], ref_feature_names: List[str], topology_paths: Union[List[str], None] = None,
         reference_topology: Union[str, None] = None, stratified_samples: Union[List[int], None] = None ) -> pd.DataFrame:
    """ 
    Read the time series data of the features from the colvars files.
    If topologies and reference_topology are given, translate the feature names of each colvars file to the reference topology.
    If not given, assume the feature names are the same in all colvars files.
    If stratified_samples is given, only read the samples corresponding to the indices in this list.
    
    Inputs
    ------

        colvars_paths:          
            List of paths to the colvars files with the time series data of the features
        
        ref_feature_names:     
            List of names feature names to read, should be present in all the colvars files
        
        topology_paths:        
            List of paths to the topology files corresponding to the different colvars files 
            (to translate the feature names if needed)
                                
        reference_topology:     
            Path to reference topology corresponding to the ref_feature_names. If no reference 
            topology is given, the first file in topology_paths is used.
                                
        stratified_samples:     
            List of indices of the samples to use starting at 1
    
    Outputs
    -------

        features_df:            Dataframe with the time series data of the features
    """
    from deep_cartograph.modules.plumed.features import FeatureTranslator
    
    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]
        
    if topology_paths:
        # Set reference topology
        if not reference_topology:
            reference_topology = topology_paths[0]
        # Check there are as many topology files as colvars files
        if len(colvars_paths) != len(topology_paths):
            logger.error(f"Number of topology files does not match the number of colvars files.")
            sys.exit(1)
    
    merged_df = pd.DataFrame()
    for i in range(len(colvars_paths)):
        
        # Check if the file exists
        if not os.path.exists(colvars_paths[i]):
            logger.error(f"Colvars file not found: {colvars_paths[i]}")
            sys.exit(1)

        # Read feature names from the colvars file
        all_feature_names = read_column_names(colvars_paths[i])
            
        # Check if there are any features
        if len(all_feature_names) == 0:
            logger.error(f'No features found in the colvars file: {colvars_paths[i]}')
            sys.exit(1)
        
        if topology_paths:
            # Translate the reference feature names to this topology
            selected_feature_names = FeatureTranslator(reference_topology, topology_paths[i], ref_feature_names).run()
        else:
            selected_feature_names = ref_feature_names
        
        for i in range(len(selected_feature_names)):
            feature_name = selected_feature_names[i]
            # Check all reference features have a translation for this topology
            if feature_name:
                # Check if the feature is in the colvars file
                if feature_name not in all_feature_names:
                    logger.error(f'Feature {feature_name} not found in the colvars file: {colvars_paths[i]}')
                    sys.exit(1)
            else:
                logger.error(f'Feature {ref_feature_names[i]} not found in the reference topology.')
                sys.exit(1)

        if stratified_samples is None:
            # Read the requested features from the colvar file using pandas
            colvars_df = pd.read_csv(colvars_paths[i], sep='\s+', dtype=np.float32, comment='#', header=0, usecols=selected_feature_names, names=all_feature_names)
        else:
            # Read the requested features and samples from the colvar file using pandas
            colvars_df = pd.read_csv(colvars_paths[i], sep='\s+', dtype=np.float32, comment='#', header=0, usecols=selected_feature_names, skiprows= lambda x: x not in stratified_samples, names=all_feature_names)

        # Change the column names to the reference names before concatenating
        colvars_df.columns = ref_feature_names
        
        # Concatenate the dataframes
        merged_df = pd.concat([merged_df, colvars_df], ignore_index=True)

    return merged_df

def check(colvars_path: str):
    ''' 
    Check colvars file content.

    - Check the file is not empty
    - Check the file doesn't contain NaN values
    
    Inputs
    ------

        colvars_path   
            COLVARS file path
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
        
def is_plumed_file(file_path: str) -> bool:
    """
    Check if given file is in PLUMED format.

    Inputs
    ------
    
    file_path
        PLUMED output file

    Returns
    -------
    
    is_plumed
        Bool indicating whether file_path is a plumed output file
    """
    headers = pd.read_csv(file_path, sep=" ", skipinitialspace=True, nrows=0)
    is_plumed = True if " ".join(headers.columns[:2]) == "#! FIELDS" else False
    return is_plumed

def load_dataframe(
    file_paths: Union[List[str], str],
    start: int = 0,
    stop: Union[int, None] = None, 
    stride: int = 1,
    **kwargs
):
    """Load dataframe(s) from file(s).
    In case of PLUMED colvar files automatically handles the column names, 
    otherwise it is just a wrapper for pd.load_csv function.

    Inputs
    ------
    
    filenames
        filenames to be loaded
        
    start: int, optional
        read from this row, default 0
        
    stop: int, optional
        read until this row, default None
        
    stride: int, optional
        read every this number, default 1
        
    kwargs:
        keyword arguments passed to pd.load_csv function

    Outputs
    -------
    
    pandas.DataFrame
        Dataframe
    """

    # if it is a single string
    if type(file_paths) == str:
        file_paths = [file_paths]
    elif type(file_paths) != list:
        raise TypeError(
            f"only strings or list of strings are supported, not {type(file_paths)}."
        )

    # list of file_paths
    df_list = []
    for i, filename in enumerate(file_paths):

        if is_plumed_file(filename):
            df_tmp = read_colvars(filename, **kwargs)
        else:
            df_tmp = pd.read_csv(filename, **kwargs)
            
        # df_tmp["walker"] = [i for _ in range(len(df_tmp))] - NOTE: is this needed?
        df_tmp = df_tmp.iloc[start:stop:stride, :]
        df_list.append(df_tmp)
    
    # Check if df_list is empty
    if len(df_list) == 0:
        logger.error("No dataframes to concatenate.")
        sys.exit(1)
    
    # concatenate dataframes
    df = pd.concat(df_list)
    df.reset_index(drop=True, inplace=True)

    return df

def create_dataframe_from_files(
    colvars_paths: Union[List[str], str],
    topology_paths: Optional[Union[List[str], str]] = None,
    reference_topology: Optional[str] = None,
    features_list: Optional[List[str]] = None,
    file_label: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Create a dataframes from a list of colvars files. 
    
    If feature list is given, filter the features based on this list
    and keep them in the order of the list. If not given, keep all features
    in the order they appear in the colvars files. In this case, it is assumed
    that all colvars files have the same features in the same order. If not, 
    an error is raised. 

    If topology_paths and reference_topology are given, translate the
    feature names of each colvars file to the reference topology before filtering
    and concatenating the dataframes. Otherwise, assume the feature names are 
    the same in all colvars files.
    
    
    Inputs
    ------
    
    colvars_paths
        Path to colvars files or list of paths to colvars files
        
    topology_paths (Optional)
        List of paths to the topology files corresponding to the different colvars files (to translate the feature names if needed).
        If not given, assume the feature names are the same in all colvars files.
    
    reference_topology (Optional)
        Path to reference topology to which the filer_args refer. If no reference topology is given, the first file in topology_paths is used.
    
    features_list (Optional)
        List of feature names to filter and keep in this order. If not given, all features are kept
        and it is assumed that all colvars files have the same feature names in the same order.
        
    file_label (Optional)
        Name of the column to be added with the file label, by default 'traj_label'
        
    kwargs (Optional)
        args passed to pd.load_csv function

    Outputs
    -------
    
    pd.DataFrame
        Pandas dataframe of all the given data
    """
    
    from deep_cartograph.modules.plumed.features import FeatureTranslator
    
    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]

    if isinstance(topology_paths, str):
        topology_paths = [topology_paths]
            
    if topology_paths:
        if (len(colvars_paths) != len(topology_paths)):
            raise TypeError(
                """topology_paths should be a list of paths of same length as colvars_paths."""
            )
        if not reference_topology:
            reference_topology = topology_paths[0]
    
    # Collect dataframes in a list
    all_dfs = []

    # load data, one colvars file at a time
    for file_index in range(len(colvars_paths)):
        
        logger.debug(f"Reading colvars file: {colvars_paths[file_index]}")
        
        tmp_df = load_dataframe(colvars_paths[file_index], **kwargs)
        
        # Remove unwanted columns by default
        tmp_df = tmp_df.filter(regex="^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)")
        
        # Translate feature names if topologies are given
        if topology_paths:
            
            logger.debug(f"Translating feature names from topology {topology_paths[file_index]} to reference topology {reference_topology}")
            
            # Original feature names
            feature_names = list(tmp_df.columns)
            
            # Translate feature names to the reference topology 
            translated_feature_names = FeatureTranslator(topology_paths[file_index], reference_topology, feature_names).run()
            
            # Create a mask for successfully translated features
            translated_features_mask = [name is not None for name in translated_feature_names]
            
            # New feature names
            new_feature_names = [name for name in translated_feature_names if name is not None]
            
            # Warn the user if some features could not be translated
            num_dropped = len(translated_feature_names) - sum(translated_features_mask)
            if num_dropped > 0:
                logger.warning(f"{num_dropped} features could not be translated from topology {topology_paths[file_index]} to reference topology {reference_topology} and will be dropped.")
            
            # Filter the dataframe to keep only the successfully translated features
            tmp_df = tmp_df.loc[:, translated_features_mask]
            
            # Change names
            tmp_df.columns = new_feature_names
        
        # Filter the dataframe
        if features_list:
            # Check for missing features and raise an error
            missing_features = set(features_list) - set(tmp_df.columns)
            if missing_features:
                raise ValueError(
                    f"Features {missing_features} not found in {colvars_paths[file_index]}."
                )
            # Select and reorder columns
            tmp_df = tmp_df[features_list]
        
        # Add file label if given
        if file_label:
            tmp_df[file_label] = file_index
        all_dfs.append(tmp_df)
        
    if not all_dfs:
        logger.error("No dataframes to concatenate.")
        return pd.DataFrame()
            
    # FIX: After the loop, validate columns if no features_list was given
    if not features_list:
        first_cols = all_dfs[0].columns
        for i, df_i in enumerate(all_dfs[1:], 1):
            if not df_i.columns.equals(first_cols):
                logger.error(f"Column names in {colvars_paths[i]} do not match those in {colvars_paths[0]}. Please provide a features_list to filter and reorder the columns.")
                sys.exit(1)
            
    # Concatenate dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Check if the dataframe is empty
    if df.empty:
        logger.error("The resulting dataframe is empty.")
        sys.exit(1)
    
    return df