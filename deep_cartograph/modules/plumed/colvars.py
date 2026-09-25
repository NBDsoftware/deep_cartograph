"""
Read and check PLUMED colvars (COLVAR) files.

A colvars file is a text table written by PLUMED: a ``#! FIELDS`` header with the column names,
then one row per saved step.
"""
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
    """
    Read a colvars file into a DataFrame, using the column names from its header.

    The ``time`` column is converted from ps to ns.

    Parameters
    ----------
    colvars_path : str
        Path to the colvars file.
    **kwargs
        Extra keyword arguments passed to ``pd.read_csv``.

    Returns
    -------
    colvars_df : pd.DataFrame
        Contents of the colvars file.
    """

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
    """
    Read the column names from the header of a colvars file.

    Parameters
    ----------
    colvars_path : str
        Path to the colvars file.
    features_only : bool, optional
        If True, drop columns that are not features (names containing ``time``, ``bias``,
        ``labels`` or ``walker``). Default is False.

    Returns
    -------
    column_names : list of str
        Column names.
    """

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

def read_features(
    colvars_paths: Union[List[str], str], 
    ref_feature_names: List[str], 
    topology_paths: Union[List[str], None] = None,
    reference_topology: Union[str, None] = None, 
    stratified_samples: Union[List[int], None] = None
    ) -> pd.DataFrame:
    """
    Read some features from one or more colvars files and join them in a single DataFrame.

    If ``topology_paths`` is given, the feature names are translated from the reference topology
    to each file's topology, so systems with different atom numbering can be combined.
    Otherwise the names are assumed to be the same in all files.
    The program exits if a feature is missing.

    Parameters
    ----------
    colvars_paths : str or list of str
        Path(s) to the colvars files.
    ref_feature_names : list of str
        Names of the features to read, as defined in the reference topology.
        They are also used as the column names of the output.
    topology_paths : list of str, optional
        Topology of each colvars file, in the same order. Default is None (no translation).
    reference_topology : str, optional
        Topology that ``ref_feature_names`` refers to. If None, the first of ``topology_paths`` is used.
    stratified_samples : list of int, optional
        Rows to read from each file, counted from 1 (the first data row). Default is None (read all rows).

    Returns
    -------
    features_df : pd.DataFrame
        Feature values from all files, one after the other.
    """
    from deep_cartograph.modules.features import Translator as FeatureTranslator

    if isinstance(colvars_paths, str):
        colvars_paths = [colvars_paths]

    # Check topology paths and set reference topology
    if topology_paths:
        if not reference_topology:
            reference_topology = topology_paths[0]
        if len(colvars_paths) != len(topology_paths):
            logger.error(f"Number of topology files does not match the number of colvars files.")
            sys.exit(1)

    merged_df = pd.DataFrame()
    for colvars_index in range(len(colvars_paths)):

        # Check if the file exists
        if not os.path.exists(colvars_paths[colvars_index]):
            logger.error(f"Colvars file not found: {colvars_paths[colvars_index]}")
            sys.exit(1)

        # Read feature names from the colvars file
        all_feature_names = read_column_names(colvars_paths[colvars_index])

        # Check if there are any features
        if len(all_feature_names) == 0:
            logger.error(f'No features found in the colvars file: {colvars_paths[colvars_index]}')
            sys.exit(1)

        if topology_paths:
            # Translate the reference feature names to this topology
            selected_feature_names = FeatureTranslator(reference_topology, topology_paths[colvars_index], ref_feature_names).run()
        else:
            selected_feature_names = ref_feature_names

        for feature_index in range(len(selected_feature_names)):
            feature_name = selected_feature_names[feature_index]
            # Check all reference features have a translation for this topology
            if feature_name:
                # Check if the feature is in the colvars file
                if feature_name not in all_feature_names:
                    logger.error(f'Feature {feature_name} not found in the colvars file: {colvars_paths[colvars_index]}')
                    sys.exit(1)
            else:
                logger.error(f'Feature {ref_feature_names[feature_index]} not found in the reference topology.')
                sys.exit(1)

        if stratified_samples is None:
            # Read the requested features from the colvar file using pandas
            colvars_df = pd.read_csv(colvars_paths[colvars_index], sep='\s+', dtype=np.float32, comment='#', usecols=selected_feature_names, names=all_feature_names)
        else:
            # Read the requested features and samples from the colvar file using pandas
            colvars_df = pd.read_csv(colvars_paths[colvars_index], sep='\s+', dtype=np.float32, comment='#', usecols=selected_feature_names, skiprows= lambda x: x not in stratified_samples, names=all_feature_names)

        # Enforce the selected features order
        colvars_df = colvars_df[selected_feature_names]
        
        # Change the column names to the reference names before concatenating
        colvars_df.columns = ref_feature_names
        
        # Concatenate the dataframes
        merged_df = pd.concat([merged_df, colvars_df], ignore_index=True)

    return merged_df

def check(colvars_path: str):
    """
    Check a colvars file exists, is not empty and has no NaN values.

    The program exits if any check fails.

    Parameters
    ----------
    colvars_path : str
        Path to the colvars file.
    """
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
    Check whether a file is a PLUMED output file (its header starts with ``#! FIELDS``).

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    is_plumed : bool
        True if the file is a PLUMED output file.
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
    """
    Load one or more files into a single DataFrame.

    PLUMED colvars files are read with ``read_colvars``; other files with ``pd.read_csv``.
    The ``start``, ``stop`` and ``stride`` rows are applied to each file before joining.

    Parameters
    ----------
    file_paths : str or list of str
        Path(s) to the files.
    start : int, optional
        First row to keep. Default is 0.
    stop : int, optional
        Row to stop at (not included). Default is None (until the end).
    stride : int, optional
        Keep one of every ``stride`` rows. Default is 1.
    **kwargs
        Extra keyword arguments passed to the reading function.

    Returns
    -------
    df : pd.DataFrame
        Data from all files, one after the other.

    Raises
    ------
    TypeError
        If ``file_paths`` is not a string or a list.
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
    Create a single DataFrame of features from one or more colvars files.

    Columns that are not features (time, bias, labels, walker) are dropped.
    If ``topology_paths`` is given, feature names are translated from each file's topology to the
    reference topology; features that cannot be translated are dropped.
    If ``features_list`` is given, only those features are kept, in that order. Otherwise all files
    must have the same columns in the same order.

    Parameters
    ----------
    colvars_paths : str or list of str
        Path(s) to the colvars files.
    topology_paths : str or list of str, optional
        Topology of each colvars file, in the same order. Default is None (no translation).
    reference_topology : str, optional
        Topology that the output feature names refer to. If None, the first of ``topology_paths`` is used.
    features_list : list of str, optional
        Features to keep, in this order. Default is None (keep all).
    file_label : str, optional
        If given, add a column with this name holding the index of the source file. Default is None.
    **kwargs
        Extra keyword arguments passed to ``load_dataframe`` (e.g. ``start``, ``stop``, ``stride``).

    Returns
    -------
    df : pd.DataFrame
        Features from all files, one after the other.

    Raises
    ------
    TypeError
        If ``topology_paths`` and ``colvars_paths`` have different lengths.
    ValueError
        If a file has NaN values or is missing a feature from ``features_list``.
    """
    
    from deep_cartograph.modules.features import Translator as FeatureTranslator
    
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
        
        # Add a quick sanity check for NaNs
        if tmp_df.isna().any().any():
            logger.error(f"NaN values detected in raw data file: {colvars_paths[file_index]}")
            raise ValueError(f"Clean your data! NaNs found in {colvars_paths[file_index]}")
        
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