"""
Standard deviation calculator
"""

def std_calculator(colvar_path: str, feature_names: list):
    """
    Function that filters the features in the colvar file based on the standard deviation 
    of the distribution of each feature. To remove features that 
    do not contain any information about the state of the system.

    This filter should only be used when all the features have the same units.

    Inputs
    ------

        colvar_path:       Path to the colvar file with the time series data of the features
        feature_names:         List of names of the features to analyze
    
    Outputs
    -------

        std_df: Dataframe with the feature names and their standard deviations
    """

    import pandas as pd

    # Compute the standard deviation of each feature
    feature_stds = std(colvar_path, feature_names)

    # Create a dataframe with the feature names and their entropies
    std_df = pd.DataFrame({'name': feature_names, 'std': feature_stds})

    # Return the dataframe
    return std_df

def std(colvar_path: str, feature_names: list) -> list:
    """
    Function that computes the std of the distribution of each feature.

    Inputs
    ------

        colvar_path:  Path to the colvar file with the time series data of the features
        feature_names:   List of names of the features to analyze
    
    Outputs
    -------

        feature_stds: List of stds of the features
    """

    import numpy as np

    from deep_cartograph.modules.common import common

    # Iterate over the features
    feature_stds = []

    for name in feature_names:

        # Read the feature time series
        feature_data = common.read_colvars_pandas(colvar_path, [name])
        feature_timeseries = feature_data[name]

        # Compute and append the std to the list
        feature_stds.append(round(np.std(feature_timeseries), 3))

    return feature_stds

