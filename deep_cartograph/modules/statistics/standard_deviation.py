"""
Standard deviation calculator
"""

from typing import List
import pandas as pd 

def std_calculator(colvars_paths: List[str], feature_names: List[str]) -> pd.DataFrame:
    """
    Function that filters the features in the colvars file based on the standard deviation 
    of the distribution of each feature. To remove features that 
    do not contain any information about the state of the system.

    This filter should only be used when all the features have the same units.

    Inputs
    ------

        colvars_paths:     List of paths to the colvars files with the time series data of the features
        feature_names:     List of names of the features to analyze
    
    Outputs
    -------

        std_df: Dataframe with the feature names and their standard deviations
    """
    
    def standard_deviation() -> List[float]:
        """
        Function that computes the std of the distribution of each feature.
        
        Outputs
        -------

            feature_stds: List of stds of the features
        """

        import numpy as np
        
        import deep_cartograph.modules.plumed as plumed

        # Iterate over the features
        feature_stds = []

        for name in feature_names:

           # Read the feature time series
            feature_df = plumed.colvars.read(colvars_paths, [name])

            # Compute and append the std to the list
            feature_stds.append(round(np.std(feature_df[name].to_numpy()), 3))

        return feature_stds

    # Compute the standard deviation of each feature
    feature_stds = standard_deviation()

    # Return a dataframe with the feature names and their standard deviations
    return pd.DataFrame({'name': feature_names, 'std': feature_stds})

