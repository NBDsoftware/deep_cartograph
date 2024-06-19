"""
Statistics related methods
"""

import numpy as np
import pandas as pd

# Local imports
from deep_cartograph.modules.common import common


# Standard deviation related methods
def std_calculator(colvar_path: str, ops_names: list):
    """
    Function that filters the order parameters in the colvar file based on the standard deviation 
    of the distribution of each order parameter. To remove order parameters that 
    do not contain any information about the state of the system.

    This filter should only be used when all the order parameters have the same units.

    Inputs
    ------

        colvar_path:       Path to the colvar file with the time series data of the order parameters
        ops_names:         List of names of the order parameters to analyze
    
    Outputs
    -------

        ops_std_df: Dataframe with the order parameter names and their standard deviations
    """

    # Compute the standard deviation of each order parameter
    ops_stds = std(colvar_path, ops_names)

    # Create a dataframe with the order parameter names and their entropies
    ops_std_df = pd.DataFrame({'op_name': ops_names, 'std': ops_stds})

    # Return the dataframe
    return ops_std_df

def std(colvar_path: str, ops_names: list) -> list:
    """
    Function that computes the std of the distribution of each order parameter.

    Inputs
    ------

        colvar_path: Path to the colvar file with the time series data of the order parameters
        ops_names:   List of names of the order parameters to analyze
    
    Outputs
    -------

        ops_stds: List of stds of the order parameters
    """

    # Iterate over the order parameters
    ops_stds = []

    for op_name in ops_names:

        # Read the order parameter time series
        op_data = common.read_colvars_pandas(colvar_path, [op_name])
        op_timeseries = op_data[op_name]

        # Compute and append the std to the list
        ops_stds.append(round(np.std(op_timeseries), 3))

    return ops_stds

