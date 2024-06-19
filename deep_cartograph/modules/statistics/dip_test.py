"""
Related to Hartigan's dip test: testing the uni-modality of the distribution of the order parameters

In the Hartigan's dip test, the null hypothesis is that the distribution is uni-modal. The alternative hypothesis is that the distribution is multi-modal.

The test statistic is the maximum difference over all sample points, between the empirical distribution function, and the unimodal distribution function that minimizes that maximum difference.

The p-value indicates the probability of rejecting the null hypothesis when it is true. The smaller the p-value, the stronger the evidence against the null hypothesis.
"""

import numpy as np
import pandas as pd
from diptest import diptest

# Local imports
from deep_cartograph.modules.common import common

def diptest_calculator(colvar_path: str, ops_names: list):
    """
    Function that computes the p-value of the Hartigan Dip test for each order parameter.

    Inputs
    ------

        colvar_path:      Path to the colvar file with the time series data of the order parameters
        ops_names:        List of names of the order parameters to analyze
    
    Outputs
    -------

        hdt_pvalue_df: Dataframe with the order parameter names and the p-values of the Hartigan Dip test
    """

    # Compute the p-value of the Hartigan Dip test for each order parameter
    ops_hdt_pvalues = compute_pvalues(colvar_path, ops_names)

    # Create a dataframe with the order parameter names and their p-values
    ops_entropies_df = pd.DataFrame({'op_name': ops_names, 'hdtp': ops_hdt_pvalues})

    # Return the dataframe
    return ops_entropies_df

def compute_pvalues(colvar_path: str, ops_names: list) -> list:
    """
    Function that computes the p-value of the Hartigan Dip test for each order parameter.

    Inputs
    ------

        colvar_path: Path to the colvar file with the time series data of the order parameters
        ops_names:   List of names of the order parameters to analyze
    
    Outputs
    -------

        ops_hdt_pvalues: List of p-values of the Hartigan Dip test for each order parameter
    """

    # Iterate over the order parameters
    ops_hdt_pvalues = []

    for op_name in ops_names:

        # Read the order parameter time series
        op_data = common.read_colvars_pandas(colvar_path, [op_name])
        op_timeseries = op_data[op_name]
        
        # Compute the p-value of the Hartigan Dip test
        op_hdt_pvalue = diptest(np.array(op_timeseries))[1]

        # Append the p-value to the list
        ops_hdt_pvalues.append(op_hdt_pvalue)

    # Return the list
    return ops_hdt_pvalues