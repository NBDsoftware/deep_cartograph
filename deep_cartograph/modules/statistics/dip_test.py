"""
Related to Hartigan's dip test: testing the uni-modality of the distribution of the features

In the Hartigan's dip test, the null hypothesis is that the distribution is uni-modal. The alternative hypothesis is that the distribution is multi-modal.

The test statistic is the maximum difference over all sample points, between the empirical distribution function, and the unimodal distribution function that minimizes that maximum difference.

The p-value indicates the probability of rejecting the null hypothesis when it is true. The smaller the p-value, the stronger the evidence against the null hypothesis.
"""

def diptest_calculator(colvar_path: str, feature_names: list):
    """
    Function that computes the p-value of the Hartigan Dip test for each feature.

    Inputs
    ------

        colvar_path:      Path to the colvar file with the time series data of the features
        feature_names:    List of names of names of the features to analyze
    
    Outputs
    -------

        hdt_pvalue_df: Dataframe with the feature names and the p-values of the Hartigan Dip test
    """

    import pandas as pd
    
    # Compute the p-value of the Hartigan Dip test for each feature
    hdt_pvalues = compute_pvalues(colvar_path, feature_names)

    # Create a dataframe with the feature names and their p-values
    results_df = pd.DataFrame({'name': feature_names, 'hdtp': hdt_pvalues})

    # Return the dataframe
    return results_df

def compute_pvalues(colvar_path: str, feature_names: list) -> list:
    """
    Function that computes the p-value of the Hartigan Dip test for each feature.

    Inputs
    ------

        colvar_path:   Path to the colvar file with the time series data of the features
        feature_names:  List of names of the features to analyze
    
    Outputs
    -------

        hdt_pvalues: List of p-values of the Hartigan Dip test for each feature
    """
    
    from diptest import diptest
    import numpy as np

    from deep_cartograph.modules.common import common
    
    # Iterate over the features
    hdt_pvalues = []

    for name in feature_names:

        # Read the feature time series
        feature_data = common.read_colvars_pandas(colvar_path, [name])
        feature_timeseries = feature_data[name]
        
        # Compute the p-value of the Hartigan Dip test
        hdt_pvalue = diptest(np.array(feature_timeseries))[1]

        # Append the p-value to the list
        hdt_pvalues.append(hdt_pvalue)

    # Return the list
    return hdt_pvalues