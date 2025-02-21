"""
Related to Hartigan's dip test: testing the uni-modality of the distribution of the features

In the Hartigan's dip test, the null hypothesis is that the distribution is uni-modal. The alternative hypothesis is that the distribution is multi-modal.

The test statistic is the maximum difference over all sample points, between the empirical distribution function, and the unimodal distribution function that minimizes that maximum difference.

The p-value indicates the probability of rejecting the null hypothesis when it is true. The smaller the p-value, the stronger the evidence against the null hypothesis.
"""

from typing import List
import pandas as pd 

def diptest_calculator(colvars_paths: List[str], feature_names: List[str]) -> pd.DataFrame:
    """
    Function that computes the p-value of the Hartigan Dip test for each feature.

    Inputs
    ------

        colvars_paths:     List of paths to the colvars files with the time series data of the features
        feature_names:     List of names of the features to analyze
    
    Outputs
    -------

        hdt_pvalue_df: Dataframe with the feature names and the p-values of the Hartigan Dip test
    """
    
    def compute_pvalues() -> List[float]:
        """
        Function that computes the p-value of the Hartigan Dip test for each feature.
        
        Outputs
        -------

            hdt_pvalues: List of p-values of the Hartigan Dip test for each feature
        """
        
        from diptest import diptest
        import numpy as np
        
        import deep_cartograph.modules.plumed as plumed
        
        # Iterate over the features
        hdt_pvalues = []

        for name in feature_names:

           # Read the feature time series
            feature_df = plumed.colvars.read(colvars_paths, [name])
            
            # Compute the p-value of the Hartigan Dip test
            hdt_pvalue = diptest(np.array(feature_df[name].to_numpy()))[1]

            # Append the p-value to the list
            hdt_pvalues.append(hdt_pvalue)

        # Return the list
        return hdt_pvalues
    
    # Compute the p-value of the Hartigan Dip test for each feature
    hdt_pvalues = compute_pvalues()  

    # Return a dataframe with the feature names and their p-values
    return pd.DataFrame({'name': feature_names, 'hdtp': hdt_pvalues})