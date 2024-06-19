"""
Related to entropy filtering: filtering the order parameters based on the Shannon entropy of their distribution

Intuitive explanation:

    Surprise of the outcome x of a random variable X: log(1/p(x)) -> the more unlikely the outcome, the more surprised we are
                                                                     and p(x) = 1 yields no surprise

    The average surprise of a random variable X: H(X) = \sum_x p(x) log(1/p(x)) = - \sum_x p(x) log(p(x))

    Which is the entropy of the distribution of X

    The average surprise is maximized when the distribution is uniform (all outcomes have the same probability, variability is maximized)

    The more uniform the distribution (many outcomes with similar probabilities, variability is maximized), the higher the entropy
    The more skewed the distribution (few outcomes with high probabilities, variability is minimized), the lower the entropy

    For continuous variables, the entropy is computed as the integral of the probability density function times the log of the probability density function
    and it measures the variability with respect to the unit uniform distribution.

    If its more spread out, the entropy is higher. If its more concentrated, the entropy is lower.

    If it has several peaks, the entropy is higher while if it has a single peak, the entropy is lower (provided they have the same spread or variance)

    Note that the Shannon entropy of continuous distributions will be sensitive to the units of the variable, thus it cannot be used to compare distributions
    of variables with different units.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

# Local imports
from deep_cartograph.modules.common import common


def entropy_calculator(colvar_path: str, ops_names: list):
    """
    Function that computes the Shannon entropy of the distribution of each order parameter.

    Inputs
    ------

        colvar_path:      Path to the colvar file with the time series data of the order parameters
        ops_names:        List of names of the order parameters to analyze
    
    Outputs
    -------

        ops_entropies_df: Dataframe with the order parameter names and their entropies
    """

    # Compute the entropy of each order parameter
    ops_entropies = shannon_entropy(colvar_path, ops_names)

    # Create a dataframe with the order parameter names and their entropies
    ops_entropies_df = pd.DataFrame({'op_name': ops_names, 'entropy': ops_entropies})

    # Return the dataframe
    return ops_entropies_df

def shannon_entropy(colvar_path: str, ops_names: list) -> list:
    """
    Function that computes the Shannon entropy of the distribution of each order parameter.

    Inputs
    ------

        colvar_path: Path to the colvar file with the time series data of the order parameters
        ops_names:   List of names of the order parameters to analyze
    
    Outputs
    -------

        ops_entropies: List of entropies of the order parameters
    """

    # Iterate over the order parameters
    ops_entropies = []

    for op_name in ops_names:

        # Read the order parameter time series
        op_data = common.read_colvars_pandas(colvar_path, [op_name])
        op_timeseries = op_data[op_name]
        
        # Compute the histogram of the order parameter
        hist, bin_edges = np.histogram(op_timeseries, bins=100, density=True)
        prob_distribution = hist * np.diff(bin_edges)

        # Compute and append the entropy to the list
        ops_entropies.append(round(entropy(prob_distribution, base=2), 3))
    
    return ops_entropies
