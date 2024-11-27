"""
Related to entropy filtering: filtering the features based on the Shannon entropy of their distribution

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

from typing import List
import pandas as pd 

def entropy_calculator(colvars_paths: List[str], feature_names: List[str]) -> pd.DataFrame:
    """
    Function that computes the Shannon entropy of the distribution of each feature.

    Inputs
    ------

        colvars_paths:     List of paths to the colvar files with the time series data of the features
        feature_names:     List of names of the features to analyze
    
    Outputs
    -------

        entropies_df: Dataframe with the feature names and their entropies
    """

    def shannon_entropy() -> List[float]:
        """
        Function that computes the Shannon entropy of the distribution of each feature.
        
        Outputs
        -------

            feature_entropies: List of entropies of the features
        """

        from scipy.stats import entropy
        import numpy as np

        from deep_cartograph.modules.common import read_colvars

        # Iterate over the features
        feature_entropies = []

        for name in feature_names:

            # Read the feature time series
            feature_df = read_colvars(colvars_paths, [name])
            
            # Compute the histogram of the feature
            hist, bin_edges = np.histogram(feature_df[name].to_numpy(), bins=100, density=True)
            prob_distribution = hist * np.diff(bin_edges)

            # Compute and append the entropy to the list
            feature_entropies.append(round(entropy(prob_distribution, base=2), 3))
        
        return feature_entropies

    # Compute the entropy of each feature
    feature_entropies = shannon_entropy()

    # Return a dataframe with the feature names and their entropies
    return pd.DataFrame({'name': feature_names, 'entropy': feature_entropies})
