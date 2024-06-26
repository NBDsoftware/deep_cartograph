"""
Class that computes metrics of the features to filter them.
"""

import os
import logging
import pandas as pd

# Local imports
from deep_cartograph.modules.statistics.entropy import entropy_calculator
from deep_cartograph.modules.statistics.standard_deviation import std_calculator
from deep_cartograph.modules.statistics.dip_test import diptest_calculator

# Set logger
logger = logging.getLogger(__name__)

class Filter:
    """
    Class that contains the information of the features to analyze and the output directory.

    It also contains the filtering methods
    """

    def __init__(self, colvar_path: str, feature_names: list, output_dir: str, settings: dict) -> None:

        # Paths
        self.colvar_path = colvar_path
        self.output_dir = output_dir

        # Configuration 
        self.compute_diptest = settings['compute_diptest']
        self.compute_entropy = settings['compute_entropy']
        self.compute_std = settings['compute_std']

        # Thresholds
        self.diptest_significance_level = settings['diptest_significance_level']
        self.entropy_quantile = settings['entropy_quantile']
        self.std_quantile = settings['std_quantile']

        # features to analyze
        self.feature_names = feature_names
        self.features_data = pd.DataFrame({'op_name': feature_names, 'pass': True})


    def calculate_entropy(self) -> None:
        """
        Compute the entropy of each order parameter.
        """

        # Compute the entropy of each order parameter
        entropy_df = entropy_calculator(self.colvar_path, self.feature_names)

        # Merge the dataframes
        self.features_data = self.features_data.merge(entropy_df, on='op_name', how='inner')

        return
    
    def calculate_std(self) -> None:
        """ 
        Compute the standard deviation of each order parameter.
        """

        # Compute the standard deviation of each order parameter
        std_df = std_calculator(self.colvar_path, self.feature_names)

        # Merge the dataframes
        self.features_data = self.features_data.merge(std_df, on='op_name', how='inner')

        return
    
    def dip_test(self) -> None:
        """
        Compute the dip test for each order parameter.
        """

        # Compute the dip test p-value for each order parameter
        dip_df = diptest_calculator(self.colvar_path, self.feature_names)

        # Merge the dataframes
        self.features_data = self.features_data.merge(dip_df, on='op_name', how='inner')

        return


    def run(self, csv_summary: bool = False) -> list:

        """
        Filter the features based on the selected metrics.

        Inputs
        ------

            csv_summary (bool): If True, saves the summary of the filtering to a csv file.
        """

        # Entropy filter: those with the entropy below the threshold don't pass the filter
        if self.compute_entropy:

            logger.info('    Computing entropy.')
            self.calculate_entropy()

            if self.entropy_quantile > 0:

                entropy_threshold = self.features_data['entropy'].quantile(q = self.entropy_quantile)
                logger.info(f'    Entropy threshold: {entropy_threshold:.2f} bits (quantile: {self.entropy_quantile:.2f})')
                self.features_data.loc[(self.features_data['entropy'] < entropy_threshold), 'pass'] = False
        

        # Standard deviation filter: those with the standard deviation below the threshold don't pass the filter
        if self.compute_std:

            logger.info('    Computing standard deviation.')
            self.calculate_std()

            if self.std_quantile > 0:

                std_threshold = self.features_data['std'].quantile(q = self.std_quantile)
                logger.info(f'    Standard deviation threshold: {std_threshold:.2f} a.u. (quantile: {self.std_quantile:.2f})')
                self.features_data.loc[(self.features_data['std'] < std_threshold), 'pass'] = False
        
        # Dip test filter: those with the p-value of the Hartigan's Dip Test above the significance level don't pass the filter
        if self.compute_diptest:

            logger.info('Computing dip test.')
            self.dip_test()

            if self.diptest_significance_level > 0:

                self.features_data.loc[(self.features_data['hdtp'] > self.diptest_significance_level), 'pass'] = False
            
        
        if csv_summary:
            # Save the dataframe to a csv file
            self.features_data.to_csv(os.path.join(self.output_dir, "filter_summary.csv"), index=False)

        # Find the initial number of features
        initial_num_features = len(self.features_data)

        # Discard the features that do not pass the filter
        self.features_data = self.features_data[self.features_data['pass'] == 1]

        # Find the final number of features
        final_num_features = len(self.features_data)

        # Log the number of features filtered
        logger.info(f'Filtered {initial_num_features - final_num_features} features.')

        # Return a list with the features to analyze
        return self.features_data['op_name'].tolist()