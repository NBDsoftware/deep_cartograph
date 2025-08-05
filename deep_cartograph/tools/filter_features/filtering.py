"""
Class that computes metrics of the features to filter them.
"""

import os
import sys
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Union, Dict

# Local imports
from deep_cartograph.modules.plumed.colvars import read_features

# Set logger
logger = logging.getLogger(__name__)

class Filter:

    def __init__(self, colvars_paths: List[str], settings: Dict, topologies: Union[List[str], None] = None, 
                 reference_topology: Union[str, None] = None, output_dir: str = 'filter_features') -> None:
        """ 
        Class that reads a series of colvars files with feature time series and filters the 
        features based on the entropy, standard deviation, and Hartigan's Dip Test. 
        
        * NOTE: Here we could add autocorrelation or kinetic variance filters
        
        Input
        -----
        
            colvars_paths
                Path to colvars files with the time series of the features
                
            settings
                Filtering options
                
            topologies (Optional)
                Paths to topology files corresponding to the different colvars files
                
            reference_topology (Optional)
                Path to reference topology that will be used to define the filtered list of features. If no reference topology 
                is given, the first file in topologies is used.
                
            output_dir (Optional)
                Path to output folder
        """
        from deep_cartograph.modules.common import save_list
        
        logger.info("Initializing Filter")
        
        # Paths
        self.colvars_paths = colvars_paths
        self.output_dir = output_dir
        
        if topologies:
            if reference_topology is None:
                reference_topology = topologies[0]
        
        self.topology_paths = topologies
        self.ref_topology_path = reference_topology
        
        if self.topology_paths:
            if len(self.colvars_paths) != len(self.topology_paths):
                logger.error('The number of colvars files must be equal to the number of topology files.')
                sys.exit(1)
        
        # Find the common set of features (with reference topology names)
        self.common_ref_features = self.find_common_features()
        
        logger.info(f'Initial size of features set (only common features): {len(self.common_ref_features)}.')
        save_list(self.common_ref_features, os.path.join(self.output_dir, 'all_features.txt'))

        # Configuration 
        self.compute_diptest = settings['compute_diptest']
        self.compute_entropy = settings['compute_entropy']
        self.compute_std = settings['compute_std']
        self.filter_features = self.compute_diptest or self.compute_entropy or self.compute_std

        # Thresholds
        self.diptest_significance_level = settings['diptest_significance_level']
        self.entropy_quantile = settings['entropy_quantile']
        self.std_quantile = settings['std_quantile']
        
        # features to analyze
        self.features_data = pd.DataFrame({'name': self.common_ref_features, 'pass': True})
        
        # Add entropy column if needed
        if self.compute_entropy:
            self.features_data['entropy'] = 0.0
        
        # Add std column if needed
        if self.compute_std:
            self.features_data['std'] = 0.0
        
        # Add dip test column if needed
        if self.compute_diptest:
            self.features_data['hdtp'] = 1.0

    def find_common_features(self) -> List[str]:
        """
        Find the common features that are present in all colvars files. If topologies are given, 
        translate the feature names of each colvars file to the reference topology before comparing them.
        """
        from deep_cartograph.modules.plumed.colvars import read_column_names
        from deep_cartograph.modules.plumed.features import FeatureTranslator
        
        # For each colvars and topology file
        common_features = None
        for i in range(len(self.colvars_paths)):
            
            colvars_path = self.colvars_paths[i]
            
            # Find the feature names in this colvars file
            feature_names = read_column_names(colvars_path, features_only=True)
            feature_names.sort() # NOTE: This sort here is just to maintain the old behavior, remove once the changes are tested, we want to keep same order of weights as in the colvars files - will make life easier for the user
            
            logger.debug(f'There are {len(feature_names)} features in {Path(colvars_path).name}: {feature_names}')
            
            if self.topology_paths:
                # Translate the feature names to the reference topology # NOTE: Here we are assuming that the mda topology is the same as the original one to translate features - won't be true when we start looking at individual atoms
                ref_feature_names = FeatureTranslator(self.topology_paths[i], self.ref_topology_path, feature_names).run()
                # Check if any feature didn't have a translation
                for i in range(len(feature_names)):
                    if ref_feature_names[i] is None:
                        logger.warning(f'Feature {feature_names[i]} from {Path(colvars_path).name} not found in the reference topology.')
                ref_feature_names = [x for x in ref_feature_names if x is not None]
            else:
                ref_feature_names = feature_names
            
            # Accumulate the common features
            if common_features:
                common_features = [feature for feature in common_features if feature in ref_feature_names]
            else:
                common_features = ref_feature_names
        
        if len(common_features) == 0:
            logger.error('No common features found in the colvars files.')
            sys.exit(1)
            
        return list(common_features)

    def run(self, csv_summary: bool = False) -> list:
        """
        Filter the features based on the selected metrics.

        Inputs
        ------

            csv_summary (bool): If True, saves the summary of the filtering to a csv file.
        """
        
        total_num_features = len(self.common_ref_features)
        log_interval = max(1, total_num_features // 10)  # Avoid division by zero
        start_time = time.time()  # Start timer
        
        # Iterate over features
        if self.filter_features:
            for i, reference_feature in enumerate(self.common_ref_features, start=1):
                
                logger.debug(f"Analyzing feature: {reference_feature}")
                
                # Read all the data for this feature
                args = {
                    'colvars_paths': self.colvars_paths,
                    'ref_feature_names': [reference_feature],
                    'topology_paths': self.topology_paths,
                    'reference_topology': self.ref_topology_path
                }
                feature_df = read_features(**args)
                    
                # Entropy
                if self.compute_entropy:
                    
                    from deep_cartograph.modules.statistics import shannon_entropy

                    # Compute and update the entropy of the feature
                    feature_entropy = shannon_entropy(feature_df)
                    self.features_data.loc[(self.features_data['name'] == reference_feature), 'entropy'] = feature_entropy

                # Standard deviation
                if self.compute_std:
                    
                    from deep_cartograph.modules.statistics import standard_deviation

                    # Compute and update the standard deviation of the feature
                    feature_std = standard_deviation(feature_df)
                    self.features_data.loc[(self.features_data['name'] == reference_feature), 'std'] = feature_std
                
                # Dip test
                if self.compute_diptest:
                    
                    from deep_cartograph.modules.statistics import dip_test

                    # Compute and update the p-value of the Hartigan's Dip Test of the feature
                    feature_hdt_pvalues = dip_test(feature_df)
                    self.features_data.loc[(self.features_data['name'] == reference_feature), 'hdtp'] = feature_hdt_pvalues[0]
            
                # Estimate remaining time every log_interval iterations
                if i % log_interval == 0 or i == total_num_features:
                    elapsed_time = time.time() - start_time
                    avg_time_per_feature = elapsed_time / i
                    estimated_remaining = avg_time_per_feature * (total_num_features - i)
                    logger.info(f'Processed {i}/{total_num_features} features. Estimated time left: {estimated_remaining:.2f} seconds.')
        
        if self.compute_entropy and self.entropy_quantile > 0:
            # Filter according to the entropy, those features with entropy below the threshold don't pass the filter
            entropy_threshold = self.features_data['entropy'].quantile(q = self.entropy_quantile)
            logger.info(f'    Entropy threshold: {entropy_threshold:.2f} bits (quantile: {self.entropy_quantile:.2f})')
            self.features_data.loc[(self.features_data['entropy'] < entropy_threshold), 'pass'] = False

        if self.compute_std and self.std_quantile > 0:
            # Filter according to the standard deviation, those features with std below the threshold don't pass the filter
            std_threshold = self.features_data['std'].quantile(q = self.std_quantile)
            logger.info(f'    Standard deviation threshold: {std_threshold:.2f} a.u. (quantile: {self.std_quantile:.2f})')
            self.features_data.loc[(self.features_data['std'] < std_threshold), 'pass'] = False
            
        # Filter according to the p-value of the Hartigan's Dip Test, those with the p-value above the significance level don't pass the filter
        if self.compute_diptest and self.diptest_significance_level > 0:
            self.features_data.loc[(self.features_data['hdtp'] > self.diptest_significance_level), 'pass'] = False
                    
        if csv_summary:
            # Save the dataframe to a csv file
            self.features_data.to_csv(os.path.join(self.output_dir, "filter_summary.csv"), index=False)

        # Discard the features that do not pass the filter
        self.features_data = self.features_data[self.features_data['pass'] == 1]

        # Find the final number of features
        final_num_features = len(self.features_data)

        # Log the number of features filtered
        logger.info(f'Filtered {total_num_features - final_num_features} features.')

        # Return a list with the features to analyze
        return self.features_data['name'].tolist()