import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.sparse import block_diag
from typing import Dict, List, Tuple, Union, Literal, Optional
from sklearn.decomposition import PCA       

from deep_cartograph.modules.common import closest_power_of_two, create_output_folder
import deep_cartograph.modules.md as md

# Set logger
logger = logging.getLogger(__name__)

# Base class for collective variables calculators
class CVCalculator:
    """
    Base class for collective variables calculators.
    """
    def __init__(self, 
        configuration: Dict,
        train_colvars_paths: List[str],
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Optional[str] = None
        ):
        """
        Initializes the base CV calculator.
        
        Parameters
        ----------
    
        configuration
            Configuration dictionary with settings for the CV
            
        train_colvars_paths 
            List of paths to colvars files with the main data used for training
                
        train_topology_paths (Optional)
            List of paths to topology files corresponding to the training colvars files (same order)
        
        ref_topology_path (Optional)
            Path to the reference topology file. If None, the first topology file is used as reference topology
            
        feature_constraints (Optional)
            List with the features to use for the training or str with regex to filter feature names.
            
        sup_colvars_paths (Optional)
            List of paths to colvars files with supplementary data to project alongside the FES of the training data 
            (e.g. experimental structures). If None, no supplementary data is used
        
        sup_topology_paths (Optional)
            List of paths to topology files corresponding to the supplementary colvars files (same order)
            
        output_path (Optional)
            Output path where the CV results folder will be created
        """
        
        # Total number of samples
        self.num_samples: int = None
        # Filter dictionary
        self.feature_filter: Union[Dict, None] = self.get_feature_filter(feature_constraints)
        
        # Configuration
        self.configuration: Dict = configuration
        self.architecture_config: Dict = configuration['architecture']
        self.training_reading_settings: Dict = configuration['input_colvars']
        self.feats_norm_mode: Literal['mean_std', 'min_max_range1', 'min_max_range2', 'none'] = configuration['features_normalization']
        self.bias: Dict = configuration['bias']
        
        # Colvars paths
        self.train_colvars_paths: List[str] = train_colvars_paths
        self.sup_colvars_paths: Optional[List[str]] = sup_colvars_paths
        
        # Topologies
        self.topologies: Optional[List[str]] = train_topology_paths     # NOTE: Should these be attributes?
        self.ref_topology_path: Optional[str] = ref_topology_path
        if self.topologies is not None:
            if self.ref_topology_path is None:
                self.ref_topology_path = self.topologies[0]
        
        # Filtered training data used to train / compute the CVs
        logger.info('Reading training data from colvars files...')
        reading_args = {
            'colvars_paths': train_colvars_paths,
            'topology_paths': self.topologies,
            'ref_topology_path': self.ref_topology_path,
            'load_args': self.training_reading_settings}
        self.training_data: pd.DataFrame = self.read_data(**reading_args)
        self.training_data_labels: np.array = self.training_data.pop("label").to_numpy()

        # Projected training data NOTE: make sure they have the labels from the training_data
        self.projected_training_data: Optional[pd.DataFrame] = None
        
        # Number of samples used to train / compute the CV - depends on the specific CV calculator
        self.num_training_samples: Union[int, None] = None
        
        # Read the supplementary data
        if sup_colvars_paths:
            logger.info('Reading supplementary data from supplementary colvars files...')
            reading_args = {
                'colvars_paths': sup_colvars_paths,
                'topology_paths': sup_topology_paths,
                'ref_topology_path': ref_topology_path,
                'load_args': self.training_reading_settings
            }
            self.supplementary_data: pd.DataFrame = self.read_data(**reading_args)
            self.supplementary_data_labels: np.array = self.supplementary_data.pop("label").to_numpy()
    
            # Make sure we use the same features between training and supplementary data
            self.training_data, self.supplementary_data = self.align_dataframes(self.training_data, self.supplementary_data)
            
        else:
            self.supplementary_data: Optional[pd.DataFrame] = None
            
        # List of features used for training (features in the colvars files after filtering) 
        self.feature_labels: List[str] = self.training_data.columns.tolist()
        self.num_features: int = len(self.feature_labels)
        logger.info(f'Number of features: {self.num_features}')
        
        # Compute training data statistics
        stats = ['mean', 'std', 'min', 'max']
        stats_df = self.training_data.agg(stats).T
        self.features_stats: Dict[str, np.array] = {stat: stats_df[stat].to_numpy() for stat in stats}
        self.features_norm_mean, self.features_norm_range = self.prepare_normalization()

        # Projected supplementary data
        self.projected_supplementary_data : Optional[pd.DataFrame] = None
        
        # General CV attributes
        self.cv_dimension: int = configuration['dimension']
        self.cv_labels: List[str] = []
        self.cv_name: str = None
        self.cv_range: List[Tuple[float, float]] = [] 

        self.output_path: str = output_path
    
    def initialize_cv(self):
        """
        Initialize the specific CV. 
        
        These are tasks that are common to all CV calculators
        but have to be done after the specific CV calculator constructor has been called.
        
            - Creates the output folder for the CV using the cv_name
            - Logs the start of the calculation using the cv_name
        """
        # Get the total number of samples
        self.num_samples = len(self.training_data)
        logger.info(f'Number of samples: {self.num_samples}')
        
        # Create output folder for this CV
        self.output_path = os.path.join(self.output_path, self.cv_name)
        create_output_folder(self.output_path)
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...') 
        
    def cv_ready(self) -> bool:
        """
        Checks if the CV is ready to be used.
        """
        
        return self.cv is not None
        
    def prepare_normalization(self) -> Tuple[np.array, np.array]:

        """ 
        Prepare the normalization parameters for the features. Computes the normalization
        means and ranges based on the feature statistics and the chosen normalization mode.
        
        The normalization will be:

                              feature - normalization_mean
        normalized feature = --------------------------------
                                    normalization_range
        
        Returns
        -------
        
        means : np.array
            Means for normalization
        ranges : np.array
            Ranges for normalization
        """
        
        def sanitize_ranges(range_array: np.ndarray):
            """
            Check the ranges are not close to zero and set them to 1 if they are.
            """
            for i in range(len(range_array)):
                if abs(range_array[i]) < 1e-8:
                    range_array[i] = 1.0
                    logger.warning(f'Range for feature {i} is close to zero. Setting it to 1.0.')
            return range_array
            
        # Set the mean and range for the normalization
        # No normalization of input features (not recommended)
        if self.feats_norm_mode == 'none':
            means = np.zeros(len(self.features_stats["mean"]))
            ranges = np.ones(len(self.features_stats["mean"]))
        # Normalized data will have mean 0 and std 1, non-defined range.
        elif self.feats_norm_mode == 'mean_std':
            means = self.features_stats["mean"]
            ranges = self.features_stats["std"]
        # Normalized data will have a range of [0, 1]
        elif self.feats_norm_mode == 'min_max_range1':
            means = self.features_stats["min"]
            ranges = self.features_stats["max"] - self.features_stats["min"]
        # Normalized data will have a range of [-1, 1]
        elif self.feats_norm_mode == 'min_max_range2':
            means = (self.features_stats["min"] + self.features_stats["max"]) / 2
            ranges = (self.features_stats["max"] - self.features_stats["min"]) / 2
        else:
            logger.error(f'Normalization mode {self.feats_norm_mode} not recognized. Exiting...')
            sys.exit(1)
        
        # Check the ranges are not close to zero
        ranges = sanitize_ranges(ranges)
        
        return means, ranges
        
    # Readers
    def read_data(self, 
                  colvars_paths: List[str],
                  topology_paths: Optional[List[str]] = None,   
                  ref_topology_path: Optional[str] = None,
                  load_args: Optional[Dict] = None
        ) -> pd.DataFrame:
        """
        Read the data from the colvars files, translate the feature names using the 
        topology files and filter the features based on the feature filter.
        
        Parameters
        ----------
        
        colvars_paths : str
            List of paths to colvars files with the data
            
        topology_paths : Optional[str]
            List of paths to topology files corresponding to the colvars files (same order)
        
        ref_topology_path : Optional[str]
            Path to the reference topology file. If None, the first topology file is used as reference topology
        
        load_args : Optional[Dict]
            Dictionary with the arguments to load the colvars files. 
            
        Returns
        -------
        
        df : pd.DataFrame
            Dataframe with the data from the colvars files
        """
        from deep_cartograph.modules.plumed.colvars import create_dataframe_from_files
        
        if load_args is None:
            load_args = {}
        df = create_dataframe_from_files(
            colvars_paths = colvars_paths,
            topology_paths = topology_paths,
            reference_topology = ref_topology_path,   
            filter_args = self.feature_filter,  
            create_labels = True,
            **load_args
        )
        
        # If the dataframe is empty, raise an error
        if df.empty:
            logger.error(f'No data found in the colvars files: {colvars_paths}. Exiting...')
            sys.exit(1)
        
        return df
    
    def align_dataframes(self, training_data: pd.DataFrame, supplementary_data: pd.DataFrame):
        """
        Align two pandas DataFrames to have the same columns in the same order, 
        based on the columns of the first DataFrame.

        This function:
        - Identifies the common columns between training_data and supplementary_data.
        - Keeps the column order as in training_data.
        - Drops any columns not shared by both DataFrames.
        - Prints a warning if any columns are dropped from either DataFrame.

        Parameters
        ----------
            training_data : pd.DataFrame
                The reference DataFrame whose column order is preserved.
            supplementary_data : pd.DataFrame
                The DataFrame to be aligned with training_data.

        Returns
        -------
            training_data_aligned : pd.DataFrame
                A version of training_data with only the common columns.
            supplementary_data_aligned : pd.DataFrame
                A version of supplementary_data with columns aligned to training_data's order.
        """
        
        logger.debug(f"Aligning Training and Supplementary data...")
    
        # Get column name sets
        cols1 = set(training_data.columns)
        cols2 = set(supplementary_data.columns)

        # Find common columns, preserving order of training_data
        common_cols = [col for col in training_data.columns if col in cols2]
        
        # If no common columns, raise an error
        if not common_cols:
            logger.error("No common columns found between training_data and supplementary_data. Exiting...")
            sys.exit(1)

        # Warn if any columns are discarded
        dropped_from_training_data = cols1 - cols2
        dropped_from_supplementary_data = cols2 - cols1

        if dropped_from_training_data:
            logger.warning(f"Dropping columns from training_data: {sorted(dropped_from_training_data)}")
        if dropped_from_supplementary_data:
            logger.warning(f"Dropping columns from supplementary_data: {sorted(dropped_from_supplementary_data)}")

        # Align both DataFrames
        training_data_aligned = training_data[common_cols]
        supplementary_data_aligned = supplementary_data[common_cols]

        return training_data_aligned, supplementary_data_aligned

    def get_feature_filter(self, feature_constraints: Union[List[str], str]) -> Dict:
        """
        Create the filter dictionary to select the features to use from the feature constraints.

        Parameters
        ----------

        feature_constraints
            List of features to use or regex to select the features
        
        Returns
        -------
        
        feature_filter
            Dictionary with the filter to select the features
        """

        if isinstance(feature_constraints, list):
            # List of features is given
            feature_filter = dict(items=feature_constraints)

        elif isinstance(feature_constraints, str):
            # Regex is given
            feature_filter = dict(regex=feature_constraints)
            
        else:
            # No constraints are given
            feature_filter = None
        
        return feature_filter
    
    # Main CV-related methods
    def run(self, cv_dimension: Union[int, None] = None) -> Union[pd.DataFrame, None]:
        """
        Runs the CV calculator.
        Overwrites the dimension in the configuration if provided.
        
        Parameters
        ----------
        
        cv_dimension : int
            Dimension of the CV. If None, the dimension from the configuration is used.
        
        Returns
        -------
        
        projected_training_data : pd.DataFrame
            Projected training data or None if the CV computation failed
        """
        
        # Overwrite the dimension from the configuration if provided
        if cv_dimension:
            self.cv_dimension = cv_dimension
        
        # Compute the CV weights using the training data
        self.compute_cv()
        
        self.set_labels()
       
        # If the CV was computed successfully
        if self.cv is not None:
            
            # NOTE: using max min normalization on the projected training data
            self.normalize_cv()
            
            self.project_training_data()
            
            # Set the cv labels to the projected training data
            self.projected_training_data.columns = self.cv_labels
            
            # Return file labels to the projected training data
            self.projected_training_data["label"] = self.training_data_labels
            
            # NOTE: If the CV is normalized, this data might fall outside the expected range - check when plotting
            self.project_supplementary_data() 
            
            if self.projected_supplementary_data is not None:
                # Set the cv labels to the projected supplementary data
                self.projected_supplementary_data.columns = self.cv_labels
                # Return file labels to the projected supplementary data
                self.projected_supplementary_data["label"] = self.supplementary_data_labels
            
            self.save_cv()
            
            self.sensitivity_analysis()

        return self.projected_training_data
        
    def compute_cv(self):
        """
        Computes the collective variables. Implement in subclasses.
        """
        
        raise NotImplementedError

    def save_cv(self):
        """
        Saves the collective variable weights to a file. Implement in subclasses.
        """
        
        raise NotImplementedError

    def get_cv_parameters(self) -> Dict:
        """
        Returns the parameters for the CV. Implement in subclasses.
        """
        
        raise NotImplementedError
    
    def get_cv_type(self) -> str:
        """
        Returns the type of the CV. Implement in subclasses.
        """
        
        raise NotImplementedError
    
    def project_training_data(self):
        """ 
        Projects the training data onto the CV space.
        Implement in subclasses.
        """
        
    def project_supplementary_data(self):
        """
        Projects the supplementary data onto the CV space. 
        Implement in subclasses.
        """
        
        raise NotImplementedError
          
    def set_labels(self):
        """
        Sets the labels of the CV.
        """
        
        self.cv_labels = [f'{cv_components_map[self.cv_name]} {i+1}' for i in range(self.cv_dimension)]
    
    def normalize_cv(self):
        """
        Min max normalization of the CV.
        Normalizes the collective variable space to the range [-1, 1]
        Using the min and max values from the evaluation of the training data.
        Implemented in subclasses.
        """
        
        raise NotImplementedError
        
    def write_plumed_input(self, topology: str, output_folder: str) -> None:
        """
        Creates a plumed input file that computes the collective variable from the features 
        for the given topology.
        
        Parameters
        ----------
        
        topology : str
            Path to the topology file
            
        output_folder : str
            Path to the output folder where the plumed input file and the mda topology will be saved
        """
        from deep_cartograph.modules.plumed.input.builder import ComputeCVBuilder, ComputeEnhancedSamplingBuilder
        from deep_cartograph.modules.plumed.features import FeatureTranslator

        # Save new PLUMED-compliant topology
        plumed_topology_path = os.path.join(output_folder, 'plumed_topology.pdb')
        md.create_pdb(topology, plumed_topology_path)
        
        # Save new temporary reference PLUMED-compliant topology
        ref_plumed_topology_path = os.path.join(output_folder, 'ref_plumed_topology.pdb')
        md.create_pdb(self.ref_topology_path, ref_plumed_topology_path)
        
        # Translate the features from the reference topology to this topology
        features_list = FeatureTranslator(ref_plumed_topology_path, plumed_topology_path, self.feature_labels).run()
        
        # Construct builder arguments for these features and this CV
        builder_args = {
            'input_path': os.path.join(output_folder, f'plumed_input_{self.cv_name}.dat'),
            'topology_path': plumed_topology_path,
            'feature_list': features_list,
            'traj_stride': 1,
            'cv_type': self.get_cv_type(),
            'cv_params': self.get_cv_parameters()
        }
        
        # Build the plumed input file to track the CV
        plumed_builder = ComputeCVBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_out.dat')
        
        builder_args.update({
            'sampling_method': self.bias["method"], 
            'sampling_params': self.bias["args"],
            'input_path': os.path.join(output_folder, f'plumed_input_{self.cv_name}_{self.bias["method"]}.dat')
            })
            
        # Build the plumed input file to perform enhanced sampling
        plumed_builder = ComputeEnhancedSamplingBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_{self.bias["method"]}_out.dat')
        
        # Erase the temporary reference topology
        os.remove(ref_plumed_topology_path)

    def sensitivity_analysis(self):
        """
        Perform a sensitivity analysis of the CV on the training data.
        Implemented in subclasses.
        """
        raise NotImplementedError("Sensitivity analysis not implemented for this CV calculator.")
    
    # Getters
    def get_projected_sup_data(self) -> Union[pd.DataFrame, None]:
        """
        Returns the projected supplementary data. 
        May be None if no supplementary data was provided or the CV computation failed.
        """
        
        return self.projected_supplementary_data
    
    def get_labels(self) -> List[str]:
        """
        Returns the labels of the collective variable.
        """
        
        return self.cv_labels
    
    def get_cv_dimension(self) -> int:
        """
        Returns the dimension of the collective variables.
        """
        
        return self.cv_dimension

    def get_range(self) -> List[Tuple[float, float]]:
        """
        Returns the limits of the collective variable.
        """
        
        return self.cv_range

# Subclass for linear collective variables calculators
class LinearCalculator(CVCalculator):
    """
    Base class for linear collective variable calculators
    """
    
    def __init__(self, 
        configuration: Dict, 
        train_colvars_paths: List[str], 
        train_topology_paths: Optional[List[str]] = None,
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """ 
        Initializes a linear CV calculator.
        """
        super().__init__(
            configuration, 
            train_colvars_paths, 
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths, 
            output_path)
                
        # Main attributes
        self.cv: Union[np.array, None] = None 
        self.weights_path: Union[str, None] = None 
        
        # Normalize the data NOTE: I'm assuming here that after init is called I will just need the normalized data
        self.training_data: pd.DataFrame = self.normalize_data(self.training_data, self.features_norm_mean, self.features_norm_range)
    
        # If supplementary data is provided, normalize it as well
        if self.supplementary_data is not None:
            self.supplementary_data = self.normalize_data(self.supplementary_data, self.features_norm_mean, self.features_norm_range)
            
    def normalize_data(self, 
                       data: pd.DataFrame,
                       normalizing_mean: np.array,
                       normalizing_range: np.array,
        ) -> pd.DataFrame:
        """
        Use the normalization mean and range to normalize the data.
        
        Parameters
        ----------
        
        data : pd.DataFrame
            Data to normalize
        
        normalizing_mean : np.array
            Mean values for normalization
        
        normalizing_range : np.array
            Range values for normalization
            
        Returns
        -------
        
        normalized_data : pd.DataFrame
            Normalized data
        """
        
        # Normalize the data in place
        for feature, m, r in zip(data.columns, normalizing_mean, normalizing_range):
            data[feature] = (data[feature] - m) / r
        
        return data
        
    def save_cv(self):
        """
        Saves the collective variable linear weights to a text file.
        """
        
        # Path to output weights
        self.weights_path = os.path.join(self.output_path, f'{self.cv_name}_weights.txt')
        
        np.savetxt(self.weights_path, self.cv, fmt='%.7g')
        
        if 'mean_std' in self.feats_norm_mode:
            np.savetxt(os.path.join(self.output_path, 'features_mean.txt'), self.features_stats['mean'], fmt='%.7g')
            np.savetxt(os.path.join(self.output_path, 'features_std.txt'), self.features_stats['std'], fmt='%.7g')
        elif 'min_max' in self.feats_norm_mode:
            np.savetxt(os.path.join(self.output_path, 'features_max.txt'), self.features_stats['max'], fmt='%.7g')
            np.savetxt(os.path.join(self.output_path, 'features_min.txt'), self.features_stats['min'], fmt='%.7g')
        
        logger.info(f'Collective variable weights saved to {self.weights_path}')

    def get_cv_parameters(self):
        """ 
        Get the collective variable parameters.
        """

        # Save CV data to parameters dictionary
        cv_parameters = {
            'cv_name': self.cv_name,
            'cv_dimension': self.cv_dimension,
            'features_norm_mode': self.feats_norm_mode,
            'features_norm_mean': self.features_norm_mean,
            'features_norm_range': self.features_norm_range,
            'cv_stats': self.cv_stats,
            'weights': self.cv
        }
        return cv_parameters
    
    def get_cv_type(self) -> str:
        """
        Returns the type of the collective variable.
        """
        
        return 'linear'
    
    def project_training_data(self):
        """
        Projects the training data onto the CV space.
        """
        
        logger.info(f'Projecting training data onto {cv_names_map[self.cv_name]} ...')
        
        # Project the training data onto the CV space
        self.projected_training_data = self.training_data @ self.cv

        # Max min normalization between -1 and 1
        normalizing_mean = (self.cv_stats['max'] + self.cv_stats['min']) / 2
        normalizing_range = (self.cv_stats['max'] - self.cv_stats['min']) / 2

        # Normalize the projected training data
        self.projected_training_data = self.normalize_data(self.projected_training_data, 
                                                           normalizing_mean, 
                                                           normalizing_range)
        
    def project_supplementary_data(self):
        """
        Projects the supplementary data onto the CV space.
        """
        
        # If supplementary data is not empty
        if self.supplementary_data is not None:
            
            logger.info(f'Projecting supplementary data onto {cv_names_map[self.cv_name]} ...')
             
            # Project the supplementary data onto the CV space NOTE: check the dimensions make sense and match
            self.projected_supplementary_data = self.supplementary_data @ self.cv
            
            # Max min normalization between -1 and 1
            normalizing_mean = (self.cv_stats['max'] + self.cv_stats['min']) / 2
            normalizing_range = (self.cv_stats['max'] - self.cv_stats['min']) / 2
            
            # Normalize the projected supplementary data
            self.projected_supplementary_data = self.normalize_data(self.projected_supplementary_data, 
                                                                    normalizing_mean, 
                                                                    normalizing_range)
            
    def normalize_cv(self):
        
        # Project the normalized training data onto the CV space
        projected_training_data = pd.DataFrame(self.training_data @ self.cv) # NOTE: is this needed or already a dataframe?
        
        # Compute statistics of the projected training data
        stats = ['min', 'max']
        stats_df = projected_training_data.agg(stats).T 
        self.cv_stats = {stat: stats_df[stat].to_numpy() for stat in stats}
        
        # Save the max/min values of each dimension - part of the final cv definition
        np.savetxt(os.path.join(self.output_path, 'cv_max.txt'), self.cv_stats['max'], fmt='%.7g')
        np.savetxt(os.path.join(self.output_path, 'cv_min.txt'), self.cv_stats['min'], fmt='%.7g')

    def sensitivity_analysis(self):
        """  
        Perform a sensitivity analysis of the CV on the training data.
        """
        
        from deep_cartograph.modules.figures import plot_sensitivity_results
        
        # For a linear CV, the sensitivity of each feature is given by the absolute value of its coefficient in the CV weights
        cv_sensitivities = np.abs(self.cv)
        
        # For each dimension of the CV
        for cv_index in range(cv_sensitivities.shape[1]):
            
            # Create directory for sensitivity analysis results
            sensitivity_output_path = os.path.join(self.output_path, f'sensitivity_analysis_{cv_index+1}')
            os.makedirs(sensitivity_output_path, exist_ok=True)
            
            sensitivities = cv_sensitivities[:, cv_index]
            logger.info(f'Shape of sensitivities for CV dimension {cv_index}: {sensitivities.shape}')

            # Order the sensitivities from lowest to highest, order the feature labels accordingly
            indices = np.argsort(sensitivities)
            sensitivities = sensitivities[indices]
            feature_labels = np.array(self.feature_labels)[indices]
        
            # Create a Dictionary with the results
            results = {
                'feature_names': feature_labels,
                'sensitivity': {
                    'Dataset' : sensitivities,
                },
                'gradients': {
                    # Not used but expected by plot_sensitivity_results
                    'Dataset': np.array([[x] for x in sensitivities]) 
                }
            }
            
            # Debug: print their shape and length
            logger.info(f'Sensitivities shape: {sensitivities.shape}, length: {len(sensitivities)}')
            logger.info(f'Feature labels shape: {feature_labels.shape}, length: {len(feature_labels)}')
            
            # Print values for the top 10 features
            logger.info("Top 10 features sensitivities:")
            for i in range(min(10, len(sensitivities))):
                logger.info(f"{feature_labels[i]}: {sensitivities[i]}")
                
                
            # Save the sensitivities to a file
            sensitivity_df = pd.DataFrame({'sensitivity': sensitivities}, index = feature_labels)
            sensitivity_path = os.path.join(sensitivity_output_path, 'sensitivity_analysis.csv')
            sensitivity_df.to_csv(sensitivity_path)

            # Plot the top sensitivities
            plot_sensitivity_results(results, modes=['barh'], output_folder=sensitivity_output_path)

        return

# Subclass for non-linear collective variables calculators
class NonLinear(CVCalculator):
    """
    Non-linear collective variables calculator (e.g. Autoencoder)
    """
    
    def __init__(self, 
        configuration: Dict, 
        train_colvars_paths: List[str], 
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None,
        sup_topology_paths: Optional[List[str]] = None, 
        output_path: Union[str, None] = None
        ):
        """ 
        Initializes a non-linear CV calculator.
        """
        
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        
        from mlcolvar.utils.trainer import MetricsCallback
        from mlcolvar.cvs import AutoEncoderCV, DeepTICA, VariationalAutoEncoderCV
        
        import torch 
                
        super().__init__(
            configuration,
            train_colvars_paths,  
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)

        self.nonlinear_cv_map: Dict = {
            'ae': AutoEncoderCV,
            'deep_tica': DeepTICA,
            'vae': VariationalAutoEncoderCV
        }
        
        # Main attributes
        self.cv: Union[AutoEncoderCV, DeepTICA, VariationalAutoEncoderCV, None] = None
        self.checkpoint: Union[ModelCheckpoint, None] = None
        self.metrics: Union[MetricsCallback, None] = None
        self.weights_path: Union[str, None] = None
        
        # Training configuration
        self.training_config: Dict = configuration['training'] 
        self.general_config: Dict = self.training_config['general']
        self.early_stopping_config: Dict  = self.training_config['early_stopping']
        self.optimizer_config: Dict = self.training_config['optimizer']
        self.lr_scheduler_config: Optional[Dict] = self.training_config['lr_scheduler_config']
        self.lr_scheduler: Optional[Dict] = self.training_config['lr_scheduler']
        
        # Training attributes
        self.max_tries: int = self.general_config['max_tries']
        self.seed: int = self.general_config['seed']
        self.training_validation_lengths: List = self.general_config['lengths']
        self.batch_size: int = self.general_config['batch_size']
        self.shuffle: bool = self.general_config['shuffle']
        self.random_split: bool = self.general_config['random_split']
        self.max_epochs: int = self.general_config['max_epochs']
        self.check_val_every_n_epoch: int = self.general_config['check_val_every_n_epoch']
        self.save_check_every_n_epoch: int = self.general_config['save_check_every_n_epoch']
        
        self.best_model_score: Union[float, None] = None
        self.tries: int = 0
        
        self.patience: int = self.early_stopping_config['patience']
        self.min_delta: float = self.early_stopping_config['min_delta']
                
        # Neural network settings
        self.encoder_config: Dict = self.architecture_config['encoder']
        self.decoder_config: Optional[Dict] = self.architecture_config['decoder'] 
        
        # Set layers from NN config 
        self.encoder_layers: List = self.set_encoder_layers()
        self.decoder_layers: Optional[List] = self.set_decoder_layers()
        
        # Set options from NN config
        self.encoder_options: Dict = self.encoder_config
        self.encoder_options.pop('layers', {})
        if self.decoder_config is not None:
            self.decoder_options = self.decoder_config
            self.decoder_options.pop('layers', {})
        else:
            self.decoder_options = self.encoder_options
        
        # Add activation function to the last layer of the decoder based on the normalization of the features
        if isinstance(self.decoder_options['activation'], list):
            # Normalized using min max with range [0, 1] -> use sigmoid activation
            if self.feats_norm_mode == 'min_max_range1':
                self.decoder_options['activation'].append('custom_sigmoid')
            # Normalized using min max with range [-1, 1] -> use tanh activation
            elif self.feats_norm_mode == 'min_max_range2':
                self.decoder_options['activation'].append('tanh')
            # In other cases (mean_std or none), avoid adding an activation function
            else:
                self.decoder_options['activation'].append(None)
        else:
            logger.warning('''Decoder activation function is a single value, the same activation will be used for all layers. 
                           Make sure the chosen activation function matches the normalization of the features when doing regression
                           or choose last_layer_activation = False.''')
        
        # We add a layer to the decoder
        
        # No dropout for the last layer of the decoder
        if isinstance(self.decoder_options['dropout'], list):
            self.decoder_options['dropout'].append(None)
        # No batch norm for the last layer of the decoder 
        if isinstance(self.decoder_options['batchnorm'], list):
            self.decoder_options['batchnorm'].append(False)
        elif self.decoder_options['batchnorm'] is True:
            logger.warning('''Batch normalization is set to True for all layers of the decoder, 
                           including the last layer. Make sure this is intended or choose batchnorm as a list.''')
        
        # Normalization of features in the Non-linear models
        # No normalization
        if self.feats_norm_mode == 'none':
            self.cv_options: Dict = {'norm_in' : None}
        # Corresponding normalization, see prepare_normalization() in base class
        else:
            self.cv_options: Dict = {'norm_in' : {'mode' : 'mean_std',
                                                  'mean': torch.tensor(self.features_norm_mean),
                                                  'range': torch.tensor(self.features_norm_range)}}

        # Optimizer
        self.opt_name: str = self.optimizer_config['name']
        self.optimizer_options: Dict = self.optimizer_config['kwargs']
        
        self.cv_options["optimizer"] = self.optimizer_options
        
        # Construct the lr_scheduler option - scheduler class and its kwargs
        if self.lr_scheduler is not None:           
            # Obtain the class from the name
            lr_scheduler_class = getattr(torch.optim.lr_scheduler, self.lr_scheduler['name'], None)
            if lr_scheduler_class is None:
                logger.error(f'Learning rate scheduler {self.lr_scheduler["name"]} not recognized. Exiting...')
                sys.exit(1)
                
            self.cv_options["lr_scheduler"] = {
                "scheduler" : lr_scheduler_class    
            }   
            
            self.cv_options["lr_scheduler"].update(self.lr_scheduler.get('kwargs', {}))
            
            # NEW FEATURE
            # If OneCycleLR is used -> Adjust epochs and steps_per_epoch according to the training configuration and the number of samples
            # If ReduceLROnPlateau is used -> Adjust patience and cooldown
            # Check where do we have the info to do the update
            
        # Construct the lr_scheduler_config option
        if self.lr_scheduler_config is not None:
            self.cv_options["lr_scheduler_config"] = self.lr_scheduler_config
    
    def _adjust_lr_scheduler(self, datamodule):
        """
        Adjusts LR scheduler parameters based on the training configuration.
        This is called right after the datamodule is created.
        """
        
        datamodule.setup(stage='fit')
        
        # Proceed only if a scheduler is defined
        if self.lr_scheduler is None:
            return

        logger.debug("Adjusting LR Scheduler parameters...") 

        scheduler_name = self.lr_scheduler.get('name', '')
        
        if scheduler_name == 'OneCycleLR':
            
            # Give reasonable default values if not provided in the configuration
            self.cv_options["lr_scheduler"]['max_lr'] = self.cv_options["lr_scheduler"].get('max_lr', 1e-3)
            self.cv_options["lr_scheduler"]['epochs'] = self.cv_options["lr_scheduler"].get('epochs', self.max_epochs)
            steps_per_epoch = len(datamodule.train_dataloader())
            self.cv_options["lr_scheduler"]['steps_per_epoch'] = self.cv_options["lr_scheduler"].get('steps_per_epoch', steps_per_epoch)
            
            # Adjust the interval from the configuration to 'step'
            self.cv_options["lr_scheduler_config"]["interval"] = 'step'
            
            logger.info(f"OneCycleLR configured with epochs = {self.cv_options['lr_scheduler']['epochs']}, steps_per_epoch = {self.cv_options['lr_scheduler'].get('steps_per_epoch')}")
            
        elif scheduler_name == 'ReduceLROnPlateau':
            
            # Give reasonable default values if not provided in the configuration
            self.cv_options["lr_scheduler"]['patience'] = self.cv_options["lr_scheduler"].get('patience', max(1, self.patience // 4, self.max_epochs // 10))
            self.cv_options["lr_scheduler"]['cooldown'] = self.cv_options["lr_scheduler"].get('cooldown', max(1, self.patience // 8, self.max_epochs // 20))
            
            # Adjust the interval from the configuration to 'epoch'
            self.cv_options["lr_scheduler_config"]["interval"] = 'epoch'
            
            logger.info(f"ReduceLROnPlateau configured with patience = {self.cv_options['lr_scheduler']['patience']}, cooldown = {self.cv_options['lr_scheduler']['cooldown']}")
            
    def check_batch_size(self):
        
       # Get the number of samples in the training set
        self.num_training_samples = int(self.num_samples*self.training_validation_lengths[0])
        
        # Check the batch size is not larger than the number of samples in the training set
        if self.batch_size >= self.num_training_samples:
            self.batch_size = closest_power_of_two(self.num_samples*self.training_validation_lengths[0])
            logger.warning(f"""The batch size is larger than the number of samples in the training set. 
                           Setting the batch size to the closest power of two: {self.batch_size}""")
    
    def set_encoder_layers() -> List:
        """ 
        Set the layers for the encoder of the non-linear model.
        Implement in subclasses.
        
        self.encoder: [input_dim, hidden_layer_1, hidden_layer_2, ..., output_dim]
            input_dim: number of features
            hidden_layer_i: number of neurons in the i-th hidden layer
            output_dim: dimension of the collective variable (latent space)
            
        Each non-linear model has different assumptions about the encoder layers input: ae, vae, deep_tica
            
        Returns
        -------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def set_decoder_layers(self) -> Optional[List]:
        """ 
        Set the layers for the decoder of the non-linear model.
        Implement in subclasses.
        
        self.decoder: [input_dim, hidden_layer_1, hidden_layer_2, ..., output_dim]
            input_dim: dimension of the collective variable (latent space)
            hidden_layer_i: number of neurons in the i-th hidden layer
            output_dim: number of features
            
        Each non-linear model has different assumptions about the decoder layers input: ae, vae
        
        Returns
        -------
        
        nn_layers : Optional[List]
            List with the layers for the decoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
            If None, no decoder is used (e.g. DeepTICA).
        """
        
        return None
        
    def create_model(self):
        
        """
        Implement this method in subclasses to create the non-linear model.
        
        Creates the non-linear model based on the configuration. 
        This is needed because there is no common API for all non-linear models in the mlcolvars library,
        thus the creation of the model has to be done in each specific CV calculator.
        
        Returns
        -------
        
        model : AutoEncoderCV, DeepTICA, VariationalAutoEncoderCV
            Non-linear model object
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_callbacks(self) -> List:
        """ 
        Get the callbacks for the training of the Nonlinear model.
        """
        
        from mlcolvar.utils.trainer import MetricsCallback
    
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        
        general_callbacks = []
        
        # Define MetricsCallback to store the loss
        self.metrics = MetricsCallback()
        general_callbacks.append(self.metrics)

        # Define EarlyStopping callback to stop training
        self.early_stopping = EarlyStopping(
            monitor="valid_loss", 
            min_delta=self.min_delta, 
            patience=self.patience, 
            mode = "min")
        general_callbacks.append(self.early_stopping)

        # Define ModelCheckpoint callback to save the best/last model
        self.checkpoint = ModelCheckpoint(
            dirpath=self.output_path,                  # Directory to save the checkpoints  
            monitor="valid_loss",                      # Quantity to monitor
            save_last=True,                            # Save the last checkpoint - useful for VAE
            save_top_k=1,                              # Number of best models to save according to the quantity monitored
            save_weights_only=False,                   # Save only the weights
            filename=None,                             # Default checkpoint file name '{epoch}-{step}'
            mode="min",                                # Best model is the one with the minimum monitored quantity
            every_n_epochs=self.save_check_every_n_epoch)   # Number of epochs between checkpoints
        general_callbacks.append(self.checkpoint)

        return general_callbacks

    def train(self) -> bool:
        """
        Trains the non-linear collective variable using the training data.
        
        Returns
        -------
        
        successfully_trained : bool
            True if the training was successful, False otherwise.
        """
        
        import torch
        import lightning
        
        from mlcolvar.data import DictModule
    
        logger.info(f'Training {cv_names_map[self.cv_name]} ...')
        
        # Training was successful
        successfully_trained = False
        
        # Train until model finds a good solution
        while not successfully_trained and self.tries < self.max_tries:
            try: 

                self.tries += 1

                logger.debug(f'Splitting the dataset...')

                # Build datamodule, split the dataset into training and validation
                datamodule = DictModule(
                    random_split = self.random_split,
                    dataset = self.training_input_dtset,
                    lengths = self.training_validation_lengths,
                    batch_size = self.batch_size,
                    shuffle = self.shuffle, 
                    generator = torch.manual_seed(self.seed + self.tries))

                self._adjust_lr_scheduler(datamodule)
                
                logger.debug(f'Initializing {cv_names_map[self.cv_name]} object...')
                
                # Define non-linear model
                model = self.create_model()

                # Set optimizer name
                model.optimizer_name = self.opt_name

                logger.info(f"Model architecture: {model}")
                
                logger.debug(f'Initializing metrics and callbacks...')
                
                logger.debug(f'Initializing Trainer...')

                # Define trainer
                trainer = lightning.Trainer(          
                    callbacks=self.get_callbacks(),
                    max_epochs=self.max_epochs, 
                    logger=False, 
                    enable_checkpointing=True,
                    enable_progress_bar = False, 
                    check_val_every_n_epoch=self.check_val_every_n_epoch)

                logger.debug(f'Training...')

                trainer.fit(model, datamodule)

                # Get validation and training loss
                validation_loss = self.metrics.metrics['valid_loss']

                # Check the evolution of the loss
                successfully_trained = self.loss_decreased(validation_loss)
                if not successfully_trained:
                    logger.warning(f'{cv_names_map[self.cv_name]} has not found a good solution. Re-starting training...')

            except Exception as e:
                logger.error(f'{cv_names_map[self.cv_name]} training failed. Error message: {e}')
                logger.info(f'Retrying {cv_names_map[self.cv_name]} training...')
        
        # Check if the checkpoint exists
        if successfully_trained:
            
            if self.cv_name == 'vae':
                # Save last model - regularized model is preferred over the best model
                model_path = os.path.join(self.output_path, 'last.ckpt')
            else:
                # Save lowest loss model
                model_path = self.checkpoint.best_model_path

            if os.path.exists(model_path):
                self.cv = self.nonlinear_cv_map[self.cv_name].load_from_checkpoint(model_path)
                self.best_model_score = self.checkpoint.best_model_score
                logger.info(f'Lowest score during training: {self.best_model_score}')
            else:
                logger.error(f'The model checkpoint {model_path} does not exist.')
        else:
            logger.error(f'{cv_names_map[self.cv_name]} has not converged after {self.max_tries} tries.')
    
        return successfully_trained
            
    def loss_decreased(self, loss: List):
        """
        Check if the loss has decreased by the end of the training.

        Inputs
        ------

            loss:         loss for each epoch.
        """

        # Soft convergence condition: Check if the minimum of the validation loss is lower than the initial value
        if min(loss) > loss[0]:
            logger.warning('Validation loss has not decreased by the end of the training.')
            return False

        return True
    
    def save_loss(self):
        """
        Saves the loss of the training and common metrics to all CVs
        """
        from mlcolvar.utils.plot import plot_metrics
        import torch
        
        # Create a new dictionary with all tensor metrics moved to the CPU
        cpu_metrics = {}
        for key, metric_list in self.metrics.metrics.items():
            # Check if the list is not empty and its first element is a tensor
            if metric_list and isinstance(metric_list[0], torch.Tensor):
                cpu_metrics[key] = [v.cpu().numpy() for v in metric_list] # Convert to numpy array directly
            else:
                cpu_metrics[key] = metric_list # Assume it's already CPU-compatible
        
        try:        
            # Move the best_model_score tensor to the CPU
            # Check if it's a tensor before calling .cpu()
            best_model_score_cpu = self.best_model_score
            if isinstance(self.best_model_score, torch.Tensor):
                best_model_score_cpu = best_model_score_cpu.cpu().detach()
            
            # Save the loss if requested
            if self.training_config['save_loss']:
                metrics_to_save = ['train_loss', 'valid_loss', 'epoch']
                for key in metrics_to_save:
                    if key not in cpu_metrics:
                        logger.warning(f'Metric {key} not found in metrics. It will not be saved.')
                        continue
                    filepath = os.path.join(self.output_path, f'{key}.npy')
                    np.save(filepath, np.array(cpu_metrics[key]))
                np.savetxt(os.path.join(self.output_path, 'model_score.txt'), np.array([best_model_score_cpu]), fmt='%.7g')
                    
            # Plot general metrics to all Non-linear CVs
            

            # Training and Validation loss
            Loss_present = ('train_loss' in cpu_metrics) and ('valid_loss' in cpu_metrics)
            if Loss_present:
                loss_plot_config = {
                    'keys': ['train_loss', 'valid_loss'],
                    'labels': ['Training', 'Validation'],
                    'colors': ['fessa1', 'fessa5'],
                    'linestyles': ['-', '-'],
                    'yscale': 'log'
                }
                ax = plot_metrics(cpu_metrics, **loss_plot_config)
                ax.figure.savefig(os.path.join(self.output_path, f'loss.png'), dpi=300, bbox_inches='tight')
                ax.figure.clf()
            else:
                logger.warning('Training and/or validation loss not found in metrics. Loss plot will not be generated.')
                
            # Learning rate
            if 'lr' in cpu_metrics:
                lr_plot_config = {
                    'keys': ['lr'],
                    'labels': ['Learning Rate'],
                    'colors': ['fessa2'],
                    'linestyles': ['-'],
                    'yscale': 'log'
                }
                ax = plot_metrics(cpu_metrics, **lr_plot_config)
                ax.figure.savefig(os.path.join(self.output_path, f'learning_rate.png'), dpi=300, bbox_inches='tight')
                ax.figure.clf()
            else:
                logger.warning('Learning rate not found in metrics. Learning rate plot will not be generated.')

        except Exception as e:
            import traceback
            logger.error(f'Failed to save/plot the loss. Error message: {e}\n{traceback.format_exc()}')

    def normalize_cv(self):
        
        import torch
        
        from mlcolvar.core.transform import Normalization
        from mlcolvar.core.transform.utils import Statistics
        
        # Data projected onto original latent space of the best model
        # The feature normalization is included in the model
        # We move the data to the device of the model (GPU or CPU) manually
        with torch.no_grad():
            self.cv.postprocessing = None
            projected_training_data = self.cv(self.training_input_dtset[:]['data'].to(self.cv.device))
        
        # Compute statistics of the projected training data
        stats = Statistics(projected_training_data)
        
        # Normalize the latent space
        norm =  Normalization(self.cv_dimension, mode='min_max', stats = stats )
        self.cv.postprocessing = norm
        
    def compute_cv(self):
        """
        Compute Non-linear CV.
        """

        # Train the non-linear model
        successfully_trained = self.train()  
        
        # If training was successful
        if successfully_trained:
        
            # Save the loss 
            self.save_loss()

            # After training, put model in evaluation mode - needed for cv normalization and data projection
            self.cv.eval()

    def save_cv(self):
        """
        Saves the collective variable model to a PyTorch TorchScript file.
        """

        # Path to output model
        self.weights_path = os.path.join(self.output_path, f'{self.cv_name}_model.pt')

        if self.cv is None:
            logger.error('No collective variable model to save.')
            return

        successfully_saved = False
        try:
            # The model is set to evaluation mode before tracing
            self.cv.eval() 
            self.cv.to_torchscript(file_path=self.weights_path, method='trace')
            logger.info(f'Collective variable model saved to {self.weights_path}')
            successfully_saved = True
        except Exception as e:
            logger.error(f'Failed to save TorchScript model using trace mode. Error: {e}')
            
        if not successfully_saved:
            logger.info('Attempting to save the model using script mode instead of trace...')
            try:
                # Attempt to save the model using script mode
                self.cv.to_torchscript(file_path=self.weights_path, method='script')
                logger.info(f'Collective variable model saved to {self.weights_path} using script mode')
            except Exception as e:
                logger.error(f'Failed to save TorchScript model using script mode. Error: {e}')
                
    def get_cv_parameters(self):
        """
        Get the collective variable parameters.
        """
        cv_parameters = {
            'cv_name': self.cv_name,
            'cv_dimension': self.cv_dimension,
            'weights_path': self.weights_path
        }
        return cv_parameters
    
    def get_cv_type(self) -> str:
        """
        Returns the type of the collective variable.
        """
        
        return "non-linear"
    
    def project_supplementary_data(self):
        """
        Projects the supplementary data onto the CV space.
        """
        
        import torch
        
        # If supplementary data is not empty
        if self.supplementary_data is not None:
            
            logger.info(f'Projecting supplementary data onto {cv_names_map[self.cv_name]} ...')

            with torch.no_grad():
                # Move data to the device of the model (GPU or CPU)
                supplementary_data_on_model_device = torch.tensor(self.supplementary_data.values).to(self.cv.device)
                # Project the supplementary data onto the CV space
                projected_sup_tensor = self.cv(supplementary_data_on_model_device)
            
            # Move to CPU and convert to numpy array
            projected_sup_array = projected_sup_tensor.cpu().numpy()
            
            self.projected_supplementary_data = pd.DataFrame(projected_sup_array, columns=self.cv_labels)

    def project_training_data(self):
        """
        Projects the training data onto the CV space.
        """
        import torch 
        
        logger.info(f'Projecting training data onto {cv_names_map[self.cv_name]} ...')
        
        # Project the training data onto the CV space
        with torch.no_grad():
            # Move data to the device of the model (GPU or CPU)
            training_data_on_model_device = torch.tensor(self.training_data.values).to(self.cv.device)
            # Project the training data onto the CV space
            projected_training_tensor = self.cv(training_data_on_model_device)
            
        # Move to CPU and convert to numpy array
        projected_training_array = projected_training_tensor.cpu().numpy()
            
        self.projected_training_data = pd.DataFrame(projected_training_array, columns=self.cv_labels)   

    def sensitivity_analysis(self):
        """  
        Perform a sensitivity analysis of the CV on the training data.
        """
        from mlcolvar.explain import sensitivity_analysis
        
        from deep_cartograph.modules.figures import plot_sensitivity_results
        
        # Create directory for sensitivity analysis results
        sensitivity_output_path = os.path.join(self.output_path, 'sensitivity_analysis')
        os.makedirs(sensitivity_output_path, exist_ok=True)
        
        # Compute the sensitivity analysis
        results = sensitivity_analysis(self.cv, self.training_input_dtset, metric="mean_abs_val", 
                                       feature_names=None, per_class=False, plot_mode=None)
        
        # Save the sensitivities to a file
        sensitivity_df = pd.DataFrame({'sensitivity': results['sensitivity']['Dataset']}, index = results['feature_names'])
        sensitivity_path = os.path.join(sensitivity_output_path, 'sensitivity_analysis.csv')
        sensitivity_df.to_csv(sensitivity_path)
        
        # Plot the sensitivity results
        modes = ['barh', 'violin']
        plot_sensitivity_results(results, modes=modes, output_folder=sensitivity_output_path)

# Collective variables calculators
class PCACalculator(LinearCalculator):
    """
    Principal component analysis calculator.
    """

    def __init__(self, 
        configuration: Dict,
        train_colvars_paths: List[str], 
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """
        Initializes the PCA calculator.
        """
        
        super().__init__(
            configuration,
            train_colvars_paths, 
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)
        
        self.cv_name = 'pca'
        
        self.initialize_cv()
        
    def compute_cv(self):
        """
        Compute Principal Component Analysis (PCA) on the input features. 
        """
        
        # Create PCA object
        pca = PCA(n_components=self.cv_dimension)
        
        # Fit the PCA model
        pca.fit(self.training_data.to_numpy())
        
        # Save the eigenvectors as CVs
        self.cv = pca.components_.T
        
        # Follow a criteria for the sign of the eigenvectors - first weight of each eigenvector should be positive
        for i in range(self.cv_dimension):
            if self.cv[0,i] < 0:
                self.cv[:,i] = -self.cv[:,i]
                
class TICACalculator(LinearCalculator):
    """ 
    Time-lagged independent component analysis calculator.
    """
    
    def __init__(self, 
        configuration: Dict,
        train_colvars_paths: List[str],  
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """
        Initializes the TICA calculator.
        """
        
        from mlcolvar.utils.timelagged import create_timelagged_dataset
        
        super().__init__(
            configuration,
            train_colvars_paths, 
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)
        
        self.cv_name = 'tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag) NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data.to_numpy(), lag_time=self.configuration['lag_time'])
        
        self.initialize_cv()
        
    def compute_cv(self):
        """
        Compute Time-lagged Independent Component Analysis (TICA) on the input features. 
        """
        from mlcolvar.core.stats import TICA
        
        # Use TICA to compute slow linear combinations of the input features
        # Here out_features is the number of eigenvectors to keep
        tica_algorithm = TICA(in_features = self.num_features, out_features=self.cv_dimension)

        try:
            # Compute TICA
            _, tica_eigvecs = tica_algorithm.compute(data=[self.training_input_dtset['data'], self.training_input_dtset['data_lag']], save_params = True, remove_average = True)
        except Exception as e:
            logger.error(f'TICA could not be computed. Error message: {e}')
            return

        # Save the first cv_dimension eigenvectors as CVs
        self.cv = tica_eigvecs.numpy()
  
class HTICACalculator(LinearCalculator):
    """ 
    Hierarchical Time-lagged independent component analysis calculator.
    
    See: 
    
    Pérez-Hernández, Guillermo, and Frank Noé. “Hierarchical Time-Lagged Independent Component Analysis: 
    Computing Slow Modes and Reaction Coordinates for Large Molecular Systems.” Journal of Chemical Theory 
    and Computation 12, no. 12 (December 13, 2016): 6118–29. https://doi.org/10.1021/acs.jctc.6b00738.
    """
    def __init__(self, 
        configuration: Dict,
        train_colvars_paths: List[str], 
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """
        Initializes the HTICA calculator.
        """
        
        from mlcolvar.utils.timelagged import create_timelagged_dataset
        
        super().__init__(
            configuration,
            train_colvars_paths,  
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)
        
        self.cv_name = 'htica'
        
        self.num_subspaces = configuration['num_subspaces']
        self.subspaces_dimension = configuration['subspaces_dimension']
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag)
        # NOTE: Are we duplicating the data here? :(
        # NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data.to_numpy(), lag_time=self.configuration['lag_time'])
        
        self.initialize_cv()
        
    
    def compute_cv(self):
        """
        Compute Hierarchical Time-lagged Independent Component Analysis (TICA) on the input features. 
        
        Initial space of features (num_features) -> TICA LEVEL 1 (subspaces_dimension x num_subspaces) -> TICA LEVEL 2 (CV_dimension)
        
            1. Divide the original dataset into num_subspaces
            2. Compute TICA on each sub-space (TICA LEVEL 1)
            3. Project each sub-space onto the TICA eigenvectors of LEVEL 1
            3. Construct the sparse - block diagonal - matrix transforming the original features into TICA LEVEL 1
            4. Compute TICA on the concatenated projected data (TICA LEVEL 2)
            5. Obtain the transformation matrix from features to TICA LEVEL 2 (final CV)
        """
        import torch
        from mlcolvar.core.stats import TICA
        
        data_tensor = self.training_input_dtset['data']
        data_lag_tensor = self.training_input_dtset['data_lag'] 
        
        # Split the data tensor into 10 tensors using torch_split
        data_tensors = torch.split(data_tensor, self.num_features//self.num_subspaces, dim=1)
        
        # Split the data lag tensor into 10 tensors using torch_split
        data_lag_tensors = torch.split(data_lag_tensor, self.num_features//self.num_subspaces, dim=1)
        
        # Initialize the eigenvectors and eigenvalues
        level_1_eigvecs = []     
        
        # Projected data
        projected_data = []
        projected_data_lag = []
        
        # Compute TICA on each of these pairs of tensors
        for data, data_lag in zip(data_tensors, data_lag_tensors):
            
            # Initialize the TICA object
            tica_algorithm = TICA(in_features = data.shape[1], out_features=self.subspaces_dimension)
            
            try:
                # Compute TICA
                _, eigvecs = tica_algorithm.compute(data=[data, data_lag], save_params = True, remove_average = True)
            except Exception as e:
                logger.error(f'TICA could not be computed. Error message: {e}')
                return
            
            # Save the eigenvectors and eigenvalues
            level_1_eigvecs.append(eigvecs.numpy())
            
            # Project each of the tensors onto the eigenvectors
            projected_data.append(torch.matmul(data, eigvecs))
            projected_data_lag.append(torch.matmul(data_lag, eigvecs))
        
        # Create the matrix that converts from the space of features to TICA LEVEL 1
        Transform_level_1_TICA = block_diag(level_1_eigvecs, format='csr') 
        
        # Concatenate the projected tensors
        projected_data = torch.concatenate(projected_data, axis=1)
        projected_data_lag = torch.concatenate(projected_data_lag, axis=1)
        
        # Apply TICA to the concatenated dataset
        tica_algorithm = TICA(in_features = projected_data.shape[1], out_features=self.cv_dimension)
        
        try:
            # Compute TICA
            _, level_2_eigvecs = tica_algorithm.compute(data=[projected_data, projected_data_lag], save_params = True, remove_average = True)
        except Exception as e:
            logger.error(f'TICA could not be computed. Error message: {e}')
            return
        
        # Obtain the transformation matrix from features to TICA LEVEL 2
        self.cv = Transform_level_1_TICA @ level_2_eigvecs
           
class AECalculator(NonLinear):
    """
    Autoencoder calculator.
    """
    def __init__(self, 
        configuration: Dict,
        train_colvars_paths: List[str],  
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """
        Initializes the Autoencoder calculator.
        """
        import torch 
        
        from mlcolvar.data import DictDataset
        
        super().__init__(
            configuration,
            train_colvars_paths, 
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths,
            sup_topology_paths, 
            output_path)
        
        # Create DictDataset NOTE: we have to find another solution as this will duplicate the data
        dictionary = {"data": torch.Tensor(self.training_data.values)}
        self.training_input_dtset = DictDataset(dictionary, feature_names=self.feature_labels)
        
        self.cv_name = 'ae'
        
        self.initialize_cv()
        
        self.check_batch_size()
        
        # NOTE: In the future we might want to allow for a custom last layer before the latent space
        # Currently it will be the same as the one for the hidden layers - if not a list
        # or no activation/dropout is applied to the last layer
        
        # Add encoder activation and dropout option for last layer if these are lists
        if isinstance(self.encoder_options['dropout'], list):
            self.encoder_options['dropout'].append(None)
        if isinstance(self.encoder_options['activation'], list):
            self.encoder_options['activation'].append(None)

        # Update options
        cv_options = {
            "encoder": self.encoder_options,
            "decoder": self.decoder_options,
            "optimizer": self.optimizer_options
        }
        self.cv_options.update(cv_options)

    def set_encoder_layers(self) -> List:
        """ 
        Set the layers for the encoder of the Autoencoder
        
        Return
        ------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        return [self.num_features] + self.architecture_config['encoder']['layers'] + [self.cv_dimension]
    
    def set_decoder_layers(self) -> Optional[List]:
        """ 
        Set the layers for the decoder of the Autoencoder
        
        Return
        ------
        
        nn_layers : Optional[List]
            List with the layers for the decoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
            If None, no decoder is used (e.g. DeepTICA).
        """
        
        if self.architecture_config['decoder'] is None:
            return None
        else:
            return [self.cv_dimension] + self.architecture_config['decoder']['layers'] + [self.num_features]
        
    def create_model(self):
        """ 
        Create the Autoencoder model.
        
        Returns
        -------
        
        model : AutoEncoderCV
        """
        
        from mlcolvar.cvs import AutoEncoderCV
        
        model = AutoEncoderCV(
            encoder_layers=self.encoder_layers, 
            decoder_layers=self.decoder_layers,
            options=self.cv_options)
         
        return model
        
class DeepTICACalculator(NonLinear):
    """
    DeepTICA calculator.
    """
    def __init__(self,  
        configuration: Dict, 
        train_colvars_paths: List[str], 
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """
        Initializes the DeepTICA calculator.
        """      
        
        from mlcolvar.utils.timelagged import create_timelagged_dataset
        
        super().__init__(
            configuration,
            train_colvars_paths,
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)
        
        self.cv_name = 'deep_tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag) NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data, lag_time=self.configuration['lag_time'])
        
        self.initialize_cv()
        
        self.check_batch_size()
            
        # NOTE: In the future we might want to allow for a custom last layer before the latent space
        # Currently it will be the same as the one for the hidden layers - if not a list
        # or no activation/dropout is applied to the last layer
        
        # Add encoder activation and dropout option for last layer if these are lists
        if isinstance(self.encoder_options['dropout'], list):
            self.encoder_options['dropout'].append(None)
        if isinstance(self.encoder_options['activation'], list):
            self.encoder_options['activation'].append(None)
            
        # Update options
        cv_options = {
            "nn": self.encoder_options
        }
        self.cv_options.update(cv_options)

    def set_encoder_layers(self) -> List:
        """ 
        Set the layers for the encoder of DeepTICA
        
        Return
        ------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        return [self.num_features] + self.encoder_config['layers'] + [self.cv_dimension]
    
    def create_model(self):
        """
        Create the DeepTICA model.
        
        Returns
        -------
        
        model : DeepTICA
            DeepTICA model object.
        """
        
        from mlcolvar.cvs import DeepTICA
        
        model = DeepTICA(
            layers=self.encoder_layers,
            options=self.cv_options)
        
        return model
    
    def save_cv(self):
        """
        Save the eigenvectors and eigenvalues of the best model.
        """
        from mlcolvar.utils.plot import plot_metrics
        
        super().save_cv()
            
        # Find the epoch where the best model was found
        best_index = self.metrics.metrics['valid_loss'].index(self.best_model_score)
        best_epoch = self.metrics.metrics['epoch'][best_index]
        logger.info(f'Took {best_epoch} epochs')

        # Find eigenvalues of the best model
        best_eigvals = [self.metrics.metrics[f'valid_eigval_{i+1}'][best_index] for i in range(self.cv_dimension)]
        for i in range(self.cv_dimension):
            logger.info(f'Eigenvalue {i+1}: {best_eigvals[i]}')
            
        np.savetxt(os.path.join(self.output_path, 'eigenvalues.txt'), np.array(best_eigvals), fmt='%.7g')
        
        # Plot eigenvalues
        ax = plot_metrics(self.metrics.metrics,
                            labels=[f'Eigenvalue {i+1}' for i in range(self.cv_dimension)], 
                            keys=[f'valid_eigval_{i+1}' for i in range(self.cv_dimension)],
                            ylabel='Eigenvalue',
                            yscale=None)

        # Save figure
        ax.figure.savefig(os.path.join(self.output_path, f'eigenvalues.png'), dpi=300, bbox_inches='tight')
        ax.figure.clf()

class VAECalculator(NonLinear):
    """
    Variational Autoencoder calculator.
    """
    def __init__(self, 
        configuration: Dict,
        train_colvars_paths: List[str],  
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None
        ):
        """
        Initializes the Variational Autoencoder calculator.
        """
        import torch 
        
        from mlcolvar.data import DictDataset
        
        super().__init__(
            configuration,
            train_colvars_paths, 
            train_topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths,
            sup_topology_paths, 
            output_path)
        
        # VAE-specific settings
        self.kl_annealing_config = self.training_config['kl_annealing']
        self.type = self.kl_annealing_config['type']
        self.start_beta = self.kl_annealing_config['start_beta']
        self.max_beta = self.kl_annealing_config['max_beta']
        self.start_epoch = self.kl_annealing_config['start_epoch']
        self.n_cycles = self.kl_annealing_config['n_cycles']
        self.n_epochs_anneal = self.kl_annealing_config['n_epochs_anneal']
        
        # Create DictDatase
        dictionary = {"data": torch.Tensor(self.training_data.values)}
        self.training_input_dtset = DictDataset(dictionary, feature_names=self.feature_labels)
        
        self.cv_name = 'vae'
        
        self.initialize_cv()
        
        self.check_batch_size()
        
        # If the activation functions / dropout are given as a list, add one for the last layer 
        # Needed due to the addition of a n_cvs layer before passing it to Feed Forward in VAE model
            
        # Update options
        nn_options = {
            "encoder": self.encoder_options,
            "decoder": self.decoder_options
        }
        self.cv_options.update(nn_options)

    def set_encoder_layers(self) -> List:
        """ 
        Set the layers for the VAE
        
        Here the model already includes a mean and variance layer with
        cv_dimension outputs, so we do not need to add them explicitly.
        
        Return
        ------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        return [self.num_features] + self.encoder_config['layers']
    
    def set_decoder_layers(self):
        """ 
        Set the layers for the decoder of the VAE
        
        Here the model already includes a layer with the latent space dimension
        as input, so we do not need to add it explicitly.
        
        Return
        ------
        
        nn_layers : Optional[List]
            List with the layers for the decoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
            If None, no decoder is used (e.g. DeepTICA).
        """
        
        if self.decoder_config is None:
            return None
        else:
            return self.decoder_config['layers'] + [self.num_features]
   
    def create_model(self):
        """
        Create the Variational Autoencoder model.
        
        Returns
        -------
        
        model : VariationalAutoEncoderCV
            Variational Autoencoder model object.
        """
        
        from mlcolvar.cvs import VariationalAutoEncoderCV
        
        model = VariationalAutoEncoderCV(
            n_cvs=self.cv_dimension,
            encoder_layers=self.encoder_layers, 
            decoder_layers=self.decoder_layers,
            options=self.cv_options)
         
        return model
        
    def get_callbacks(self) -> List:
        """ 
        Get the callbacks for the VAE training.
        
        Uses the parent class method to get the callbacks and adds the VAE-specific callbacks.
        
        Returns
        -------
        
        callbacks: List
            List of callbacks for the VAE training.
        """
        import deep_cartograph.modules.ml as ml
        
        general_callbacks = super().get_callbacks()
        
        kl_annealing_args = {
            'type': self.type,
            'start_beta': self.start_beta,
            'max_beta': self.max_beta,
            'start_epoch': self.start_epoch,
            'n_cycles': self.n_cycles,
            'n_epochs_anneal': self.n_epochs_anneal
        }
        kl_anneal_callback = ml.KLAAnnealing(**kl_annealing_args)
        general_callbacks.append(kl_anneal_callback)
        
        # If ReduceLROnPlateau is the scheduler, add our custom manager
        if self.lr_scheduler and self.lr_scheduler.get('name') == 'ReduceLROnPlateau':
            # The start epoch should be when the kl annealing finishes
            start_monitoring_epoch = self.start_epoch + self.n_epochs_anneal
        
            lr_plateau_manager = ml.LROnPlateauManager(start_epoch=start_monitoring_epoch)
            general_callbacks.append(lr_plateau_manager)
        
        return general_callbacks
    
    def save_loss(self): 
        """ 
        Saves the loss of the training. Adds saving of VAE-specific metrics such as KL divergence.
        """      
        super().save_loss()
        
        from mlcolvar.utils.plot import plot_metrics
        import torch
        
        try:
            # Save the KL and reconstruction losses if requested
            if self.training_config['save_loss']:
                np.save(os.path.join(self.output_path, 'kl_divergence.npy'), np.array(self.metrics.metrics['train_kl_loss']))
                np.save(os.path.join(self.output_path, 'valid_kl_divergence.npy'), np.array(self.metrics.metrics['valid_kl_loss']))
                np.save(os.path.join(self.output_path, 'reconstruction_loss.npy'), np.array(self.metrics.metrics['train_reconstruction_loss']))
                np.save(os.path.join(self.output_path, 'valid_reconstruction_loss.npy'), np.array(self.metrics.metrics['valid_reconstruction_loss']))
                np.save(os.path.join(self.output_path, 'beta.npy'), np.array(self.metrics.metrics['beta']))

            # Create a dictionary with CPU-based data for plotting
            # This ensures the plotting function doesn't receive GPU tensors.
            metrics_for_plotting = self.metrics.metrics.copy()
            metrics_for_plotting['train_kl_loss'] = self.metrics.metrics['train_kl_loss']
            metrics_for_plotting['valid_kl_loss'] = self.metrics.metrics['valid_kl_loss']
            metrics_for_plotting['train_reconstruction_loss'] = self.metrics.metrics['train_reconstruction_loss']
            metrics_for_plotting['valid_reconstruction_loss'] = self.metrics.metrics['valid_reconstruction_loss']
            
            # Plot loss using the CPU-safe metrics dictionary # NOTE: are we assuming that we have one sample per epoch?
            ax = plot_metrics(metrics_for_plotting, 
                                labels=['Training KL', 'Validation KL', 'Training Reconstruction', 'Validation Reconstruction'], 
                                keys=['train_kl_loss', 'valid_kl_loss', 'train_reconstruction_loss', 'valid_reconstruction_loss'], 
                                linestyles=['-','-','-','-'], colors=['fessa1','fessa5','fessa2','fessa6'], 
                                yscale='log')
            # Save figure
            ax.figure.savefig(os.path.join(self.output_path, f'vae_loss.png'), dpi=300, bbox_inches='tight')
            ax.figure.clf()
            
            metrics_for_plotting['beta'] = self.metrics.metrics['beta']
            
            # Plot beta
            ax = plot_metrics(metrics_for_plotting, 
                                labels=['Beta'], 
                                keys=['beta'], 
                                linestyles=['-'], colors=['fessa3'], 
                                yscale='linear')
            
            # Save figure
            ax.figure.savefig(os.path.join(self.output_path, f'vae_beta.png'), dpi=300, bbox_inches='tight')
            ax.figure.clf()
            
        except Exception as e:
            import traceback
            logger.error(f'Failed to save/plot the loss. Error message: {e}\n{traceback.format_exc()}')
            
# Mappings
cv_calculators_map = {
    'pca': PCACalculator,
    'ae': AECalculator,
    'tica': TICACalculator,
    'htica': HTICACalculator,
    'deep_tica': DeepTICACalculator,
    'vae': VAECalculator
}

cv_names_map = {
    'pca': 'PCA',
    'ae': 'AE',
    'tica': 'TICA',
    'htica': 'HTICA',
    'deep_tica': 'DeepTICA',
    'vae': 'VAE'
}

cv_components_map = {
    'pca': 'PC',
    'ae': 'AE',
    'tica': 'TIC',
    'htica': 'HTIC',
    'deep_tica': 'DeepTIC',
    'vae': 'VAE'
}