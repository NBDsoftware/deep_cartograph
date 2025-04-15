import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import block_diag
from typing import Dict, List, Tuple, Union, Literal, Optional
from sklearn.decomposition import PCA       

from deep_cartograph.modules.common import closest_power_of_two, create_output_folder
import deep_cartograph.modules.plumed as plumed
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
        training_colvars_paths: List[str],
        topology_paths: Optional[List[str]] = None, 
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
            
        training_colvars_paths 
            List of paths to colvars files with the main data used for training
                
        topology_paths (Optional)
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
        self.feats_norm_mode: Literal['mean_std', 'min_max', 'none'] = configuration['features_normalization']
        
        # Colvars paths
        self.training_colvars_paths: List[str] = training_colvars_paths
        self.sup_colvars_paths: Optional[List[str]] = sup_colvars_paths
        
        # Topologies
        self.topologies: Optional[List[str]] = topology_paths     # NOTE: Should these be attributes?
        self.ref_topology_path: Optional[str] = ref_topology_path
        if self.topologies is not None:
            if self.ref_topology_path is None:
                self.ref_topology_path = self.topologies[0]
        
        # Filtered training data used to train / compute the CVs
        logger.info('Reading training data from colvars files...')
        reading_args = {
            'colvars_paths': training_colvars_paths,
            'topology_paths': self.topologies,
            'ref_topology_path': self.ref_topology_path,
            'load_args': self.training_reading_settings}
        self.training_data: pd.DataFrame = self.read_data(**reading_args)
        self.training_data_labels: np.array = self.training_data.pop("label").to_numpy()

        # Projected training data NOTE: make sure they have the labels from the training_data
        self.projected_training_data: Optional[pd.DataFrame] = None
        
        # Number of samples used to train / compute the CV - depends on the specific CV calculator
        self.num_training_samples: Union[int, None] = None
        
        # List of features used for training (features in the colvars files after filtering) 
        self.feature_labels: List[str] = self.training_data.columns.tolist()
        self.num_features: int = len(self.feature_labels)
        logger.info(f'Number of features: {self.num_features}')
        
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
    
            # Check if the feature labels are the same as the training data
            sup_feature_labels = self.supplementary_data.columns.tolist()
            if len(sup_feature_labels) != self.num_features:
                logger.error(f"""Number of features in supplementary colvars files ({len(sup_feature_labels)}) is different 
                             from the number of features in the training dataset ({self.num_features}). Exiting...""")
                sys.exit(1)
            if self.feature_labels != sup_feature_labels:
                logger.error(f"""Feature labels in supplementary colvars files ({sup_feature_labels}) are different 
                             from the feature labels in the training dataset ({self.feature_labels}). Exiting...""")
                sys.exit(1)
        else:
            self.supplementary_data: Optional[pd.DataFrame] = None
        
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
        if load_args is None:
            load_args = {}
            
        df = plumed.colvars.create_dataframe_from_files(
            colvars_paths = colvars_paths,
            topology_paths = topology_paths,
            reference_topology = ref_topology_path,   
            filter_args = self.feature_filter,  
            create_labels = True,
            **load_args
        )
        
        return df
    
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
    
    # Main methods
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
            
            self.save_projected_supplementary_data()
            
            self.save_cv()
            
            self.write_plumed_input()

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

    def project_training_data(self):
        """ 
        Projects the training data onto the CV space and saves the projection to a csv file.
        Implement in subclasses.
        """
        
    def project_supplementary_data(self):
        """
        Projects the supplementary data onto the CV space and saves the projection to a csv file. 
        Implement in subclasses.
        """
        
        raise NotImplementedError
    
    def save_projected_supplementary_data(self):
        """
        Saves the projected supplementary data to csv files, if there is any.
        """
        
        supplementary_data_folder = os.path.join(self.output_path, 'supplementary_data')
        
        if self.projected_supplementary_data is not None:
            
            if not os.path.exists(supplementary_data_folder):
                os.makedirs(supplementary_data_folder)
        
            # Find the names of the supplementary colvars paths
            sup_colvars_names = [Path(path).stem for path in self.sup_colvars_paths]
            
            # For each supplementary colvars
            for i in range(len(sup_colvars_names)):
                
                # Get path to projected data
                projected_data_path = os.path.join(supplementary_data_folder, f'{sup_colvars_names[i]}.csv')
                
                logger.debug(f'Saving supplementary data from {self.sup_colvars_paths[i]} to {projected_data_path}')
                
                # Extract the projected data for this colvars file
                projected_data = self.projected_supplementary_data[self.projected_supplementary_data["label"] == i]
                
                # Drop file_label column
                projected_data = projected_data.drop(columns=["label"])
                
                # Save the projected data to a csv file NOTE: We are assuming that the dataframe has the correct labels
                projected_data.to_csv(projected_data_path, index=False, float_format='%.4f')
          
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
        
    def write_plumed_input(self):
        """ 
        Create a plumed input file that computes the collective variable from the features. 
        Implement in subclasses.
        """
        
        raise NotImplementedError

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
        training_colvars_paths: List[str], 
        topology_paths: Optional[List[str]] = None,
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
            training_colvars_paths, 
            topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths, 
            output_path)
                
        # Main attributes
        self.cv: Union[np.array, None] = None 
        self.weights_path: Union[str, None] = None 
        
        # NOTE: This will be have to be moved to the base class - it is expected in this way by the assembler of the plumed input - non-linear calculators can use this info to build their own objects
        # Compute training data statistics
        stats = ['mean', 'std', 'min', 'max']
        stats_df = self.training_data.agg(stats).T
        self.features_stats: Dict[str, np.array] = {stat: stats_df[stat].to_numpy() for stat in stats}
        
        # Normalize the data NOTE: I'm assuming here that after init is called I will just need the normalized data
        self.training_data: pd.DataFrame = self.normalize_data(self.training_data, self.features_stats, self.feats_norm_mode)
    
        if self.supplementary_data is not None:
            self.supplementary_data = self.normalize_data(self.supplementary_data, self.features_stats, self.feats_norm_mode)
            
    def normalize_data(self, 
        data: pd.DataFrame, 
        stats: Dict[str, np.array], 
        normalization_mode: Literal['mean_std', 'min_max', 'none']
    ) -> pd.DataFrame:
        """
        Use the stats and the normalization mode to normalize the data
        
        Parameters
        ----------
        
        data : pd.DataFrame
            Data to normalize
            
        stats : Dict[str, np.array]
            Dictionary with the statistics to use for normalization
        
        normalization_mode : str
            Normalization mode. Can be 'mean_std', 'min_max', 'none'
            
        Returns
        -------
        
        normalized_data : pd.DataFrame
            Normalized data
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
        if normalization_mode == 'none':
            means = np.zeros(len(stats["mean"]))
            ranges = np.ones(len(stats["mean"]))
        elif normalization_mode == 'mean_std':
            means = stats["mean"]
            ranges = stats["std"]
        elif normalization_mode == 'min_max':
            means = (stats["min"] + stats["max"]) / 2
            ranges = (stats["max"] - stats["min"]) / 2
        else:
            logger.error(f'Normalization mode {normalization_mode} not recognized. Exiting...')
            sys.exit(1)
        
        # Check the ranges are not close to zero
        ranges = sanitize_ranges(ranges)
        
        # Normalize the data in place
        for feature, m, r in zip(data.columns, means, ranges):
            data[feature] = (data[feature] - m) / r
        
        return data
        
    def save_cv(self):
        """
        Saves the collective variable linear weights to a text file.
        """
        
        # Path to output weights
        self.weights_path = os.path.join(self.output_path, f'{self.cv_name}_weights.txt')
        
        np.savetxt(self.weights_path, self.cv, fmt='%.7g')
        
        if self.feats_norm_mode == 'mean_std':
            np.savetxt(os.path.join(self.output_path, 'features_mean.txt'), self.features_stats['mean'], fmt='%.7g')
            np.savetxt(os.path.join(self.output_path, 'features_std.txt'), self.features_stats['std'], fmt='%.7g')
        elif self.feats_norm_mode == 'min_max':
            np.savetxt(os.path.join(self.output_path, 'features_max.txt'), self.features_stats['max'], fmt='%.7g')
            np.savetxt(os.path.join(self.output_path, 'features_min.txt'), self.features_stats['min'], fmt='%.7g')
        
        logger.info(f'Collective variable weights saved to {self.weights_path}')

    def project_training_data(self):
        """
        Projects the training data onto the CV space.
        """
        
        logger.info(f'Projecting training data onto {cv_names_map[self.cv_name]} ...')
        
        # Project the training data onto the CV space
        self.projected_training_data = self.training_data @ self.cv
        
        # Normalize the projected training data
        self.projected_training_data = self.normalize_data(self.projected_training_data, self.cv_stats, 'min_max')
        
    def project_supplementary_data(self):
        """
        Projects the supplementary data onto the CV space.
        """
        
        # If supplementary data is not empty
        if self.supplementary_data is not None:
            
            logger.info(f'Projecting supplementary data onto {cv_names_map[self.cv_name]} ...')
             
            # Project the supplementary data onto the CV space NOTE: check the dimensions make sense and match
            self.projected_supplementary_data = self.supplementary_data @ self.cv
            
            # Normalize the projected supplementary data
            self.projected_supplementary_data = self.normalize_data(self.projected_supplementary_data, self.cv_stats, 'min_max')
            
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
    
    def write_plumed_input(self):
        """
        Creates a plumed input file that computes the collective variable from the features.
        """
        
        # Save new PLUMED-compliant topology
        plumed_topology_path = os.path.join(self.output_path, 'plumed_topology.pdb')
        md.create_pdb(self.topologies[0], plumed_topology_path)
        
        # Save CV data to parameters dictionary
        cv_parameters = {
            'cv_name': self.cv_name,
            'cv_dimension': self.cv_dimension,
            'features_norm_mode': self.feats_norm_mode,
            'features_stats': self.features_stats,
            'cv_stats': self.cv_stats, # NOTE: the builder will assume max-min normalization for the cv
            'weights': self.cv
        }
        
        # Construct builder arguments
        builder_args = {
            'input_path': os.path.join(self.output_path, f'plumed_input_{self.cv_name}.dat'),
            'topology_path': plumed_topology_path,
            'feature_list': self.feature_labels,
            'traj_stride': 1,
            'cv_type': 'linear',
            'cv_params': cv_parameters
        }
        
        # Build the plumed input file to track the CV
        plumed_builder = plumed.input.builder.ComputeCVBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_out.dat')
        
        # Save enhanced sampling parameters to parameters dictionary
        sampling_params = {
            'sigma': 0.05,
            'height': 1.0,
            'biasfactor': 10.0,
            'temp': 300,
            'pace': 500,
            'grid_min': -1,
            'grid_max': 1,
            'grid_bin': 300
        }
        
        builder_args.update({
            'sampling_method': 'wt-metadynamics', 
            'sampling_params': sampling_params,
            'input_path': os.path.join(self.output_path, f'plumed_input_{self.cv_name}_metad.dat')
            })
            
        # Build the plumed input file to perform enhanced sampling
        plumed_builder = plumed.input.builder.ComputeEnhancedSamplingBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_metad_out.dat')

# Subclass for non-linear collective variables calculators
class NonLinear(CVCalculator):
    """
    Non-linear collective variables calculator (e.g. Autoencoder)
    """
    
    def __init__(self, 
        configuration: Dict, 
        training_colvars_paths: List[str], 
        topology_paths: Optional[List[str]] = None, 
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
        from mlcolvar.cvs import AutoEncoderCV, DeepTICA
                
        super().__init__(
            configuration,
            training_colvars_paths,  
            topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)

        self.nonlinear_cv_map: Dict = {
            'ae': AutoEncoderCV,
            'deep_tica': DeepTICA
        }
        
        # Main attributes
        self.cv: Union[AutoEncoderCV, DeepTICA, None] = None
        self.checkpoint: Union[ModelCheckpoint, None] = None
        self.metrics: Union[MetricsCallback, None] = None
        self.weights_path: Union[str, None] = None
        
        # Training configuration
        self.training_config: Dict = configuration['training'] 
        self.general_config: Dict = self.training_config['general']
        self.early_stopping_config: Dict  = self.training_config['early_stopping']
        self.optimizer_config: Dict = self.training_config['optimizer']
        
        # Training attributes
        self.max_tries: int = self.general_config['max_tries']
        self.seed: int = self.general_config['seed']
        self.training_validation_lengths: List = self.general_config['lengths']
        self.batch_size: int = self.general_config['batch_size']
        self.shuffle: bool = self.general_config['shuffle']
        self.random_split: bool = self.general_config['random_split']
        self.max_epochs: int = self.general_config['max_epochs']
        self.dropout: float = self.general_config['dropout']
        self.check_val_every_n_epoch: int = self.general_config['check_val_every_n_epoch']
        self.save_check_every_n_epoch: int = self.general_config['save_check_every_n_epoch']
        
        self.best_model_score: Union[float, None] = None
        self.converged: bool = False
        self.tries: int = 0
        
        self.patience: int = self.early_stopping_config['patience']
        self.min_delta: float = self.early_stopping_config['min_delta']
        
        self.hidden_layers: List = self.architecture_config['hidden_layers']
        
        # Neural network settings
        self.nn_layers: List = [self.num_features] + self.hidden_layers + [self.cv_dimension]
        self.nn_options: Dict = {'activation': 'shifted_softplus', 'dropout': self.dropout} 
        
        # Normalization of features in the Non-linear models: min_max or mean_std
        if self.feats_norm_mode == 'min_max':
            self.cv_options: Dict = {'norm_in' : {'mode' : 'min_max'}}
        elif self.feats_norm_mode == 'mean_std':
            self.cv_options: Dict = {'norm_in' : {'mode' : 'mean_std'}}
        elif self.feats_norm_mode == 'none':
            self.cv_options: Dict = {'norm_in' : None}
        
        # Optimizer
        self.opt_name: str = self.optimizer_config['name']
        self.optimizer_options: Dict = self.optimizer_config['kwargs']
    
    def check_batch_size(self):
        
       # Get the number of samples in the training set
        self.num_training_samples = int(self.num_samples*self.training_validation_lengths[0])
        
        # Check the batch size is not larger than the number of samples in the training set
        if self.batch_size >= self.num_training_samples:
            self.batch_size = closest_power_of_two(self.num_samples*self.training_validation_lengths[0])
            logger.warning(f"""The batch size is larger than the number of samples in the training set. 
                           Setting the batch size to the closest power of two: {self.batch_size}""")
            
    def train(self):
        
        import torch
        import lightning
        
        from mlcolvar.data import DictModule
        from mlcolvar.utils.trainer import MetricsCallback
        
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    
        logger.info(f'Training {cv_names_map[self.cv_name]} ...')
        
        # Train until model finds a good solution
        while not self.converged and self.tries < self.max_tries:
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
        
                logger.debug(f'Initializing {cv_names_map[self.cv_name]} object...')
                
                # Define non-linear model
                model = self.nonlinear_cv_map[self.cv_name](self.nn_layers, options=self.cv_options)

                # Set optimizer name
                model._optimizer_name = self.opt_name
                
                logger.info(f"Model architecture: {model}")
                
                logger.debug(f'Initializing metrics and callbacks...')

                # Define MetricsCallback to store the loss
                self.metrics = MetricsCallback()

                # Define EarlyStopping callback to stop training
                early_stopping = EarlyStopping(
                    monitor="valid_loss", 
                    min_delta=self.min_delta, 
                    patience=self.patience, 
                    mode = "min")

                # Define ModelCheckpoint callback to save the best model
                self.checkpoint = ModelCheckpoint(
                    dirpath=self.output_path,
                    monitor="valid_loss",                      # Quantity to monitor
                    save_last=False,                           # Save the last checkpoint
                    save_top_k=1,                              # Number of best models to save according to the quantity monitored
                    save_weights_only=True,                    # Save only the weights
                    filename=None,                             # Default checkpoint file name '{epoch}-{step}'
                    mode="min",                                # Best model is the one with the minimum monitored quantity
                    every_n_epochs=self.save_check_every_n_epoch)   # Number of epochs between checkpoints
                
                logger.debug(f'Initializing Trainer...')

                # Define trainer
                trainer = lightning.Trainer(          
                    callbacks=[self.metrics, early_stopping, self.checkpoint],
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
                self.converged = self.model_has_converged(validation_loss)
                if not self.converged:
                    logger.warning(f'{cv_names_map[self.cv_name]} has not found a good solution. Re-starting training...')

            except Exception as e:
                logger.error(f'{cv_names_map[self.cv_name]} training failed. Error message: {e}')
                logger.info(f'Retrying {cv_names_map[self.cv_name]} training...')
        
        # Check if the checkpoint exists
        if self.converged:
            
            if os.path.exists(self.checkpoint.best_model_path):
                # Load the best model
                self.cv = self.nonlinear_cv_map[self.cv_name].load_from_checkpoint(self.checkpoint.best_model_path)
                os.remove(self.checkpoint.best_model_path)
                
                # Find the score of the best model
                self.best_model_score = self.checkpoint.best_model_score
                logger.info(f'Best model score: {self.best_model_score}')
            else:
                logger.error('The best model checkpoint does not exist.')
        else:
            logger.error(f'{cv_names_map[self.cv_name]} has not converged after {self.max_tries} tries.')
            
    def model_has_converged(self, validation_loss: List):
        """
        Check if there is any problem with the training of the model.

        - Check if the validation loss has decreased by the end of the training.
        - Check if we have at least 'patience' x 'check_val_every_n_epoch' epochs.

        Inputs
        ------

            validation_loss:         Validation loss for each epoch.
        """

        # Soft convergence condition: Check if the minimum of the validation loss is lower than the initial value
        if min(validation_loss) > validation_loss[0]:
            logger.warning('Validation loss has not decreased by the end of the training.')
            return False

        # Check if we have at least 'patience' x 'check_val_every_n_epoch' epochs
        if len(validation_loss) < self.patience*self.check_val_every_n_epoch:
            logger.warning('The trainer did not run for enough epochs.')
            return False

        return True
    
    def save_loss(self):
        """
        Saves the loss of the training.
        """
        from mlcolvar.utils.plot import plot_metrics
        
        try:        
            # Save the loss if requested
            if self.training_config['save_loss']:
                np.save(os.path.join(self.output_path, 'train_loss.npy'), np.array(self.metrics.metrics['train_loss']))
                np.save(os.path.join(self.output_path, 'valid_loss.npy'), np.array(self.metrics.metrics['valid_loss']))
                np.save(os.path.join(self.output_path, 'epochs.npy'), np.array(self.metrics.metrics['epoch']))
                np.savetxt(os.path.join(self.output_path, 'model_score.txt'), np.array([self.best_model_score]), fmt='%.7g')
                
            # Plot loss
            ax = plot_metrics(self.metrics.metrics, 
                                labels=['Training', 'Validation'], 
                                keys=['train_loss', 'valid_loss'], 
                                linestyles=['-','-'], colors=['fessa1','fessa5'], 
                                yscale='log')

            # Save figure
            ax.figure.savefig(os.path.join(self.output_path, f'loss.png'), dpi=300, bbox_inches='tight')
            ax.figure.clf()

        except Exception as e:
            logger.error(f'Failed to save/plot the loss. Error message: {e}')

    def normalize_cv(self):
        
        import torch
        
        from mlcolvar.core.transform import Normalization
        from mlcolvar.core.transform.utils import Statistics
        
        # Data projected onto original latent space of the best model - feature normalization included in the model
        with torch.no_grad():
            self.cv.postprocessing = None
            projected_training_data = self.cv(self.training_input_dtset[:]['data'])

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
        self.train()  
        
        # Save the loss 
        self.save_loss()

        # After training, put model in evaluation mode - needed for cv normalization and data projection
        self.cv.eval()
        
    def save_cv(self):
        """
        Saves the collective variable non-linear weights to a pytorch script file.
        """
        
        # Path to output model
        self.weights_path = os.path.join(self.output_path, f'{self.cv_name}_weights.pt')
        
        if self.cv is None:
            logger.error('No collective variable to save.')
            return

        self.cv.to_torchscript(file_path = self.weights_path, method='trace') # NOTE: check if this also saves the normalization layer
        
        logger.info(f'Collective variable weights saved to {self.weights_path}')

    def project_supplementary_data(self):
        """
        Projects the supplementary data onto the CV space.
        """
        
        import torch
        
        # If supplementary data is not empty
        if self.supplementary_data is not None:
            
            logger.info(f'Projecting supplementary data onto {cv_names_map[self.cv_name]} ...')

            with torch.no_grad():
                projected_sup_array = self.cv(torch.tensor(self.supplementary_data.values)).numpy()
            
            self.projected_supplementary_data = pd.DataFrame(projected_sup_array, columns=self.cv_labels)

    def project_training_data(self):
        """
        Projects the training data onto the CV space.
        """
        import torch 
        
        logger.info(f'Projecting training data onto {cv_names_map[self.cv_name]} ...')
        
        # Project the training data onto the CV space
        with torch.no_grad():
            projected_training_array = self.cv(torch.tensor(self.training_data.values)).numpy() 
            
        self.projected_training_data = pd.DataFrame(projected_training_array, columns=self.cv_labels)   
    
    def write_plumed_input(self):
        """
        Creates a plumed input file that computes the collective variable from the features.
        """
        
        # Save new PLUMED-compliant topology
        plumed_topology_path = os.path.join(self.output_path, 'plumed_topology.pdb')
        md.create_pdb(self.topologies[0], plumed_topology_path)
        
        cv_parameters = {
            'cv_name': self.cv_name,
            'cv_dimension': self.cv_dimension,
            'weights_path': self.weights_path
        }
        
        builder_args = {
            'input_path': os.path.join(self.output_path, f'plumed_input_{self.cv_name}.dat'),
            'topology_path': plumed_topology_path,
            'feature_list': self.feature_labels,
            'traj_stride': 1,
            'cv_type': 'non-linear',
            'cv_params': cv_parameters
        }
        
        # Build the plumed input file to track the CV
        plumed_builder = plumed.input.builder.ComputeCVBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_out.dat')
        
        # Save enhanced sampling parameters to parameters dictionary
        sampling_params = {
            'sigma': 0.05,
            'height': 1.0,
            'biasfactor': 10.0,
            'temp': 300,
            'pace': 500,
            'grid_min': -1,
            'grid_max': 1,
            'grid_bin': 300
        }
        
        builder_args.update({
            'sampling_method': 'wt-metadynamics',
            'sampling_params': sampling_params,
            'input_path': os.path.join(self.output_path, f'plumed_input_{self.cv_name}_metad.dat')
            })
        
        # Build the plumed input file to perform enhanced sampling
        plumed_builder = plumed.input.builder.ComputeEnhancedSamplingBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_metad_out.dat')

# Collective variables calculators
class PCACalculator(LinearCalculator):
    """
    Principal component analysis calculator.
    """

    def __init__(self, 
        configuration: Dict,
        training_colvars_paths: List[str], 
        topology_paths: Optional[List[str]] = None, 
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
            training_colvars_paths, 
            topology_paths, 
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
        training_colvars_paths: List[str],  
        topology_paths: Optional[List[str]] = None, 
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
            training_colvars_paths, 
            topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)
        
        self.cv_name = 'tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag) NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data.to_numpy(), lag_time=self.architecture_config['lag_time'])
        
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
        training_colvars_paths: List[str], 
        topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        feature_constraints: Union[List[str], str, None] = None, 
        sup_colvars_paths: Optional[List[str]] = None, 
        sup_topology_paths: Optional[List[str]] = None,
        output_path: Union[str, None] = None):
        """
        Initializes the HTICA calculator.
        """
        
        from mlcolvar.utils.timelagged import create_timelagged_dataset
        
        super().__init__(
            configuration,
            training_colvars_paths,  
            topology_paths, 
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
        self.training_input_dtset = create_timelagged_dataset(self.training_data.to_numpy(), lag_time=self.architecture_config['lag_time'])
        
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
        training_colvars_paths: List[str],  
        topology_paths: Optional[List[str]] = None, 
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
            training_colvars_paths, 
            topology_paths, 
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
        
        # Update options
        self.cv_options.update({"encoder": self.nn_options,
                                "decoder": self.nn_options,
                                "optimizer": self.optimizer_options})
        
class DeepTICACalculator(NonLinear):
    """
    DeepTICA calculator.
    """
    def __init__(self,  
        configuration: Dict, 
        training_colvars_paths: List[str], 
        topology_paths: Optional[List[str]] = None, 
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
            training_colvars_paths,
            topology_paths, 
            ref_topology_path, 
            feature_constraints, 
            sup_colvars_paths, 
            sup_topology_paths,
            output_path)
        
        self.cv_name = 'deep_tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag) NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data, lag_time=self.architecture_config['lag_time'])
        
        self.initialize_cv()
        
        self.check_batch_size()
            
        # Update options
        self.cv_options.update({"nn": self.nn_options,
                                "optimizer": self.optimizer_options})
        
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

# Mappings
cv_calculators_map = {
    'pca': PCACalculator,
    'ae': AECalculator,
    'tica': TICACalculator,
    'htica': HTICACalculator,
    'deep_tica': DeepTICACalculator
}

cv_names_map = {
    'pca': 'PCA',
    'ae': 'AE',
    'tica': 'TICA',
    'htica': 'HTICA',
    'deep_tica': 'DeepTICA'
}

cv_components_map = {
    'pca': 'PC',
    'ae': 'AE',
    'tica': 'TIC',
    'htica': 'HTIC',
    'deep_tica': 'DeepTIC'
}