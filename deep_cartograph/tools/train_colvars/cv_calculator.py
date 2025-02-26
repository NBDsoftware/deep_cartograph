import os
import re
import sys
import torch
import lightning
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import block_diag
from typing import Dict, List, Tuple, Union, Literal
from sklearn.decomposition import PCA       

from mlcolvar.utils.timelagged import create_timelagged_dataset # NOTE: this function returns less samples than expected: N-lag_time-2
from mlcolvar.utils.io import create_dataset_from_files
from mlcolvar.data import DictModule, DictDataset
from mlcolvar.cvs import AutoEncoderCV, DeepTICA
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.core.stats import TICA
from mlcolvar.core.transform import Normalization
from mlcolvar.core.transform.utils import Statistics

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from deep_cartograph.modules.common import closest_power_of_two, create_output_folder

# Set logger
logger = logging.getLogger(__name__)

# Base class for collective variables calculators
class CVCalculator:
    """
    Base class for collective variables calculators.
    """
    def __init__(self, colvars_paths: List[str], topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the base CV calculator.
        
        NOTE: for the comparison of different topologies, we will need the topologies to make sure that the features are 
        equivalent and all the colvars files can be used together to learn a CV. 
        
        Parameters
        ----------
        
        colvars_paths : str
            List of paths to colvars files with the main data used for training
        topology_paths : str
            List of paths to topology files corresponding to the colvars files
        feature_constraints : Union[List[str], str]
            List with the features to use for the training or str with regex to filter feature names.
        ref_colvars_paths : Union[List[str], None]
            List of paths to colvars files with reference data
        configuration : Dict
            Configuration dictionary for the CV
        output_path : str
            Output path where the CV results folder will be created
        """
        
        # Training data
        self.training_input_dtset: DictDataset = None  # Used to train / compute the CVs, contains just the samples defined in training_reading_settings
        
        self.num_features: int = None
        self.num_samples: int = None
        
        # Reference data
        self.ref_datasets: List[DictDataset] = []
        
        # Filter dictionary
        self.feature_filter: Union[Dict, None] = self.get_feature_filter(feature_constraints)
        
        # List of features used for training (features in the colvars file after filtering) NOTE: this will be a list of lists when we consider different topologies
        self.features: List[str] = self.read_features(colvars_paths[0])      
        
        # Configuration
        self.configuration: Dict = configuration
        self.architecture_config: Dict = configuration['architecture']
        self.training_reading_settings: Dict = configuration['input_colvars']
        self.feats_norm_mode: Union[Literal['mean_std', 'min_max'], None] = configuration['features_normalization']
        self.feats_normalization: Union[Normalization, None] = None
        
        # Read the data
        self.read_training_data(colvars_paths)
        self.read_reference_data(ref_colvars_paths)
        
        self.ref_names: List[str] = [Path(path).stem for path in ref_colvars_paths] if ref_colvars_paths else []
        
        # Topologies
        self.topologies: List[str] = topology_paths
        
        # General CV attributes
        self.cv_dimension: int = configuration['dimension']
        self.cv_labels: List[str] = []
        self.cv_name: str = None
        self.cv_range: List[Tuple[float, float]] = []
        
        # Output 
        self.projected_ref: List[np.ndarray] = []
    
        self.output_path: str =  output_path
    
    def initialize(self):
        """
        Initializes the specific CV calculator:
        
            - Finds the number of samples from the input dataset
            - Creates the output folder for the CV using the cv_name
            - Logs the start of the calculation using the cv_name
        """
        
        # Get the number of samples - input_dataset depends on the specific CV calculator
        self.num_samples = self.training_input_dtset["data"].shape[0]
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
    def read_training_data(self, colvars_paths: List[str]):
        """
        Creates filtered training dataset from input colvars files.
        
        Parameters
        ----------
        
        colvars_paths : str
            List of paths to colvars files with the main data used for training
        """
        
        logger.info('Reading training data from colvars files...')
        
        # Use same load args for all colvars files
        load_args = [self.training_reading_settings for _ in colvars_paths]
        
        # Main data
        self.training_input_dtset = create_dataset_from_files(
            file_names=colvars_paths,
            load_args=load_args,       
            filter_args=self.feature_filter, 
            verbose=False, 
            return_dataframe=False
        )
        
        # Find number of features
        self.num_features = self.training_input_dtset["data"].shape[1]
        logger.info(f'Number of features: {self.num_features}')

    def read_reference_data(self, ref_colvars_paths: Union[List[str], None]):
        """
        Reads the reference data from the colvars files.
        """
        
        # Reference data (if provided)
        if ref_colvars_paths:
            for path in ref_colvars_paths:
                ref_dataset = create_dataset_from_files(
                    file_names=[path], 
                    filter_args=self.feature_filter, 
                    verbose=False, 
                    return_dataframe=False
                )
                
                # Check if the number of features is the same
                if ref_dataset["data"].shape[1] != self.num_features:
                    logger.error(f"""Number of features in colvars file {path} is {ref_dataset["data"].shape[1]} and does 
                                    not match the number of features in the training dataset ({self.num_features}). Exiting...""")
                    sys.exit(1)
                
                # Append to lists
                self.ref_datasets.append(ref_dataset) 
              
    def read_colvars_data(self, colvars_path: str) -> DictDataset:
        """
        Reads the colvars data from the colvars file.
        """
        
        # Read the colvars file
        colvars_dataset = create_dataset_from_files(
            file_names=[colvars_path], 
            filter_args=self.feature_filter, 
            verbose=False, 
            return_dataframe=False
        )
        
        # Check if the number of features is the same
        if colvars_dataset["data"].shape[1] != self.num_features:
            logger.error(f"""Number of features in colvars file {colvars_path} is {colvars_dataset["data"].shape[1]} and does 
                            not match the number of features in the training dataset ({self.num_features}). Exiting...""")
            sys.exit(1)
            
        return colvars_dataset
    
    def read_features(self, colvars_path: str) -> List[str]:
        """ 
        Read the list of feature names from the colvars file and filter the list based on the feature constraints.
        
        Parameters
        ----------
        
        colvars_path : str
            Path to the colvars file
        
        Returns
        -------
            features : List[str]
                List of feature names after filtering
        """
        from deep_cartograph.modules.plumed.colvars import read_column_names
        
        # Find all the features in the colvars file
        features = read_column_names(colvars_path)
        
        # Filter the features based on the constraints, if any
        if self.feature_filter:
            if self.feature_filter.keys() == 'items':
                features = [feat for feat in features if feat in self.feature_filter['items']]
            elif self.feature_filter.keys() == 'regex':
                features = [feat for feat in features if re.search(self.feature_filter['regex'], feat)]
        
        # Additional regex used by create_dataset_from_files()
        default_regex = "^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)"
        
        # Filter the features based on the default regex
        features = [feat for feat in features if re.search(default_regex, feat)]
        
        return features
    
    def get_feature_filter(self, feature_constraints: Union[List[str], str]) -> Dict:
        """
        Create the filter dictionary to select the features to use from the feature constraints.

        Parameters
        ----------

        feature_constraints: Union[List[str], str]
            List of features to use or regex to select the features
        
        Returns
        -------
        
        feature_filter : dict
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
    def run(self, cv_dimension: Union[int, None] = None):
        """
        Runs the CV calculator.
        Overwrites the dimension in the configuration if provided.
        """
        if cv_dimension:
            self.cv_dimension = cv_dimension
            
        self.compute_cv()
        
        # If the CV was computed successfully
        if self.cv is not None:
            
            self.set_labels()

            self.normalize_cv()
            
            self.project_reference()
            
            self.save_projected_ref()
            
            self.cv_specific_tasks()
            
            self.save_cv()
        
    def compute_cv(self):
        """
        Computes the collective variables. Implement in subclasses.
        """
        
        raise NotImplementedError

    def cv_specific_tasks(self):
        """
        Performs specific tasks for the CV. Implement in subclasses.
        """
            
        pass

    def save_cv(self):
        """
        Saves the collective variable weights to a file. Implement in subclasses.
        """
        
        raise NotImplementedError
        
    def project_colvars(self, colvars_path: str):
        """
        Projects the given colvars onto the CV space. Implement in subclasses.
        """
        
        raise NotImplementedError

    def project_reference(self):
        """
        Projects the reference data onto the CV space. Implement in subclasses.
        """
        
        raise NotImplementedError
    
    def save_projected_ref(self):
        """
        Saves the projected reference data to files, if there is any.
        """
        
        projected_ref_folder = os.path.join(self.output_path, 'reference_data')
        
        if self.projected_ref:
            
            if not os.path.exists(projected_ref_folder):
                os.makedirs(projected_ref_folder)
        
            for i, ref in enumerate(self.projected_ref):
                ref = pd.DataFrame(ref, columns=self.get_labels())
                ref.to_csv(os.path.join(projected_ref_folder,f'{self.ref_names[i]}.csv'), index=False, float_format='%.4f')
    
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
        """
        
        raise NotImplementedError
        
    def write_plumed_input(self):
        """ 
        Create a plumed input file that computes the collective variable from the features. Implement in subclasses.
        """
        
        raise NotImplementedError

    # Getters
    def get_projected_ref(self) -> List[np.ndarray]:
        """
        Returns the projected reference features.
        """
        
        return self.projected_ref
    
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
class LinearCVCalculator(CVCalculator):
    """
    Linear collective variables calculator (e.g. PCA)
    """
    
    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """ 
        Initializes a linear CV calculator.
        """
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
                
        # Main attributes
        self.cv: Union[torch.tensor, None] = None
        
        # Compute training data statistics
        self.feats_statistics: Dict = Statistics(self.training_input_dtset[:]['data']).to_dict()
        
        # Set the features normalization object
        if self.feats_norm_mode == 'none':
            stats_length = len(self.feats_statistics["mean"])
            self.feats_normalization = Normalization(self.num_features, mean=torch.zeros(stats_length), range=torch.ones(stats_length))
        else:
            self.feats_normalization = Normalization(self.num_features, mode=self.feats_norm_mode, stats=self.feats_statistics)

        # Normalize the training data
        self.normalized_training_data: torch.Tensor = self.feats_normalization(self.training_input_dtset[:]['data'])
    
    # Main methods
    def save_cv(self):
        """
        Saves the collective variable linear weights to a text file.
        """
        
        weights_path = os.path.join(self.output_path, f'weights.txt')
        np.savetxt(weights_path, self.cv.numpy())
        
        if self.feats_norm_mode == 'mean_std':
            np.savetxt(os.path.join(self.output_path, 'features_mean.txt'), self.feats_statistics['mean'])
            np.savetxt(os.path.join(self.output_path, 'features_std.txt'), self.feats_statistics['std'])
        elif self.feats_norm_mode == 'min_max':
            np.savetxt(os.path.join(self.output_path, 'features_max.txt'), self.feats_statistics['max'])
            np.savetxt(os.path.join(self.output_path, 'features_min.txt'), self.feats_statistics['min'])
        
        logger.info(f'Collective variable weights saved to {weights_path}')

    def project_reference(self):
        """
        Projects the reference data onto the CV space.
        """
        
        logger.info(f'Projecting reference data onto {cv_names_map[self.cv_name]} ...')
        
        # If reference data is provided
        if self.ref_datasets:
            
            for ref_dtset in self.ref_datasets:
                
                # Get the torch tensor
                ref_tensor = ref_dtset[:]['data']
                
                # Normalize the reference data
                ref_tensor = self.feats_normalization(ref_tensor)
                
                # Project the reference data onto the CV space
                projected_ref = ref_tensor @ self.cv
                
                self.projected_ref.append(self.cv_normalization(projected_ref).numpy())  
  
    def project_colvars(self, colvars_path: str) -> Union[np.ndarray, None]:
        """
        Projects the samples from the colvars file onto the CV space.
        
        Parameters
        ----------
        
        colvars_path : str  
            Path to the colvars file
            
        Returns
        -------
        
        projected_colvars : np.ndarray
            Projected features onto the CV space
        """
        
        if self.cv is None:
            logger.error('No collective variable to project.')
            return None
        
        logger.info(f'Projecting {Path(colvars_path).stem} features onto {cv_names_map[self.cv_name]} ...')
        
        colvars_dataset= self.read_colvars_data(colvars_path)
        
        # Get the torch tensor
        colvars_tensor = colvars_dataset[:]['data']
        
        # Normalize the features
        colvars_tensor = self.feats_normalization(colvars_tensor)
        
        # Project the features onto the CV space and return the resulting array
        projected_colvars = colvars_tensor @ self.cv
        
        return self.cv_normalization(projected_colvars).numpy()
            
    def normalize_cv(self):
        
        # Project the normalized training data onto the CV space
        projected_training_data = self.normalized_training_data @ self.cv
        
        # Compute statistics of the projected training data
        stats = Statistics(projected_training_data)
        
        # Find the normalization for the CV
        self.cv_normalization =  Normalization(self.cv_dimension, mode='min_max', stats = stats )
        
        # Get the dictionary with the statistics
        stats_dict = stats.to_dict()
        
        # Save the max/min values of each dimension - part of the final cv definition
        np.savetxt(os.path.join(self.output_path, 'cv_max.txt'), stats_dict['max'])
        np.savetxt(os.path.join(self.output_path, 'cv_min.txt'), stats_dict['min'])
        
# Subclass for non-linear collective variables calculators
class NonLinearCVCalculator(CVCalculator):
    """
    Non-linear collective variables calculator (e.g. Autoencoder)
    """
    
    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """ 
        Initializes a non-linear CV calculator.
        """
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
        
        # Main attributes
        self.cv: Union[AutoEncoderCV, DeepTICA, None] = None
        self.checkpoint: Union[ModelCheckpoint, None] = None
        self.metrics: Union[MetricsCallback, None] = None
        
        # Training configuration
        self.training_config: Dict = configuration['training'] 
        self.general_config: Dict = self.training_config['general']
        self.early_stopping_config: Dict  = self.training_config['early_stopping']
        self.optimizer_config: Dict = self.training_config['optimizer']
        self.lr_scheduler_config: Dict = self.training_config['lr_scheduler']
        
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
        
        self.num_training_samples: Union[int, None] = None
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
        
        # Learning rate scheduler
        if self.lr_scheduler_config is not None:

            lr_scheduler = {'scheduler': getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])}
            lr_scheduler.update(self.lr_scheduler_config['kwargs'])

            # Update options
            self.cv_options.update({"lr_scheduler": lr_scheduler, 
                                    "lr_interval": "epoch", 
                                    "lr_monitor": "valid_loss", 
                                    "lr_frequency": self.check_val_every_n_epoch})

            # Make early stopping patience larger than learning rate scheduler patience
            self.patience = max(self.patience, self.lr_scheduler_config['kwargs']['patience']*2)
    
    def check_batch_size(self):
        
       # Get the number of samples in the training set
        self.num_training_samples = int(self.num_samples*self.training_validation_lengths[0])
        
        # Check the batch size is not larger than the number of samples in the training set
        if self.batch_size >= self.num_training_samples:
            self.batch_size = closest_power_of_two(self.num_samples*self.training_validation_lengths[0])
            logger.warning(f"""The batch size is larger than the number of samples in the training set. 
                           Setting the batch size to the closest power of two: {self.batch_size}""")
            
    def train(self):
        
        logger.info(f'Training {cv_names_map[self.cv_name]} ...')
        
        # Train until model finds a good solution
        while not self.converged and self.tries < self.max_tries:
            try: 

                self.tries += 1

                # Debug
                logger.debug(f'Splitting the dataset...')

                # Build datamodule, split the dataset into training and validation
                datamodule = DictModule(
                    random_split = self.random_split,
                    dataset = self.training_input_dtset,
                    lengths = self.training_validation_lengths,
                    batch_size = self.batch_size,
                    shuffle = self.shuffle, 
                    generator = torch.manual_seed(self.seed + self.tries))
        
                # Debug
                logger.debug(f'Initializing {cv_names_map[self.cv_name]} object...')
                
                # Define non-linear model
                model = nonlinear_cv_map[self.cv_name](self.nn_layers, options=self.cv_options)

                # Set optimizer name
                model._optimizer_name = self.opt_name
                
                logger.info(f"Model architecture: {model}")
                
                # Debug
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
                
                # Debug
                logger.debug(f'Initializing Trainer...')

                # Define trainer
                trainer = lightning.Trainer(          
                    callbacks=[self.metrics, early_stopping, self.checkpoint],
                    max_epochs=self.max_epochs, 
                    logger=False, 
                    enable_checkpointing=True,
                    enable_progress_bar = False, 
                    check_val_every_n_epoch=self.check_val_every_n_epoch)

                # Debug
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
                self.cv = nonlinear_cv_map[self.cv_name].load_from_checkpoint(self.checkpoint.best_model_path)
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
        
        try:        
            # Save the loss if requested
            if self.training_config['save_loss']:
                np.save(os.path.join(self.output_path, 'train_loss.npy'), np.array(self.metrics.metrics['train_loss']))
                np.save(os.path.join(self.output_path, 'valid_loss.npy'), np.array(self.metrics.metrics['valid_loss']))
                np.save(os.path.join(self.output_path, 'epochs.npy'), np.array(self.metrics.metrics['epoch']))
                np.savetxt(os.path.join(self.output_path, 'model_score.txt'), np.array([self.best_model_score]))
                
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
        
        if self.cv is None:
            logger.error('No collective variable to save.')
            return
    
        weights_path = os.path.join(self.output_path, f'weights.ptc')
        self.cv.to_torchscript(file_path = weights_path, method='trace') # NOTE: check if this also saves the normalization layer
        
        logger.info(f'Collective variable weights saved to {weights_path}')

    def project_reference(self):
        """
        Projects the reference data onto the CV space.
        """
        
        logger.info(f'Projecting reference data onto {cv_names_map[self.cv_name]} ...')

        # If reference data is provided, project it as well
        if self.ref_datasets:
            for dataset in self.ref_datasets:
                ref_tensor = dataset[:]['data']
                with torch.no_grad():
                    self.projected_ref.append(self.cv(ref_tensor).numpy())
                    
    def project_colvars(self, colvars_path: str) -> Union[np.ndarray, None]:
        """
        Projects the samples from the colvars file onto the CV space.
        
        Parameters
        ----------
        
        colvars_path : str
            Path to the colvars file
        
        Returns
        -------
        
        projected_colvars : np.ndarray
            Projected features onto the CV space
        """
        
        if self.cv is None:
            logger.error('No collective variable to project.')
            return None
        
        logger.info(f'Projecting features onto {cv_names_map[self.cv_name]} ...')
        
        colvars_dataset = self.read_colvars_data(colvars_path)
        
        colvars_tensor = colvars_dataset[:]['data']
        
        # Data projected onto normalized latent space - feature and latent space normalization included in the model
        with torch.no_grad():
            projected_colvars = self.cv(colvars_tensor).numpy()
   
        return projected_colvars
    
# Collective variables calculators
class PCACalculator(LinearCVCalculator):
    """
    Principal component analysis calculator.
    """

    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the PCA calculator.
        """
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
        
        self.cv_name = 'pca'
        
        self.initialize()
        
    def compute_cv(self):
        """
        Compute Principal Component Analysis (PCA) on the input features. 
        """
        
        # NOTE: This could still be useful for large datasets?
        # Choose between deterministic and non-deterministic PCA
        # if pca_lowrank_q < self.num_features:
             # Non-deterministic PCA using randomized SVD
             # out_features is q in torch.pca_lowrank -> Controls the dimensionality of the random projection in the randomized SVD algorithm (trade-off between speed and accuracy)
        #    pca_cv = PCA(in_features = self.num_features, out_features=min(pca_lowrank_q, self.num_features, self.num_samples))
        #    try:
        #        pca_eigvals, pca_eigvecs = pca_cv.compute(X=torch.tensor(self.normalized_training_data.numpy()), center = True)
        #    except Exception as e:
        #        logger.error(f'PCA could not be computed. Error message: {e}')
        #        return
        #    # Extract the first cv_dimension eigenvectors as CVs 
        #    self.cv = pca_eigvecs[:,0:self.cv_dimension].numpy()
        
        # Create PCA object
        pca = PCA(n_components=self.cv_dimension)
        
        # Fit the PCA model
        pca.fit(self.normalized_training_data.numpy())
        
        # Save the eigenvectors as CVs
        self.cv = torch.tensor(pca.components_.T)
        
        # Follow a criteria for the sign of the eigenvectors - first weight of each eigenvector should be positive
        for i in range(self.cv_dimension):
            if self.cv[0,i] < 0:
                self.cv[:,i] = -self.cv[:,i]
                
class TICACalculator(LinearCVCalculator):
    """ 
    Time-lagged independent component analysis calculator.
    """
    
    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the TICA calculator.
        """
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
        
        self.cv_name = 'tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag)
        self.training_input_dtset = create_timelagged_dataset(self.normalized_training_data.numpy(), lag_time=self.architecture_config['lag_time'])
        
        self.initialize()
        
    def compute_cv(self):
        """
        Compute Time-lagged Independent Component Analysis (TICA) on the input features. 
        """

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
        self.cv = tica_eigvecs
  
class HTICACalculator(LinearCVCalculator):
    """ 
    Hierarchical Time-lagged independent component analysis calculator.
    
    See: 
    
    Pérez-Hernández, Guillermo, and Frank Noé. “Hierarchical Time-Lagged Independent Component Analysis: 
    Computing Slow Modes and Reaction Coordinates for Large Molecular Systems.” Journal of Chemical Theory 
    and Computation 12, no. 12 (December 13, 2016): 6118–29. https://doi.org/10.1021/acs.jctc.6b00738.
    """
    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the HTICA calculator.
        """
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
        
        self.cv_name = 'htica'
        
        self.num_subspaces = configuration['num_subspaces']
        self.subspaces_dimension = configuration['subspaces_dimension']
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag)
        self.training_input_dtset = create_timelagged_dataset(self.normalized_training_data.numpy(), lag_time=self.architecture_config['lag_time'])
        
        self.initialize()
        
    
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
        self.cv = torch.tensor(Transform_level_1_TICA @ level_2_eigvecs)
           
class AECalculator(NonLinearCVCalculator):
    """
    Autoencoder calculator.
    """
    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the Autoencoder calculator.
        """
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
        
        self.cv_name = 'ae'
        
        self.initialize()
        
        self.check_batch_size()
        
        # Update options
        self.cv_options.update({"encoder": self.nn_options,
                                "decoder": self.nn_options,
                                "optimizer": self.optimizer_options})
        
class DeepTICACalculator(NonLinearCVCalculator):
    """
    DeepTICA calculator.
    """
    def __init__(self, colvars_path: str, topology_paths: List[str], feature_constraints: Union[List[str], str], 
                 ref_colvars_paths: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the DeepTICA calculator.
        """      
        
        super().__init__(colvars_path, topology_paths, feature_constraints, ref_colvars_paths, configuration, output_path)
        
        self.cv_name = 'deep_tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag)
        self.training_input_dtset = create_timelagged_dataset(self.training_input_dtset[:]['data'].numpy(), lag_time=self.architecture_config['lag_time'])
        
        self.initialize()
        
        self.check_batch_size()
            
        # Update options
        self.cv_options.update({"nn": self.nn_options,
                                "optimizer": self.optimizer_options})
        
    def cv_specific_tasks(self):
        """
        Save the eigenvectors and eigenvalues of the best model.
        """
            
        # Find the epoch where the best model was found
        best_index = self.metrics.metrics['valid_loss'].index(self.best_model_score)
        best_epoch = self.metrics.metrics['epoch'][best_index]
        logger.info(f'Took {best_epoch} epochs')

        # Find eigenvalues of the best model
        best_eigvals = [self.metrics.metrics[f'valid_eigval_{i+1}'][best_index] for i in range(self.cv_dimension)]
        for i in range(self.cv_dimension):
            logger.info(f'Eigenvalue {i+1}: {best_eigvals[i]}')
            
        np.savetxt(os.path.join(self.output_path, 'eigenvalues.txt'), np.array(best_eigvals))
        
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

nonlinear_cv_map = {
    'ae': AutoEncoderCV,
    'deep_tica': DeepTICA
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