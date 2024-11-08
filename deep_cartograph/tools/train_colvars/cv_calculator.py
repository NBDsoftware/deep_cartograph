import os
import sys
import torch
import lightning
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union

from mlcolvar.utils.timelagged import create_timelagged_dataset # NOTE: this function returns less samples than expected: N-lag_time-2
from mlcolvar.utils.io import create_dataset_from_files
from mlcolvar.data import DictModule, DictDataset
from mlcolvar.cvs import AutoEncoderCV, DeepTICA
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.core.stats import TICA, PCA
from mlcolvar.core.transform import Normalization
from mlcolvar.core.transform.utils import Statistics


from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from deep_cartograph.modules.common import (get_filter_dict, closest_power_of_two, create_output_folder)

# Set logger
logger = logging.getLogger(__name__)

# Base class for collective variables calculators
class CVCalculator:
    """
    Base class for collective variables calculators.
    """
    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the base CV calculator.
        
        Parameters
        ----------
        
        colvars_path : str
            Path to the colvars file
        feature_constraints : Union[List[str], str]
            List with the features to use for the training or str with regex to filter feature names.
        ref_colvars_path : Union[List[str], None]
            List of paths to colvars files with reference data
        configuration : Dict
            Configuration dictionary for the CV
        output_path : str
            Output path where the CV results folder will be created
        """
        
        # Input data
        self.training_input_dtset: DictDataset = None     # Used to train / compute the CVs, contains just the samples defined in input_colvars_settings
        self.projection_input_df: pd.DataFrame = None  # Used to project the feature samples onto the CV space, contains all samples
        
        self.ref_dtsets: List[DictDataset] = []
        self.ref_dfs: List[pd.DataFrame] = []
        
        self.num_features: int = None
        self.num_samples: int = None
        
        # Configuration
        self.configuration: Dict = configuration
        self.architecture_config: Dict = configuration['architecture']
        self.input_colvars_settings: Dict = configuration['input_colvars']
        
        self.read_data(colvars_path, feature_constraints, ref_colvars_path)
        
        # General CV attributes
        self.cv_dimension: int = configuration['dimension']
        self.cv_labels: List[str] = []
        self.cv_name: str = None
        
        # Output 
        self.projected_input: np.ndarray = None
        self.projected_ref: List[np.ndarray] = []
    
        self.output_path: str =  output_path

    def read_data(self, colvars_path: str, feature_constraints: Union[List[str], str],
                      ref_colvars_path: Union[List[str], None]):
        """
        Creates datasets and dataframes from input colvars and filters them.
        
        Parameters
        ----------
        
        colvars_path : str
            Path to the colvars file
        feature_constraints : Union[List[str], str]
            List with the features to use for the training or str with regex to filter feature names.
        ref_colvars_path : Union[List[str], None]
            List of paths to colvars files with reference data
        """
        
        logger.info('Reading data from colvars files...')
        
        filter_dict = get_filter_dict(feature_constraints)
            
        # Main data
        self.training_input_dtset, self.projection_input_df = create_dataset_from_files(
            file_names=[colvars_path],
            load_args=[self.input_colvars_settings],       # NOTE: only training should use input_colvars_settings, otherwise remove it
            filter_args=filter_dict, 
            verbose=False, 
            return_dataframe=True
        )
        self.projection_input_df = self.projection_input_df.filter(**filter_dict)
        
        # Number of features
        self.num_features = self.projection_input_df.shape[1]
        logger.info(f'Number of features: {self.num_features}')

        # Reference data (if provided)
        if ref_colvars_path:
            for path in ref_colvars_path:
                ref_dataset, ref_dataframe = create_dataset_from_files(
                    file_names=[path], 
                    filter_args=filter_dict, 
                    verbose=False, 
                    return_dataframe=True
                )
                
                # Check if the number of features is the same
                if ref_dataframe.shape[1] != self.num_features:
                    logger.error(f"""Number of features in reference dataset {path} is {ref_dataframe.shape[1]} and does 
                                 not match the number of features in the main dataset ({self.num_features}). Exiting...""")
                    sys.exit(1)
                
                # Append to lists # NOTE: we need just one of the two?
                self.ref_dtsets.append(ref_dataset)
                self.ref_dfs.append(ref_dataframe.filter(**filter_dict)) 
                
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
        
    def project_features(self):
        """
        Projects the features onto the CV space. Implement in subclasses.
        """
        
        raise NotImplementedError
          
    def run(self, cv_dimension: Union[int, None] = None):
        """
        Runs the CV calculator.
        Overwrites the dimension in the configuration if provided.
        """
        if cv_dimension:
            self.cv_dimension = cv_dimension
            
        self.compute_cv()
        
        self.cv_specific_tasks()
        
        self.save_cv()
        
        self.project_features()
        
        self.set_labels()
    
    def set_labels(self):
        """
        Sets the labels of the features.
        """
        
        self.cv_labels = [f'{cv_components_map[self.cv_name]} {i+1}' for i in range(self.cv_dimension)]
    
    def get_projected_input(self) -> np.ndarray:
        """
        Returns the projected input features.
        """
        
        return self.projected_input
    
    def get_projected_ref(self) -> List[np.ndarray]:
        """
        Returns the projected reference features.
        """
        
        return self.projected_ref
    
    def get_labels(self) -> List[str]:
        """
        Returns the labels of the features.
        """
        
        return self.cv_labels
    
    def get_cv_dimension(self) -> int:
        """
        Returns the dimension of the collective variables.
        """
        
        return self.cv_dimension
    
# Subclass for linear collective variables calculators
class LinearCVCalculator(CVCalculator):
    """
    Linear collective variables calculator (e.g. PCA)
    """
    
    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """ 
        Initializes a linear CV calculator.
        """
        
        super().__init__(colvars_path, feature_constraints, ref_colvars_path, configuration, output_path)
                
        # Main attributes
        self.cv: Union[np.array, None] = None
        
    def save_cv(self):
        """
        Saves the collective variable linear weights to a text file.
        """
        
        if self.cv is None:
            logger.error('No collective variable to save.')
            return
        
        cv_path = os.path.join(self.output_path, f'weights.txt')
        np.savetxt(cv_path, self.cv)
        
        logger.info(f'Collective variable saved to {cv_path}')
        
    def project_features(self):
        """
        Projects the features onto a linear CV space.
        """
        
        logger.info(f'Projecting features onto {cv_names_map[self.cv_name]} ...')
        
        # Find a numpy array of features
        features_array = self.projection_input_df.to_numpy(dtype=np.float32)
        
        # Project the input dataframe onto the CV space
        self.projected_input = np.matmul(features_array, self.cv)
        
        # Project the reference dataframe onto the CV space
        if self.ref_dfs:
            self.projected_ref = [np.matmul(df.to_numpy(dtype=np.float32), self.cv) for df in self.ref_dfs]
       
# Subclass for non-linear collective variables calculators
class NonLinearCVCalculator(CVCalculator):
    """
    Non-linear collective variables calculator (e.g. Autoencoder)
    """
    
    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """ 
        Initializes a non-linear CV calculator.
        """
        
        super().__init__(colvars_path, feature_constraints, ref_colvars_path, configuration, output_path)
        
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
        self.cv_options: Dict = {}
        
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
            patience = max(patience, self.lr_scheduler_config['kwargs']['patience']*2)
    
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
                    generator = torch.manual_seed(self.seed))
        
                # Debug
                logger.debug(f'Initializing {cv_names_map[self.cv_name]} object...')
                
                # Define non-linear model
                model = cv_classes_map[self.cv_name](self.nn_layers, options=self.cv_options)

                # Set optimizer name
                model._optimizer_name = self.opt_name

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
                self.cv = cv_classes_map[self.cv_name].load_from_checkpoint(self.checkpoint.best_model_path)
                os.remove(self.checkpoint.best_model_path)
                
                # Find the score of the best model
                self.best_model_score = self.checkpoint.best_model_score
                logger.info(f'Best model score: {self.best_model_score}')
            else:
                logger.error('The best model checkpoint does not exist.')
        
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

    def compute_cv(self):
        """
        Compute Non-linear CV.
        """

        # Train the non-linear model
        self.train()  
        
        # Save the loss 
        self.save_loss()
        
    def save_cv(self):
        """
        Saves the collective variable non-linear weights to a pytorch script file.
        """
        
        if self.cv is None:
            logger.error('No collective variable to save.')
            return
    
        cv_path = os.path.join(self.output_path, f'weights.ptc')
        self.cv.to_torchscript(file_path = cv_path, method='trace')
        
        logger.info(f'Collective variable saved to {cv_path}')

    def project_features(self):
        """
        Projects the features onto a non-linear CV space.
        """
        
        logger.info(f'Projecting features onto {cv_names_map[self.cv_name]} ...')

        # Put model in evaluation mode
        self.cv.eval()
        
        # Data projected onto original latent space of the best model
        with torch.no_grad():
            self.cv.postprocessing = None
            projected_input = self.cv(torch.Tensor(self.projection_input_df.values))

        # Normalize the latent space
        norm =  Normalization(self.cv_dimension, mode='min_max', stats = Statistics(projected_input) )
        self.cv.postprocessing = norm
        
        # Data projected onto normalized latent space
        with torch.no_grad():
            self.projected_input = self.cv(torch.Tensor(self.projection_input_df.values)).numpy()

        # If reference data is provided, project it as well
        if self.ref_dfs:
            for df in self.ref_dfs:
                ref_features_array = df.to_numpy(dtype=np.float32)
                with torch.no_grad():
                    self.projected_ref.append(self.cv(torch.Tensor(ref_features_array)).numpy())
   
        
# Collective variables calculators
class PCACalculator(LinearCVCalculator):
    """
    Principal component analysis calculator.
    """

    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the PCA calculator.
        """
        
        super().__init__(colvars_path, feature_constraints, ref_colvars_path, configuration, output_path)
        
        self.cv_name = 'pca'
        
        self.initialize()
        
    def compute_cv(self):
        """
        Compute Principal Component Analysis (PCA) on the input features. 
        """
        
        # Find the user requested q for torch.pca_lowrank
        pca_lowrank_q = self.architecture_config['pca_lowrank_q']
        if pca_lowrank_q is None:
            pca_lowrank_q = self.num_features
        
        # Use PCA to compute high variance linear combinations of the input features
        # out_features is q in torch.pca_lowrank -> Controls the dimensionality of the random projection in the randomized SVD algorithm (trade-off between speed and accuracy)
        pca_cv = PCA(in_features = self.num_features, out_features=min(pca_lowrank_q, self.num_features, self.num_samples))
        
        # Compute PCA
        try:
            pca_eigvals, pca_eigvecs = pca_cv.compute(X=torch.tensor(self.training_input_dtset[:]['data'].numpy()), center = True)
        except Exception as e:
            logger.error(f'PCA could not be computed. Error message: {e}')
            return
        
        # Extract the first cv_dimension eigenvectors as CVs 
        self.cv = pca_eigvecs[:,0:self.cv_dimension].numpy()
        
        # Follow a criteria for the sign of the eigenvectors - first weight of each eigenvector should be positive
        for i in range(self.cv_dimension):
            if self.cv[0,i] < 0:
                self.cv[:,i] = -self.cv[:,i]
                
class TICACalculator(LinearCVCalculator):
    """ 
    Time-lagged independent component analysis calculator.
    """
    
    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the TICA calculator.
        """
        
        super().__init__(colvars_path, feature_constraints, ref_colvars_path, configuration, output_path)
        
        self.cv_name = 'tica'
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag)
        self.training_input_dtset = create_timelagged_dataset(self.training_input_dtset[:]['data'].numpy(), lag_time=self.architecture_config['lag_time'])
        
        self.initialize()
        
    def compute_cv(self):
        """
        Compute Time-lagged Independent Component Analysis (TICA) on the input features. 
        """

        # Use TICA to compute slow linear combinations of the input features
        # Here out_features is the number of eigenvectors to keep
        tica_cv = TICA(in_features = self.num_features, out_features=self.cv_dimension)

        try:
            # Compute TICA
            tica_eigvals, tica_eigvecs = tica_cv.compute(data=[self.training_input_dtset['data'], self.training_input_dtset['data_lag']], save_params = True, remove_average = True)
        except Exception as e:
            logger.error(f'TICA could not be computed. Error message: {e}')
            return

        # Save the first cv_dimension eigenvectors as CVs
        self.cv = tica_eigvecs.numpy()
        
class AECalculator(NonLinearCVCalculator):
    """
    Autoencoder calculator.
    """
    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the Autoencoder calculator.
        """
        
        super().__init__(colvars_path, feature_constraints, ref_colvars_path, configuration, output_path)
        
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
    def __init__(self, colvars_path: str, feature_constraints: Union[List[str], str], 
                 ref_colvars_path: Union[List[str], None], configuration: Dict, output_path: str):
        """
        Initializes the DeepTICA calculator.
        """      
        
        super().__init__(colvars_path, feature_constraints, ref_colvars_path, configuration, output_path)
        
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
    'deep_tica': DeepTICACalculator
}

cv_classes_map = {
    'pca': PCA,
    'ae': AutoEncoderCV,
    'tica': TICA,
    'deep_tica': DeepTICA
}

cv_names_map = {
    'pca': 'PCA',
    'ae': 'AE',
    'tica': 'TICA',
    'deep_tica': 'DeepTICA'
}

cv_components_map = {
    'pca': 'PC',
    'ae': 'AE',
    'tica': 'TIC',
    'deep_tica': 'DeepTIC'
}