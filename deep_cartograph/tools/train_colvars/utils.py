"""
Utils for the train_colvars tool. 

These are functions that are used by the train_colvars tool that leverage or not
the modules in the deep_cartograph package.
"""

# General imports
import os
import logging
import numpy as np
import pandas as pd
import torch, lightning
from typing import List, Dict, Union

from mlcolvar.data import DictModule, DictDataset
from mlcolvar.cvs import AutoEncoderCV, DeepTICA
from mlcolvar.utils.timelagged import create_timelagged_dataset
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.core.stats import TICA, PCA
from mlcolvar.core.transform import Normalization
from mlcolvar.core.transform.utils import Statistics

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# Local imports
from deep_cartograph.modules.md import md
from deep_cartograph.modules.common import common
from deep_cartograph.modules.figures import figures
from deep_cartograph.modules.statistics import statistics

# Set logger
logger = logging.getLogger(__name__)


def compute_pca(features_dataframe: pd.DataFrame, ref_features_dataframe: Union[List[pd.DataFrame], None], ref_labels: Union[List[str], None], 
                cv_settings: Dict, figures_settings: Dict, clustering_settings: Dict, trajectory: str, topology: str, output_path: str):
    """
    Compute Principal Component Analysis (PCA) on the input features. 
    Compute the Free Energy Surface (FES) along the PCA CVs.
    Project the trajectory onto the PCA space and cluster the projected features if requested.

    Inputs
    ------

        features_dataframe:     DataFrame containing the time series of the input features for the main trajectory. Each column is a feature and each row is a time step.
        ref_features_dataframe: List of DataFrames containing the time series of the input features for the reference data. Each column is a feature and each row is a time step.
        ref_labels:             List of labels for the reference data.
        cv_settings:            Dictionary containing the settings for the CVs.
        figures_settings:       Dictionary containing the settings for figures.
        clustering_settings:    Dictionary containing the settings for clustering the projected features.
        trajectory:             Path to the trajectory file that will be clustered corresponding to the features dataframe.
        topology:               Path to the topology file of the trajectory.
        output_path:            Path to the output folder where the PCA results will be saved.
    """

    # Create output directory
    output_path = common.get_unique_path(output_path)
    common.create_output_folder(output_path)

    # Find cv dimension
    cv_dimension = cv_settings['dimension']

    logger.info('Calculating PCA...')

    # Get the number of features and samples
    num_features = features_dataframe.shape[1]
    num_samples = features_dataframe.shape[0]

    # Find the user requested q for torch.pca_lowrank
    pca_lowrank_q = cv_settings['architecture']['pca_lowrank_q']
    if pca_lowrank_q is None:
        pca_lowrank_q = num_features

    # Use PCA to compute high variance linear combinations of the input features
    # out_features is q in torch.pca_lowrank -> Controls the dimensionality of the random projection in the randomized SVD algorithm (trade-off between speed and accuracy)
    pca = PCA(in_features = num_features, out_features=min(pca_lowrank_q, num_features, num_samples))

    try:
        # Compute PCA
        pca_eigvals, pca_eigvecs = pca.compute(X=torch.tensor(features_dataframe.to_numpy()), center = True)
    except Exception as e:
        logger.error(f'PCA could not be computed. Error message: {e}')
        return
    
    # Extract the first cv_dimension eigenvectors as CVs 
    pca_cv = pca_eigvecs[:,0:cv_dimension].numpy()

    # Follow a criteria for the sign of the eigenvectors - first weight of each eigenvector should be positive
    for i in range(cv_dimension):
        if pca_cv[0,i] < 0:
            pca_cv[:,i] = -pca_cv[:,i]

    # Save the first cv_dimension eigenvectors as CVs
    np.savetxt(os.path.join(output_path,'weights.txt'), pca_cv)        
    
    # Transform to array
    features_array = features_dataframe.to_numpy(dtype=np.float32)

    # Project features onto the CV space
    projected_features = np.matmul(features_array, pca_cv)

    # If reference data is provided, project it as well
    if ref_features_dataframe is not None:
        
        projected_ref_features = []
        for df in ref_features_dataframe:
            ref_features_array = df.to_numpy(dtype=np.float32)
            projected_ref_features.append(np.matmul(ref_features_array, pca_cv))
    else:
        projected_ref_features = None

    # Create CV labels 
    cv_labels = [f'PC {i+1}' for i in range(cv_dimension)]

    try:
        # Create FES along the CV
        figures.plot_fes(
            X=projected_features, 
            cv_labels=cv_labels,
            X_ref=projected_ref_features,
            X_ref_labels=ref_labels,
            settings=figures_settings['fes'], 
            output_path=output_path)
    except Exception as e:
        logger.error(f'Failed to plot the FES. Error message: {e}')

    try:
        # Project the trajectory onto the CV space
        project_traj(projected_features, cv_labels, figures_settings, clustering_settings, trajectory, topology, output_path)
    except Exception as e:
        logger.error(f'Failed to project the trajectory. Error message: {e}')

def compute_ae(features_dataset: DictDataset, ref_features_dataset: Union[List[DictDataset], None], ref_labels: Union[List[str], None], 
               cv_settings: Dict, figures_settings: Dict, clustering_settings: Dict, trajectory: str, topology: str, output_path: str):
    """
    Train Autoencoder on the input features. The CV is the latent space of the Autoencoder. 
    Compute the Free Energy Surface (FES) along the Autoencoder CVs.
    Project the trajectory onto the Autoencoder space and cluster the projected features if requested.

    Inputs
    ------

        features_dataset:      Dataset containing the input features for the main trajectory.
        ref_features_dataset:  List of Datasets containing the reference data.
        ref_labels:            List of labels for the reference data.
        cv_settings:           Dictionary containing the settings for the CVs.
        figures_settings:      Dictionary containing the settings for the figures.
        clustering_settings:   Dictionary containing the settings for clustering the projected features.
        trajectory:             Path to the trajectory file that will be clustered corresponding to the features dataframe.
        topology:               Path to the topology file of the trajectory.
        output_path:           Path to the output folder where the Autoencoder results will be saved.
    """

    # Create output directory
    output_path = common.get_unique_path(output_path)
    common.create_output_folder(output_path)

    # Find the dimension of the CV
    cv_dimension = cv_settings['dimension']

    # Training settings
    training_settings = cv_settings['training']

    general_settings = training_settings['general']
    early_stopping_settings = training_settings['early_stopping']
    optimizer_settings = training_settings['optimizer']
    lr_scheduler_settings = training_settings['lr_scheduler']

    max_tries = general_settings['max_tries']
    seed = general_settings['seed']
    training_validation_lengths = general_settings['lengths']
    batch_size = general_settings['batch_size']
    shuffle = general_settings['shuffle']
    random_split = general_settings['random_split']
    max_epochs = general_settings['max_epochs']
    dropout = general_settings['dropout']
    check_val_every_n_epoch = general_settings['check_val_every_n_epoch']
    save_check_every_n_epoch = general_settings['save_check_every_n_epoch']

    patience = early_stopping_settings['patience']
    min_delta = early_stopping_settings['min_delta']

    # Architecture settings
    architecture_settings = cv_settings['architecture']
    
    hidden_layers = architecture_settings['hidden_layers']

    # Get the number of features and samples
    num_features = features_dataset["data"].shape[1]
    num_samples = features_dataset["data"].shape[0]
    num_training_samples =int(num_samples*training_validation_lengths[0])

    # Check the batch size is not larger than the number of samples in the training set
    if batch_size >= num_training_samples:
        batch_size = common.closest_power_of_two(num_samples*training_validation_lengths[0])
        logger.warning(f'The batch size is larger than the number of samples in the training set. Setting the batch size to the closest power of two: {batch_size}')

    logger.info('Training Autoencoder CV...')

    # Autoencoder settings
    encoder_layers = [num_features] + hidden_layers + [cv_dimension]
    nn_options = {'activation': 'shifted_softplus', 'dropout': dropout} 

    # Optimizer
    opt_name = optimizer_settings['name']
    optimizer_options = optimizer_settings['kwargs']

    # Create options
    options = {'encoder': nn_options,
                'decoder': nn_options,
                "optimizer": optimizer_options}

    # Learning rate scheduler
    if lr_scheduler_settings is not None:

        lr_scheduler = {'scheduler': getattr(torch.optim.lr_scheduler, lr_scheduler_settings['name'])}
        lr_scheduler.update(lr_scheduler_settings['kwargs'])

        # Update options
        options.update({"lr_scheduler": lr_scheduler, 
                        "lr_interval": "epoch", 
                        "lr_monitor": "valid_loss", 
                        "lr_frequency": check_val_every_n_epoch})

        # Make early stopping patience larger than learning rate scheduler patience
        patience = max(patience, lr_scheduler_settings['kwargs']['patience']*2)

    converged = False
    tries = 0

    # Train until model finds a good solution
    while not converged and tries < max_tries:
        try:
        
            tries += 1

            # Debug
            logger.debug(f'Splitting the dataset...')

            # Build datamodule, split the dataset into training and validation
            datamodule = DictModule(
                random_split = random_split,
                dataset = features_dataset,
                lengths = training_validation_lengths,
                batch_size = batch_size,
                shuffle = shuffle, 
                generator = torch.manual_seed(seed))
    
            # Debug
            logger.debug(f'Initializing Autoencoder object...')

            # Define model
            model = AutoEncoderCV(encoder_layers, options=options) 

            # Set optimizer name
            model._optimizer_name = opt_name

            # Debug
            logger.debug(f'Initializing metrics and callbacks...')

            # Define MetricsCallback to store the loss
            metrics = MetricsCallback()

            # Define EarlyStopping callback to stop training if the loss does not decrease
            early_stopping = EarlyStopping(
                monitor="valid_loss", 
                min_delta=min_delta,
                patience=patience,
                mode="min")

            # Define ModelCheckpoint callback to save the best model
            checkpoint = ModelCheckpoint(
                dirpath=output_path,
                monitor="valid_loss",                      # Quantity to monitor
                save_last=False,                           # Save the last checkpoint
                save_top_k=1,                              # Number of best models to save according to the quantity monitored
                save_weights_only=True,                    # Save only the weights
                filename=None,                             # Default checkpoint file name '{epoch}-{step}'
                mode="min",                                # Best model is the one with the minimum monitored quantity
                every_n_epochs=save_check_every_n_epoch)   # Number of epochs between checkpoints

            # Debug
            logger.debug(f'Initializing Trainer...')

            # Define trainer
            trainer = lightning.Trainer(                           # accelerator="cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
                callbacks=[metrics, early_stopping, checkpoint], 
                max_epochs=max_epochs,
                logger=False, 
                enable_checkpointing = True,
                enable_progress_bar = False,
                check_val_every_n_epoch=check_val_every_n_epoch)

            # Debug
            logger.debug(f'Training...')

            trainer.fit(model, datamodule)

            # Get validation and training loss
            validation_loss = metrics.metrics['valid_loss']

            # Check the evolution of the loss
            converged = model_has_converged(validation_loss, patience, check_val_every_n_epoch)
            if not converged:
                logger.warning('Autoencoder has not found a good solution. Re-starting training...')
        
        except Exception as e:
            logger.error(f'Autoencoder training failed. Error message: {e}')
            logger.info('Retrying Autoencoder training...')

    try:
        # Load best model from checkpoint
        best_model = AutoEncoderCV.load_from_checkpoint(checkpoint.best_model_path)
        best_model_score = checkpoint.best_model_score

        # Log score
        logger.info(f'Best model score: {best_model_score}')

        # Save the loss if requested
        if training_settings.get('save_loss', False):
            np.save(os.path.join(output_path, 'train_loss.npy'), np.array(metrics.metrics['train_loss']))
            np.save(os.path.join(output_path, 'valid_loss.npy'), np.array(metrics.metrics['valid_loss']))
            np.save(os.path.join(output_path, 'epochs.npy'), np.array(metrics.metrics['epoch']))
            np.savetxt(os.path.join(output_path, 'model_score.txt'), np.array([best_model_score]))

        # Put model in evaluation mode
        best_model.eval()

        # Plot loss
        ax = plot_metrics(metrics.metrics, 
                            labels=['Training', 'Validation'], 
                            keys=['train_loss', 'valid_loss'], 
                            linestyles=['-','-'], colors=['fessa1','fessa5'], 
                            yscale='log')

        # Save figure
        ax.figure.savefig(os.path.join(output_path, f'loss.png'), dpi=300, bbox_inches='tight')
        ax.figure.clf()

    except Exception as e:
        logger.error(f'Failed to save/plot the loss. Error message: {e}')

    if converged:
        try:
            # Data projected onto original latent space of the best model
            with torch.no_grad():
                best_model.postprocessing = None
                projected_features = best_model(torch.Tensor(features_dataset[:]["data"]))

            # Normalize the latent space
            norm =  Normalization(cv_dimension, mode='min_max', stats = Statistics(projected_features) ) 
            best_model.postprocessing = norm

            # Data projected onto normalized latent space
            with torch.no_grad():
                projected_features = best_model(torch.Tensor(features_dataset[:]["data"])).numpy()
            
            # If reference data is provided, project it as well
            if ref_features_dataset is not None:
                projected_ref_features = []
                for dataset in ref_features_dataset:
                    with torch.no_grad():
                        projected_ref_features.append(best_model(torch.Tensor(dataset[:]["data"])).numpy())
            else:
                projected_ref_features = None

            # Save the best model with the latent space normalized
            best_model.to_torchscript(file_path = os.path.join(output_path, 'weights.ptc'), method='trace')

            # Delete checkpoint file
            if os.path.exists(checkpoint.best_model_path):
                os.remove(checkpoint.best_model_path)

            # Create CV labels
            cv_labels = [f'AE {i+1}' for i in range(cv_dimension)]

            # Create FES along the CV
            figures.plot_fes(
                X=projected_features, 
                cv_labels=cv_labels,
                X_ref=projected_ref_features,
                X_ref_labels=ref_labels,
                settings=figures_settings['fes'], 
                output_path=output_path)

            project_traj(projected_features, cv_labels, figures_settings, clustering_settings, trajectory, topology, output_path)

        except Exception as e:
            logger.error(f'Failed to project the trajectory. Error message: {e}')
    else:
        logger.warning('Autoencoder training did not find a good solution after maximum tries.')

def compute_tica(features_dataframe: pd.DataFrame, ref_features_dataframe: Union[List[pd.DataFrame], None], ref_labels: Union[List[str], None], 
                 cv_settings: Dict, figures_settings: Dict, clustering_settings: Dict, trajectory: str, topology: str, output_path: str):
    """
    Compute Time-lagged Independent Component Analysis (TICA) on the input features. Also, compute the Free Energy Surface (FES) along the TICA CVs.

    Inputs
    ------

        features_dataframe:     DataFrame containing the time series of the input features for the main trajectory. Each column is a feature and each row is a time step.
        ref_features_dataframe: List of DataFrames containing the reference data. Each column is a feature and each row is a time step. 
        ref_labels:             List of labels for the reference data.
        cv_settings:            Dictionary containing the settings for the CVs.
        figures_settings:       Dictionary containing the settings for figures.
        clustering_settings:    Dictionary containing the settings for clustering the projected features.
        trajectory:             Path to the trajectory file that will be clustered corresponding to the features dataframe.
        topology:               Path to the topology file of the trajectory.
        output_path:            Path to the output folder where the PCA results will be saved.
    """

    # Create output directory
    output_path = common.get_unique_path(output_path)
    common.create_output_folder(output_path)

    # Find cv dimension
    cv_dimension = cv_settings['dimension']

    # Lag time for TICA
    lag_time = cv_settings['architecture']['lag_time']

    logger.info('Calculating TICA CV...')

    # Build time-lagged dataset (composed by pairs of configs at time t, t+lag)
    timelagged_dataset = create_timelagged_dataset(features_dataframe, lag_time=lag_time)

    # Get the number of features and samples
    num_features = timelagged_dataset["data"].shape[1]
    num_samples = timelagged_dataset["data"].shape[0]
    
    # Transform to array
    features_array = features_dataframe.to_numpy(dtype=np.float32)

    # Use TICA to compute slow linear combinations of the input features
    # Here out_features is the number of eigenvectors to keep
    tica = TICA(in_features = num_features, out_features=cv_dimension)

    try:
        # Compute TICA
        tica_eigvals, tica_eigvecs = tica.compute(data=[timelagged_dataset['data'], timelagged_dataset['data_lag']], save_params = True, remove_average = True)
    except Exception as e:
        logger.error(f'TICA could not be computed. Error message: {e}')
        return

    # Save the first cv_dimension eigenvectors as CVs
    tica_cv = tica_eigvecs.numpy()
    np.savetxt(os.path.join(output_path,'weights.txt'), tica_cv)

    # Evaluate the CV on the colvars data
    projected_features = np.matmul(features_array, tica_cv)

    # If reference data is provided, project it as well
    if ref_features_dataframe is not None:
        projected_ref_features = []
        for df in ref_features_dataframe:
            ref_features_array = df.to_numpy(dtype=np.float32)
            projected_ref_features.append(np.matmul(ref_features_array, tica_cv))
    else:
        projected_ref_features = None

    # Create CV labels
    cv_labels = [f'TIC {i+1}' for i in range(cv_dimension)]

    try:
        # Create FES along the CV
        figures.plot_fes(
            X=projected_features, 
            cv_labels=cv_labels,
            X_ref=projected_ref_features,
            X_ref_labels=ref_labels,
            settings=figures_settings['fes'], 
            output_path=output_path)
    except Exception as e:
        logger.error(f'Failed to plot the FES. Error message: {e}')
    
    try:
        # Project the trajectory onto the CV space
        project_traj(projected_features, cv_labels, figures_settings, clustering_settings, trajectory, topology, output_path)
    except Exception as e:
        logger.error(f'Failed to project the trajectory. Error message: {e}')

def compute_deep_tica(features_dataframe: pd.DataFrame, ref_features_dataframe: Union[List[pd.DataFrame], None], ref_labels: Union[List[str], None], 
                      cv_settings: Dict, figures_settings: Dict, clustering_settings: Dict, trajectory: str, topology: str, output_path: str):
    """
    Train DeepTICA on the input features. The CV is the latent space of the DeepTICA model. Also, compute the Free Energy Surface (FES) along the DeepTICA CVs.

    Inputs
    ------

        features_dataframe:     Dataset containing the input features for the main trajectory.
        ref_features_dataframe: List of Datasets containing the reference data.
        ref_labels:             List of labels for the reference data.
        cv_settings:            Dictionary containing the settings for the CVs.
        figures_settings:       Dictionary containing the settings for the figures.
        training_settings:      Dictionary containing the settings for training the DeepTICA model.
        trajectory:             Path to the trajectory file that will be clustered corresponding to the features dataframe.
        topology:               Path to the topology file of the trajectory.
        output_path:            Path to the output folder where the DeepTICA results will be saved.
    """

    # Create output directory
    output_path = common.get_unique_path(output_path)
    common.create_output_folder(output_path)

    # Find the dimension of the CV
    cv_dimension = cv_settings['dimension']

    # Training settings
    training_settings = cv_settings['training']

    general_settings = training_settings['general']
    early_stopping_settings = training_settings['early_stopping']
    optimizer_settings = training_settings['optimizer']
    lr_scheduler_settings = training_settings['lr_scheduler']

    max_tries = general_settings['max_tries']
    seed = general_settings['seed']
    training_validation_lengths = general_settings['lengths']
    batch_size = general_settings['batch_size']
    shuffle = general_settings['shuffle']
    random_split = general_settings['random_split']
    max_epochs = general_settings['max_epochs']
    dropout = general_settings['dropout']
    check_val_every_n_epoch = general_settings['check_val_every_n_epoch']
    save_check_every_n_epoch = general_settings['save_check_every_n_epoch']

    patience = early_stopping_settings['patience']
    min_delta = early_stopping_settings['min_delta']

    # Architecture settings
    architecture_settings = cv_settings['architecture']

    hidden_layers = architecture_settings['hidden_layers']
    lag_time = architecture_settings['lag_time']

    # Build time-lagged dataset (composed by pairs of configs at time t, t+lag)
    timelagged_dataset = create_timelagged_dataset(features_dataframe, lag_time=lag_time)

    # Get the number of features and samples
    num_features = timelagged_dataset["data"].shape[1]
    num_samples = timelagged_dataset["data"].shape[0]
    num_training_samples =int(num_samples*training_validation_lengths[0])

    # Check the batch size is not larger than the number of samples in the training set
    if batch_size >= num_training_samples:
        batch_size = common.closest_power_of_two(num_samples*training_validation_lengths[0])
        logger.warning(f'The batch size is larger than the number of samples in the training set. Setting the batch size to the closest power of two: {batch_size}')

    logger.info('Calculating DeepTICA CV...')

    # DeepTICA settings
    nn_layers = [num_features] + hidden_layers + [cv_dimension]
    nn_options = {'activation': 'shifted_softplus', 'dropout': dropout} 

    # Optimizer
    opt_name = optimizer_settings['name']
    optimizer_options = optimizer_settings['kwargs']

    # Create options
    options = {"nn": nn_options,
               "optimizer": optimizer_options}

    # Learning rate scheduler
    if lr_scheduler_settings is not None:

        lr_scheduler = {'scheduler': getattr(torch.optim.lr_scheduler, lr_scheduler_settings['name'])}
        lr_scheduler.update(lr_scheduler_settings['kwargs'])

        # Update options
        options.update({"lr_scheduler": lr_scheduler, 
                        "lr_interval": "epoch", 
                        "lr_monitor": "valid_loss", 
                        "lr_frequency": check_val_every_n_epoch})
        
        # Make early stopping patience larger than learning rate scheduler patience
        patience = max(patience, lr_scheduler_settings['kwargs']['patience']*2)

    converged = False
    tries = 0

    # Train until model finds a good solution
    while not converged and tries < max_tries:
        try: 

            tries += 1

            # Debug
            logger.debug(f'Splitting the dataset...')

            # Build time-lagged datamodule, split the dataset into training and validation
            timelagged_datamodule = DictModule(
                random_split = random_split,
                dataset = timelagged_dataset,
                lengths = training_validation_lengths,
                batch_size = batch_size,
                shuffle = shuffle, 
                generator = torch.manual_seed(seed))
    
            # Debug
            logger.debug(f'Initializing DeepTICA object...')
            
            # Define model
            dtica_model = DeepTICA(nn_layers, options=options)

            # Set optimizer name
            dtica_model._optimizer_name = opt_name

            # Debug
            logger.debug(f'Initializing metrics and callbacks...')

            # Define MetricsCallback to store the loss
            metrics = MetricsCallback()

            # Define EarlyStopping callback to stop training
            early_stopping = EarlyStopping(
                monitor="valid_loss", 
                min_delta=min_delta, 
                patience=patience, 
                mode = "min")

            # Define ModelCheckpoint callback to save the best model
            checkpoint = ModelCheckpoint(
                dirpath=output_path,
                monitor="valid_loss",                      # Quantity to monitor
                save_last=False,                           # Save the last checkpoint
                save_top_k=1,                              # Number of best models to save according to the quantity monitored
                save_weights_only=True,                    # Save only the weights
                filename=None,                             # Default checkpoint file name '{epoch}-{step}'
                mode="min",                                # Best model is the one with the minimum monitored quantity
                every_n_epochs=save_check_every_n_epoch)   # Number of epochs between checkpoints
            
            # Debug
            logger.debug(f'Initializing Trainer...')

            # Define trainer
            trainer = lightning.Trainer(            # accelerator="cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
                callbacks=[metrics, early_stopping, checkpoint],
                max_epochs=max_epochs, 
                logger=False, 
                enable_checkpointing=True,
                enable_progress_bar = False, 
                check_val_every_n_epoch=check_val_every_n_epoch)

            # Debug
            logger.debug(f'Training...')

            trainer.fit(dtica_model, timelagged_datamodule)

            # Get validation and training loss
            validation_loss = metrics.metrics['valid_loss']

            # Check the evolution of the loss
            converged = model_has_converged(validation_loss, patience, check_val_every_n_epoch)
            if not converged:
                logger.warning('Deep TICA has not found a good solution. Re-starting training...')

        except Exception as e:
            logger.error(f'Deep TICA training failed. Error message: {e}')
            logger.info('Retrying Deep TICA training...')

    try:
        # Load best model from checkpoint
        best_model = DeepTICA.load_from_checkpoint(checkpoint.best_model_path)
        
        # Find the score of the best model
        best_model_score = checkpoint.best_model_score
        logger.info(f'Best model score: {best_model_score}')

        # Find the epoch where the best model was found
        best_index = metrics.metrics['valid_loss'].index(best_model_score)
        best_epoch = metrics.metrics['epoch'][best_index]
        logger.info(f'Took {best_epoch} epochs')

        # Find eigenvalues of the best model
        best_eigvals = [metrics.metrics[f'valid_eigval_{i+1}'][best_index] for i in range(cv_dimension)]
        for i in range(cv_dimension):
            logger.info(f'Eigenvalue {i+1}: {best_eigvals[i]}')

        # Save the loss if requested
        if training_settings.get('save_loss', False):
            np.save(os.path.join(output_path, 'train_loss.npy'), np.array(metrics.metrics['train_loss']))
            np.save(os.path.join(output_path, 'valid_loss.npy'), np.array(metrics.metrics['valid_loss']))
            np.save(os.path.join(output_path, 'epochs.npy'), np.array(metrics.metrics['epoch']))
            np.savetxt(os.path.join(output_path, 'model_score.txt'), np.array([best_model_score]))
            np.savetxt(os.path.join(output_path, 'eigenvalues.txt'), np.array(best_eigvals))

        # Put model in evaluation mode
        best_model.eval()

        # Plot eigenvalues
        ax = plot_metrics(metrics.metrics,
                            labels=[f'Eigenvalue {i+1}' for i in range(cv_dimension)], 
                            keys=[f'valid_eigval_{i+1}' for i in range(cv_dimension)],
                            ylabel='Eigenvalue',
                            yscale=None)

        # Save figure
        ax.figure.savefig(os.path.join(output_path, f'eigenvalues.png'), dpi=300, bbox_inches='tight')
        ax.figure.clf()

        # Plot loss: squared sum of the eigenvalues
        ax = plot_metrics(metrics.metrics,
                            labels=['Training', 'Validation'], 
                            keys=['train_loss', 'valid_loss'], 
                            linestyles=['--','-'], colors=['fessa1','fessa5'], 
                            yscale=None)

        # Save figure
        ax.figure.savefig(os.path.join(output_path, f'loss.png'), dpi=300, bbox_inches='tight')
        ax.figure.clf()
        
    except Exception as e:
        logger.error(f'Failed to save/plot the loss. Error message: {e}')

    if converged:
        try:
            # Data projected onto original latent space of the best model
            with torch.no_grad():
                best_model.postprocessing = None
                projected_features = best_model(torch.Tensor(timelagged_dataset[:]["data"]))

            # Normalize the latent space
            norm =  Normalization(cv_dimension, mode='min_max', stats = Statistics(projected_features) )
            best_model.postprocessing = norm
            
            # Data projected onto normalized latent space
            with torch.no_grad():
                projected_features = best_model(torch.Tensor(timelagged_dataset[:]["data"])).numpy()

            # If reference data is provided, project it as well
            if ref_features_dataframe is not None:
                projected_ref_features = []
                for df in ref_features_dataframe:
                    ref_features_array = df.to_numpy(dtype=np.float32)
                    with torch.no_grad():
                        projected_ref_features.append(best_model(torch.Tensor(ref_features_array)).numpy())
            else:
                projected_ref_features = None

            # Save model
            best_model.to_torchscript(os.path.join(output_path,'weights.ptc'), method='trace')

            # Delete checkpoint file
            if os.path.exists(checkpoint.best_model_path):
                os.remove(checkpoint.best_model_path)

            # Create CV labels
            cv_labels = [f'DeepTIC {i+1}' for i in range(cv_dimension)]

            # Create FES along the CV
            figures.plot_fes(
                X=projected_features, 
                cv_labels=cv_labels, 
                X_ref=projected_ref_features,
                X_ref_labels=ref_labels,
                settings=figures_settings['fes'], 
                output_path=output_path) 

            project_traj(projected_features, cv_labels, figures_settings, clustering_settings, trajectory, topology, output_path)

        except Exception as e:
            logger.error(f'Failed to project the trajectory. Error message: {e}')
    else:
        logger.warning('Deep TICA training did not find a good solution after maximum tries.')

def model_has_converged(validation_loss: List, patience: int, check_val_every_n_epoch: int):
    """
    Check if there is any problem with the training of the model.

    - Check if the validation loss has decreased by the end of the training.
    - Check if we have at least 'patience' x 'check_val_every_n_epoch' epochs.

    Inputs
    ------

        validation_loss:         Validation loss for each epoch.
        training_loss:           Training loss for each epoch.
        patience:                Number of checks with no improvement after which training will be stopped.
        check_val_every_n_epoch: Number of epochs between checks.
        min_delta:               Minimum change in the validation loss to qualify as an improvement.
    """

    # Soft convergence condition: Check if the minimum of the validation loss is lower than the initial value
    if min(validation_loss) > validation_loss[0]:
        logger.warning('Validation loss has not decreased by the end of the training.')
        return False

    # Check if we have at least 'patience' x 'check_val_every_n_epoch' epochs
    if len(validation_loss) < patience*check_val_every_n_epoch:
        logger.warning('The trainer did not run for enough epochs.')
        return False

    return True

def project_traj(projected_features: np.ndarray, cv_labels: List[str], figures_settings: Dict, clustering_settings: Dict, 
                 trajectory: str, topology: str, output_path: str):
    """
    Plot the trajectory projection onto the CV space and cluster the projected features if requested.

    Inputs
    ------

        projected_features:  Array containing the projected features.
        cv_labels:           List of labels for the CVs.
        figures_settings:    Dictionary containing the settings for figures.
        clustering_settings: Dictionary containing the settings for clustering the projected features.
        trajectory:          Path to the trajectory file that will be clustered corresponding to the features dataframe.
        topology:            Path to the topology file of the trajectory.
        output_path:         Path to the output folder where the projected trajectory will be saved.   
    """

    logger.info('Projecting trajectory...')

    # Create a pandas DataFrame from the data and the labels
    projected_traj_df = pd.DataFrame(projected_features, columns=cv_labels)
    
    # Add a column with the order of the data points
    projected_traj_df['order'] = np.arange(projected_traj_df.shape[0])

    if clustering_settings['run']:

        figure_settings = figures_settings['projected_clustered_trajectory']
    
        # Cluster the projected features
        cluster_labels, centroids = statistics.optimize_clustering(projected_features, clustering_settings)

        # Add cluster labels to the projected trajectory DataFrame
        projected_traj_df['cluster'] = cluster_labels

        if len(centroids) > 0:
            # Find centroids in data
            centroids_df = statistics.find_centroids(projected_traj_df, centroids, cv_labels)

        # Generate color map for clusters
        num_clusters = len(np.unique(cluster_labels))
        cmap = figures.generate_cmap(num_clusters, figure_settings['cmap'])
        
        if len(cv_labels) == 2:

            # Create a 2D plot of the projected trajectory
            figures.plot_clustered_trajectory(
                data_df = projected_traj_df, 
                axis_labels = cv_labels,
                cluster_label = 'cluster', 
                settings = figure_settings, 
                file_path = os.path.join(output_path,'trajectory_clustered.png'),
                cmap = cmap)

        # Create a plot with the size of the clusters
        figures.plot_clusters_size(cluster_labels, cmap, output_path)

        # Extract frames from the trajectory
        if (None not in [trajectory, topology]) and (len(centroids) > 0):
            md.extract_clusters_from_traj(trajectory_path = trajectory, 
                                        topology_path = topology, 
                                        traj_df = projected_traj_df, 
                                        centroids_df = centroids_df,
                                        cluster_label = 'cluster',
                                        frame_label = 'order', 
                                        output_folder = os.path.join(output_path, 'clustered_traj'))
        elif len(centroids) == 0:
            logger.warning('No centroids found. Skipping the extraction of clusters from the trajectory.')
        else:
            logger.warning('Trajectory and/or topology files are missing. Skipping the extraction of clusters from the trajectory.')
            logger.info(f"Trajectory: {trajectory}")
            logger.info(f"Topology: {topology}")
            
    if len(cv_labels) == 2:

        # Create a 2D plot of the projected trajectory
        figures.plot_projected_trajectory(
            data_df = projected_traj_df, 
            axis_labels = cv_labels, 
            frame_label = 'order',
            settings = figures_settings['projected_trajectory'], 
            file_path = os.path.join(output_path,'trajectory.png'))
    
    # Erase the order column
    projected_traj_df.drop('order', axis=1, inplace=True)
    
    # Save the projected trajectory DataFrame
    projected_traj_df.to_csv(os.path.join(output_path,'projected_trajectory.csv'), index=False)
