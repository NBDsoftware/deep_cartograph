"""
Utils for the train_colvars tool. 

These are functions that are used by the train_colvars tool that leverage or not
the modules in the mlcolvars_utils package.
"""

# General imports
import os
import logging
import numpy as np
import pandas as pd
import torch, lightning
from typing import List, Dict

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
from mlcolvar_utils.modules.md import md
from mlcolvar_utils.modules.common import common
from mlcolvar_utils.modules.figures import figures
from mlcolvar_utils.modules.statistics import statistics

# Set logger
logger = logging.getLogger(__name__)


def compute_pca(features_dataframe: pd.DataFrame, ref_features_dataframe: pd.DataFrame, cv_dimension: int, figures_settings: Dict, clustering_settings: Dict, output_folder: str):
    """
    Compute Principal Component Analysis (PCA) on the input features. Also, compute the Free Energy Surface (FES) along the PCA CVs.

    Inputs
    ------

        features_dataframe:     DataFrame containing the time series of the input features. Each column is a feature and each row is a time step.
        ref_features_dataframe: DataFrame containing the time series of the reference features. Each column is a feature and each row is a time step.
        cv_dimension:           Number of PCA components to consider for the CVs.
        figures_settings:       Dictionary containing the settings for figures.
        clustering_settings:    Dictionary containing the settings for clustering the projected features.
        output_folder:          Path to the output folder where the PCA results will be saved.
    """
    # NOTE: Normalization included or needed?
    # NOTE: Increase resolution in try/except blocks

    # Create output directory
    pca_output_path = common.create_output_folder(output_folder, 'pca')

    logger.info('Calculating PCA...')

    num_features = features_dataframe.shape[1]
    num_samples = features_dataframe.shape[0]

    # Use PCA to compute high variance linear combinations of the input features
    pca = PCA(in_features = num_features, out_features=min(num_features, num_samples))

    # Try to compute PCA
    try:
        # NOTE: Try as well
        # X = torch.tensor(features_dataframe.to_numpy())
        # pca_eigvals, pca_eigvecs = pca.compute(X, center = True)
        # projected_features_pca = pca(X)

        pca_eigvals, pca_eigvecs = pca.compute(X=torch.tensor(features_dataframe.to_numpy()))

        # Save the first cv_dimension eigenvectors as CVs 
        pca_cv = pca_eigvecs[:,0:cv_dimension].numpy()
        np.savetxt(os.path.join(pca_output_path,'weights.txt'), pca_cv)

        # Transform to array
        features_array = features_dataframe.to_numpy(dtype=np.float32)

        # Evaluate the CV on the colvars data
        projected_features = np.matmul(features_array, pca_cv)

        # If reference data is provided, project it as well
        if ref_features_dataframe is not None:
            ref_features_array = ref_features_dataframe.to_numpy(dtype=np.float32)
            projected_ref_features = np.matmul(ref_features_array, pca_cv)
        else:
            projected_ref_features = None

        # Create CV labels 
        cv_labels = [f'PC {i+1}' for i in range(cv_dimension)]

        # Create FES along the CV
        figures.plot_fes(
            X=projected_features, 
            X_ref=projected_ref_features,
            labels=cv_labels,
            settings=figures_settings.get('fes', {}), 
            output_path=pca_output_path)

        project_traj(projected_features, cv_labels, figures_settings, clustering_settings, pca_output_path)

    except Exception as e:
        logger.info(f'ERROR: PCA could not be computed. Error message: {e}')
        logger.info('Skipping PCA...')

def compute_ae(features_dataset: DictDataset, ref_features_dataset: DictDataset, cv_dimension: int, figures_settings: Dict, training_settings: Dict, clustering_settings: Dict, output_folder: str):
    """
    Train Autoencoder on the input features. The CV is the latent space of the Autoencoder. Also, compute the Free Energy Surface (FES) along the Autoencoder CVs.

    Inputs
    ------

        features_dataset:      Dataset containing the input features.
        ref_features_dataset:  Dataset containing the reference input features.
        cv_dimension:          Dimension of the Autoencoder latent space (= dimension of the CVs).
        figures_settings:      Dictionary containing the settings for the figures.
        training_settings:     Dictionary containing the settings for training the Autoencoder.
        clustering_settings:   Dictionary containing the settings for clustering the projected features.
        output_folder:         Path to the output folder where the Autoencoder results will be saved.
    """

    # Training settings
    training_validation_lengths = training_settings.get('lengths', [0.8, 0.2])
    batch_size = int(training_settings.get('batch_size', 32))
    shuffle = bool(training_settings.get('shuffle', True))
    seed = int(training_settings.get('seed', 0))
    patience = int(training_settings.get('patience', 10))
    min_delta = float(training_settings.get('min_delta', 1e-5))
    max_epochs = int(training_settings.get('max_epochs', 1000))
    dropout = float(training_settings.get('dropout', 0.1))
    hidden_layers = training_settings.get('hidden_layers', [15, 15])
    check_val_every_n_epoch = int(training_settings.get('check_val_every_n_epoch', 1))
    max_tries = int(training_settings.get('max_tries', 10))

    # Get the number of features and samples
    num_features = features_dataset["data"].shape[1]
    num_samples = features_dataset["data"].shape[0]

    # Check the batch size is not larger than the number of samples
    if batch_size > num_samples:
        logger.warning('WARNING: The batch size is larger than the number of samples. Setting the batch size to the number of samples.')
        batch_size = num_samples

    # Build datamodule, split the dataset into training and validation
    datamodule = DictModule(
        dataset = features_dataset,
        lengths = training_validation_lengths,
        batch_size = batch_size,
        shuffle = shuffle, 
        generator = torch.manual_seed(seed))

    # Create output directory
    output_path = common.create_output_folder(output_folder, 'autoencoder')

    logger.info('Training Autoencoder CV...')

    # Autoencoder settings
    encoder_layers = [num_features] + hidden_layers + [cv_dimension]
    nn_args = {'activation': 'shifted_softplus', 'dropout': dropout} 
    options=  {'encoder': nn_args, 'decoder': nn_args}

    converged = False
    tries = 0

    # Train until model finds a good solution
    while not converged and tries < max_tries:
        try:
        
            tries += 1

            # Define model
            ae_model = AutoEncoderCV(encoder_layers, options=options)  

            # Define callbacks
            metrics = MetricsCallback()
            early_stopping = EarlyStopping(
                monitor="valid_loss", 
                min_delta=min_delta,                 # Minimum change in the monitored quantity to qualify as an improvement
                patience=patience,                   # Number of checks with no improvement after which training will be stopped (see check_val_every_n_epoch)
                verbose=True, 
                mode="min")

            # Checkpoint to save the best model
            checkpoint = ModelCheckpoint(
                dirpath=output_path,
                monitor="valid_loss",               # Quantity to monitor
                save_last=True,                     # Save the last checkpoint NOTE: set to false? we are intereseted in the best model only
                save_top_k=1,                       # Number of best models to save according to the quantity monitored
                save_weights_only=True,             # Save only the weights
                filename=None,                      # Default checkpoint file name '{epoch}-{step}'
                mode="min",
                every_n_epochs=1)                   # Number of epochs between checkpoints

            # Define trainer
            trainer = lightning.Trainer(            # accelerator="cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
                callbacks=[metrics, early_stopping, checkpoint], 
                max_epochs=max_epochs,
                logger=None, 
                enable_checkpointing = True,
                enable_progress_bar = False,
                check_val_every_n_epoch=check_val_every_n_epoch)   # Check validation every n epochs  

            trainer.fit(ae_model, datamodule)

            # Get validation and training loss
            validation_loss = metrics.metrics['valid_loss']
            training_loss = metrics.metrics['train_loss_epoch']

            # Check the evolution of the loss
            if model_has_converged(validation_loss, training_loss, patience, check_val_every_n_epoch):
                converged = True
            else:
                logger.warning('WARNING: Autoencoder has not found a good solution. Re-starting training...')
        
        except Exception as e:
            logger.error(f'ERROR: Autoencoder training failed. Error message: {e}')
            logger.info('Retrying Autoencoder training...')

    if converged:
        try:
            # Load best model from checkpoint
            best_model = AutoEncoderCV.load_from_checkpoint(checkpoint.best_model_path)
            best_model_score = checkpoint.best_model_score

            # Log score
            logger.info(f'Best model score: {best_model_score}')

            # Save the loss if requested
            if training_settings.get('save_loss', False):
                np.save(os.path.join(output_path, 'train_loss.npy'), np.array(metrics.metrics['train_loss_epoch']))
                np.save(os.path.join(output_path, 'valid_loss.npy'), np.array(metrics.metrics['valid_loss']))
                np.savetxt(os.path.join(output_path, 'model_score.txt'), np.array([best_model_score]))

            # Put model in evaluation mode
            best_model.eval()

            # Plot loss
            ax = plot_metrics(metrics.metrics, 
                              labels=['Training', 'Validation'], 
                              keys=['train_loss_epoch', 'valid_loss'], 
                              linestyles=['--','-'], colors=['fessa1','fessa5'], 
                              yscale='log')

            # Save figure
            ax.figure.savefig(os.path.join(output_path, f'loss.png'), dpi=300)
            ax.figure.clf()

            # Data projected onto original latent space of the best model
            with torch.no_grad():
                projected_features = best_model(torch.Tensor(features_dataset[:]["data"]))

            # Normalize the latent space
            norm =  Normalization(cv_dimension, mode='min_max', stats = Statistics(projected_features) ) 
            best_model.postprocessing = norm

            # Data projected onto normalized latent space
            with torch.no_grad():
                projected_features = best_model(torch.Tensor(features_dataset[:]["data"])).numpy()
            
            # If reference data is provided, project it as well
            if ref_features_dataset is not None:
                with torch.no_grad():
                    projected_ref_features = best_model(torch.Tensor(ref_features_dataset[:]["data"])).numpy()
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
                X_ref=projected_ref_features,
                labels=cv_labels, 
                settings=figures_settings.get('fes', {}), 
                output_path=output_path)

            project_traj(projected_features, cv_labels, figures_settings, clustering_settings, output_path)

        except Exception as e:
            logger.error(f'ERROR: Failed to save/evaluate the best autoencoder model. Error message: {e}')
    else:
        logger.warning('WARNING: Autoencoder training did not find a good solution after maximum tries.')

def compute_tica(features_dataframe: pd.DataFrame, ref_features_dataframe: pd.DataFrame, cv_dimension: int, figures_settings: Dict, clustering_settings: Dict, output_folder: str):
    """
    Compute Time-lagged Independent Component Analysis (TICA) on the input features. Also, compute the Free Energy Surface (FES) along the TICA CVs.

    Inputs
    ------

        features_dataframe:     DataFrame containing the time series of the input features. Each column is a feature and each row is a time step.
        ref_features_dataframe: DataFrame containing the time series of the reference features. Each column is a feature and each row is a time step. 
        cv_dimension:           Number of TICA components to consider for the CVs.
        figures_settings:       Dictionary containing the settings for figures.
        clustering_settings:    Dictionary containing the settings for clustering the projected features.
        output_folder:          Path to the output folder where the PCA results will be saved.
    """
    
    # NOTE: Normalization included or needed?
    # NOTE: Increase resolution in try/except blocks

    # Create output directory
    tica_output_path = common.create_output_folder(output_folder, 'tica')

    logger.info('Calculating TICA CV...')

    # Build time-lagged dataset (composed by pairs of configs at time t, t+lag)
    timelagged_dataset = create_timelagged_dataset(features_dataframe, lag_time=10)

    # Get the number of features and samples
    num_features = timelagged_dataset["data"].shape[1]
    num_samples = timelagged_dataset["data"].shape[0]
    
    # Transform to array
    features_array = features_dataframe.to_numpy(dtype=np.float32)

    # Use TICA to compute slow linear combinations of the input features
    tica = TICA(in_features = num_features, out_features=min(num_features, num_samples))

    # Try to compute TICA
    try:
        tica_eigvals, tica_eigvecs = tica.compute(data=[timelagged_dataset['data'], timelagged_dataset['data_lag']], save_params = True, remove_average = True)

        # Save TICA eigenvectors and eigenvalues # NOTE: make this an option
        np.savetxt(os.path.join(tica_output_path,'tica_eigvals.txt'), tica_eigvals.numpy())
        np.savetxt(os.path.join(tica_output_path,'tica_eigvecs.txt'), tica_eigvecs.numpy())

        # Save the first cv_dimension eigenvectors as CVs
        tica_cv = tica_eigvecs[:,0:cv_dimension].numpy()
        np.savetxt(os.path.join(tica_output_path,'weights.txt'), tica_cv)

        # Evaluate the CV on the colvars data
        projected_features = np.matmul(features_array, tica_cv)

        # If reference data is provided, project it as well
        if ref_features_dataframe is not None:
            ref_features_array = ref_features_dataframe.to_numpy(dtype=np.float32)
            projected_ref_features = np.matmul(ref_features_array, tica_cv)
        else:
            projected_ref_features = None

        # Create CV labels
        cv_labels = [f'TIC {i+1}' for i in range(cv_dimension)]

        # Create FES along the CV
        figures.plot_fes(
            X=projected_features, 
            X_ref=projected_ref_features,
            labels=cv_labels,
            settings=figures_settings.get('fes', {}), 
            output_path=tica_output_path)
    
        project_traj(projected_features, cv_labels, figures_settings, clustering_settings, tica_output_path)
    
    except Exception as e:
        logger.info(f'ERROR: TICA could not be computed. Error message: {e}')
        logger.info('Skipping TICA...')

def compute_deep_tica(features_dataframe: pd.DataFrame, ref_features_dataframe: pd.DataFrame, cv_dimension: int, figures_settings: Dict, training_settings: Dict, clustering_settings: Dict, output_folder: str):
    """
    Train DeepTICA on the input features. The CV is the latent space of the DeepTICA model. Also, compute the Free Energy Surface (FES) along the DeepTICA CVs.

    Inputs
    ------

        features_dataframe:     Dataset containing the input features.
        ref_features_dataframe: Dataset containing the reference input features.
        cv_dimension:           Dimension of the DeepTICA latent space (= dimension of the CVs).
        figures_settings:       Dictionary containing the settings for the figures.
        training_settings:      Dictionary containing the settings for training the DeepTICA model.
        output_folder:          Path to the output folder where the DeepTICA results will be saved.
    """

    # Training settings
    training_validation_lengths = training_settings.get('lengths', [0.8, 0.2])
    batch_size = training_settings.get('batch_size', 32) # NOTE: should be larger for deep tica?
    shuffle = training_settings.get('shuffle', True) # NOTE: should be False? 
    seed = training_settings.get('seed', 0)
    patience = training_settings.get('patience', 10)
    min_delta = training_settings.get('min_delta', 1e-5)
    max_epochs = training_settings.get('max_epochs', 1000)
    dropout = training_settings.get('dropout', 0.1)
    hidden_layers = training_settings.get('hidden_layers', [15, 15])
    check_val_every_n_epoch = training_settings.get('check_val_every_n_epoch', 1)
    max_tries = training_settings.get('max_tries', 10)

    # Build time-lagged dataset (composed by pairs of configs at time t, t+lag)
    timelagged_dataset = create_timelagged_dataset(features_dataframe, lag_time=10)

    # Get the number of features and samples
    num_features = timelagged_dataset["data"].shape[1]
    num_samples = timelagged_dataset["data"].shape[0]

    # Check the batch size is not larger than the number of samples
    if batch_size > num_samples:
        logger.warning('WARNING: The batch size is larger than the number of samples. Setting the batch size to the number of samples.')
        batch_size = num_samples

    # Build time-lagged datamodule, split the dataset into training and validation
    timelagged_datamodule = DictModule(
        dataset = timelagged_dataset,
        lengths = training_validation_lengths,
        batch_size = batch_size,
        shuffle = shuffle, 
        generator = torch.manual_seed(seed))

    # Create output directory
    output_path = common.create_output_folder(output_folder, 'deep_tica')

    logger.info('Calculating DeepTICA CV...')

    # DeepTICA settings
    nn_layers = [num_features] + hidden_layers + [cv_dimension]
    nn_args = {'activation': 'shifted_softplus', 'dropout': dropout} 
    options= {'nn': nn_args}

    converged = False
    tries = 0

    # Train until model finds a good solution
    while not converged and tries < max_tries:
        try: 

            tries += 1
            
            # Define model
            dtica_model = DeepTICA(nn_layers, options=options)

            # Define callbacks
            metrics = MetricsCallback()
            early_stopping = EarlyStopping(
                monitor="valid_loss", 
                min_delta=min_delta, 
                patience=patience, 
                verbose = True, 
                mode = "min")

            # Checkpoint to save the best model
            checkpoint = ModelCheckpoint(
                save_top_k=1,                       # Number of best models to save according to the quantity monitored
                dirpath=output_path,
                filename=None,                      # Default checkpoint file name '{epoch}-{step}'
                monitor="valid_loss",               # Quantity to monitor
                mode="min",
                every_n_epochs=1)                   # Number of epochs between checkpoints.

            
            # Define trainer
            trainer = lightning.Trainer(            # accelerator="cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
                callbacks=[metrics, early_stopping, checkpoint],
                max_epochs=max_epochs, 
                logger=None, 
                enable_checkpointing=True,
                enable_progress_bar = False, 
                check_val_every_n_epoch=check_val_every_n_epoch)

            trainer.fit(dtica_model, timelagged_datamodule)

            # Get validation and training loss
            validation_loss = metrics.metrics['valid_loss']
            training_loss = metrics.metrics['train_loss_epoch']
            
            # Evaluate DeepTICA
            dtica_model.eval()

            # Check the evolution of the loss
            if model_has_converged(validation_loss, training_loss, patience, check_val_every_n_epoch):
                converged = True
            else:
                logger.warning('WARNING: Deep TICA has not found a good solution. Re-starting training...')

        except Exception as e:
            logger.error(f'ERROR: Deep TICA training failed. Error message: {e}')
            logger.info('Retrying Deep TICA training...')

    if converged:
        try:
            # Load best model from checkpoint
            best_model = DeepTICA.load_from_checkpoint(checkpoint.best_model_path)
            best_model_score = checkpoint.best_model_score

            # Log score
            logger.info(f'Best model score: {best_model_score}')

            # Save the loss if requested
            if training_settings.get('save_loss', False):
                np.save(os.path.join(output_path, 'train_loss.npy'), np.array(metrics.metrics['train_loss_epoch']))
                np.save(os.path.join(output_path, 'valid_loss.npy'), np.array(metrics.metrics['valid_loss']))
                np.savetxt(os.path.join(output_path, 'model_score.txt'), np.array([best_model_score]))

            # Put model in evaluation mode
            best_model.eval()

            # Plot eigenvalues
            ax = plot_metrics(metrics.metrics,
                                labels=[f'Eigenvalue {i+1}' for i in range(cv_dimension)], 
                                keys=[f'valid_eigval_{i+1}' for i in range(cv_dimension)],
                                yscale='log')

            # Save figure
            ax.figure.savefig(os.path.join(output_path, f'eigenvalues.png'), dpi=300)
            ax.figure.clf()

            # Plot loss: squared sum of the eigenvalues
            ax = plot_metrics(metrics.metrics,
                                labels=['Training', 'Validation'], 
                                keys=['train_loss_epoch', 'valid_loss'], 
                                linestyles=['--','-'], colors=['fessa1','fessa5'], 
                                yscale='log')
            
            # Save figure
            ax.figure.savefig(os.path.join(output_path, f'loss.png'), dpi=300)
            ax.figure.clf()

            # Data projected onto original latent space of the best model
            with torch.no_grad():
                projected_features = best_model(torch.Tensor(timelagged_dataset[:]["data"]))

            # Normalize the latent space
            norm =  Normalization(cv_dimension, mode='min_max', stats = Statistics(projected_features) )
            best_model.postprocessing = norm
            
            # Data projected onto normalized latent space
            with torch.no_grad():
                projected_features = best_model(torch.Tensor(timelagged_dataset[:]["data"])).numpy()

            # If reference data is provided, project it as well
            if ref_features_dataframe is not None:
                with torch.no_grad():
                    projected_ref_features = best_model(torch.Tensor(ref_features_dataframe.to_numpy(dtype=np.float32))).numpy()
            else:
                projected_ref_features = None

            # Save model
            best_model.to_torchscript(os.path.join(output_path,'model.ptc'), method='trace')

            # Create CV labels
            cv_labels = [f'DeepTIC {i+1}' for i in range(cv_dimension)]

            # Create FES along the CV
            figures.plot_fes(
                X=projected_features, 
                X_ref=projected_ref_features,
                labels=cv_labels, 
                settings=figures_settings.get('fes', {}), 
                output_path=output_path) 

            project_traj(projected_features, cv_labels, figures_settings, clustering_settings, output_path)

        except Exception as e:
            logger.info(f'ERROR: DeepTICA could not be computed. Error message: {e}')
            logger.info('Skipping DeepTICA...')

def model_has_converged(validation_loss: List, training_loss: list, patience: int, check_val_every_n_epoch: int, val_train_ratio: float = 2.0):
    """
    Check if there is any problem with the training of the model.

    - Check if the validation loss has decreased in the last 'patience' x 'check_val_every_n_epoch' epochs.
    - Check 
    Inputs
    ------

        validation_loss:         Validation loss for each epoch.
        training_loss:           Training loss for each epoch.
        patience:                Number of checks with no improvement after which training will be stopped.
        check_val_every_n_epoch: Number of epochs between checks.
        min_delta:               Minimum change in the validation loss to qualify as an improvement.
        val_train_ratio:         Threshold ratio between the validation and training loss at the end of the training.
    """

    # Check if the loss function at the end of the training has decreased wrt the initial value
    if validation_loss[-1] > validation_loss[0]:
        logger.info('WARNING: Validation loss has increased at the end of the training.')
        return False

    # Check if we have at least 'patience' x 'check_val_every_n_epoch' epochs
    if len(validation_loss) < patience*check_val_every_n_epoch:
        logger.info('WARNING: The trainer did not run for enough epochs.')
        return False

    # Check if the validation loss has decreased overall in the last 'patience' x 'check_val_every_n_epoch' epochs
    if not validation_loss[-1] < validation_loss[-patience*check_val_every_n_epoch]:
        logger.info('WARNING: Validation loss has not decreased in the last patience x check_val_every_n_epoch epochs.')
        return False
    
    # Check if the training and validation loss are similar at the end
    if validation_loss[-1] > val_train_ratio*training_loss[-1]:
        logger.info(f'WARNING: The validation loss is {val_train_ratio} times larger than the training loss at the end of the training.')
        return False

    return True

def project_traj(projected_features: np.ndarray, cv_labels: List[str], figures_settings: Dict, clustering_settings: Dict, 
                 output_path: str):
    """
    Plot the trajectory projection onto the CV space and cluster the projected features if requested.

    Inputs
    ------

        projected_features:  Array containing the projected features.
        cv_labels:           List of labels for the CVs.
        figures_settings:    Dictionary containing the settings for figures.
        clustering_settings: Dictionary containing the settings for clustering the projected features.
        output_path:         Path to the output folder where the projected trajectory will be saved.   
    """

    # Create a pandas DataFrame from the data and the labels
    projected_traj_df = pd.DataFrame(projected_features, columns=cv_labels)
    
    if clustering_settings.get('run', False):

        figure_settings = figures_settings.get('projected_clustered_trajectory', {})
    
        # Cluster the projected features
        cluster_labels, centroids = statistics.optimize_clustering(projected_features, clustering_settings)

        # Add a column with the order of the data points
        projected_traj_df['order'] = np.arange(projected_traj_df.shape[0])

        # Add cluster labels to the projected trajectory DataFrame
        projected_traj_df['cluster'] = cluster_labels

        # Find centroids in data
        centroids_df = statistics.find_centroids(projected_traj_df, centroids, cv_labels)

        # Generate color map for clusters
        num_clusters = len(np.unique(cluster_labels))
        cmap = figures.generate_cmap(num_clusters, figure_settings.get('cmap', 'viridis'))
        
        if len(cv_labels) == 2:

            # Create a 2D plot of the projected trajectory
            figures.plot_projected_trajectory(
                projected_traj_df, 
                axis_labels = cv_labels,
                cmap_label = 'cluster', 
                settings = figure_settings, 
                file_path = os.path.join(output_path,'trajectory_clustered.png'),
                cmap = cmap)

        # Create a plot with the size of the clusters
        figures.plot_clusters_size(cluster_labels, cmap, output_path)

        # Extract frames from the trajectory - not implemented yet
        md.extract_clusters_from_traj(trajectory_path = clustering_settings.get('traj_path'), 
                                      topology_path = clustering_settings.get('top_path'), 
                                      traj_df = projected_traj_df, 
                                      centroids_df = centroids_df,
                                      cluster_label = 'cluster',
                                      frame_label = 'order', 
                                      output_folder = os.path.join(output_path, 'clustered_traj'))

    if len(cv_labels) == 2:

        # Create a 2D plot of the projected trajectory
        figures.plot_projected_trajectory(
            projected_traj_df, 
            axis_labels = cv_labels, 
            cmap_label = 'order',
            settings = figures_settings.get('projected_trajectory', {}), 
            file_path = os.path.join(output_path,'trajectory.png'))
    
    # Erase the order column
    projected_traj_df.drop('order', axis=1, inplace=True)
    
    # Save the projected trajectory DataFrame
    projected_traj_df.to_csv(os.path.join(output_path,'projected_trajectory.csv'), index=False)
