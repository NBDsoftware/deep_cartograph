import torch, lightning
import logging.config
import numpy as np
import argparse
import shutil
import os

from lightning.pytorch.loggers import CSVLogger

#logger = CSVLogger(save_dir="experiments",   # directory where to save file
#                    name='myCV',             # name of experiment
#                    version=None             # version number (if None it will be automatically assigned)
#                    )

# assign callback to trainer
#trainer = lightning.Trainer(callbacks=[logger])

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# Import mlcolvars 
from mlcolvar.data import DictModule
from mlcolvar.core.stats import TICA, PCA
from mlcolvar.cvs import AutoEncoderCV, DeepTICA
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.timelagged import create_timelagged_dataset
from mlcolvar.utils.io import  create_dataset_from_files, load_dataframe    

# Import local modules
from mlcolvar_utils.modules.common import common

# Set up logger
logger = logging.getLogger(__name__)

########
# MAIN #
########

def train_cvs(colvars_path, features_path, feat_regex, cv_dimension, max_fes, temperature, output_path):
    """
    Main function of the script. It can be used to run the script from the command line.

    Parameters
    ----------

    colvars_path : str
        Path to the colvars file with the input data

    features_path : str
        Path to a file containing the features that should be used (these are used if the path is given)

    feat_regex : str
        Regular expression to select the features that should be used (all are used if no regex is given)

    cv_dimension : int
        Dimension of the CVs to train

    max_fes : float
        maximum value of the FES in the plots
    
    temperature : float
        Temperature in Kelvin of the simulation that generated the data

    output_path : str
        Path where the output files are saved
    """
    
    logger.info("This script trains collective variables using the mlcolvar library. \n")
    logger.info("The following CVs are computed: \n")
    logger.info(" - PCA")
    logger.info(" - Autoencoder")
    logger.info(" - TICA")
    logger.info(" - DeepTICA \n")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create output directory
    utils.create_output_folder(output_path)

    #################
    # INPUT COLVARS #
    #################

    # Get filter dictionary to select input features for the training
    filter_dict = common.get_filter_dict(features_path, feat_regex)

    logger.info('Loading input colvars...')

    # Load dataframe from colvars file
    colvars_dataframe_a = load_dataframe(colvars_path, start=0, stop=None, stride=1)
    features_dataset = common.create_dataset_from_dataframe(colvars_dataframe_a, filter_args=filter_dict, verbose=False)

    # Build dataset from colvars file with the selected features
    features_dataset, colvars_dataframe = create_dataset_from_files(file_names=[colvars_path], filter_args=filter_dict, verbose = False, return_dataframe=True)         

    # Filter dataframe 
    features_dataframe = colvars_dataframe.filter(**filter_dict)

    # Save feature names
    feature_names = features_dataframe.columns
    common.save_list(feature_names, os.path.join(output_path,'feature_names.txt'))

    # Transform to array
    features_array = features_dataframe.to_numpy(dtype=np.float32)

    # Log number of pairwise distances and samples
    num_features = features_dataframe.shape[1]
    num_samples = features_dataframe.shape[0]
    logger.info(f' Number of samples: {num_samples}')
    logger.info(f' Number of features: {num_features}')

    # NOTE: mean = 0 in features dataset and features dataframe?

    # Build datamodule, split the dataset into training and validation
    datamodule = DictModule(features_dataset, lengths =[0.8 ,0.2]) # NOTE: batch_size = ..., shuffle?

    # Build time-lagged dataset (composed by pairs of configs at time t, t+lag)
    timelagged_dataset = create_timelagged_dataset(features_dataframe, lag_time=10)

    # Build time-lagged datamodule, split the dataset into training and validation
    timelagged_datamodule = DictModule(timelagged_dataset, lengths =[0.8 ,0.2]) # NOTE: batch_size = ..., shuffle?


    ###########
    # CV: PCA #
    ###########

    # Create output directory
    pca_output_path = common.create_output_folder(output_path, 'pca')

    logger.info('Calculating PCA CV...')

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
        np.savetxt(os.path.join(pca_output_path,'pca.txt'), pca_cv)

        # Evaluate the CV on the colvars data
        projected_features_pca = np.matmul(features_array, pca_cv)

        # Create CV labels 
        cv_labels = [f'PC {i+1}' for i in range(cv_dimension)]

        # Create FES along the CV
        fes_pca, grid_pca, bounds_pca = common.create_fes_plot(X = projected_features_pca,
                                                        temperature=temperature,
                                                        num_bins=100,
                                                        bandwidth=0.01,
                                                        labels=cv_labels,
                                                        max_fes = max_fes,
                                                        file_path = os.path.join(pca_output_path,'fes_pca.png'))
        
    except Exception as e:
        logger.info(f'ERROR: PCA could not be computed. Error message: {e}')
        logger.info('Skipping PCA...')

    ###################
    # CV: Autoencoder #
    ###################

    # Create output directory
    ae_output_path = common.create_output_folder(output_path, 'autoencoder')

    logger.info('Calculating Autoencoder CV...')

    # Autoencoder settings
    encoder_layers = [num_features, 15, 15, cv_dimension]
    nn_args = {'activation': 'shifted_softplus'} 
    # norm_in = {'mean':  , 'range':  , 'mode': 'mean_std'}
    options=  {'encoder': nn_args, 'decoder': nn_args}      # NOTE: see norm_in option (normalization layer)

    good_sol = False
    max_tries = 10
    tries = 0

    # Try to train the Autoencoder
    try:
        # Train until model finds a good solution
        while not good_sol and tries < max_tries:
        
            # Define model
            ae_model = AutoEncoderCV(encoder_layers, options=options)  

            # Define callbacks
            metrics = MetricsCallback()
            early_stopping = EarlyStopping(monitor="valid_loss", min_delta=1e-5, patience=10, verbose = True, mode = "min")

            # Checkpoint to save the best model
            checkpoint = ModelCheckpoint(save_top_k=1,          # number of models to save (top_k=1 means only the best one is stored)
                                        monitor="valid_loss")    # quantity to monitor

            # Define trainer
            trainer = lightning.Trainer(callbacks=[metrics, early_stopping, checkpoint], max_epochs=1000, logger=None, enable_checkpointing=True) # accelerator="cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"

            logger.info('Training Autoencoder...')
            trainer.fit(ae_model, datamodule)

            # Evaluate Autoencoder
            ae_model.eval()

            # Plot loss
            ax = plot_metrics(metrics.metrics, keys=['train_loss_epoch','valid_loss'], linestyles=['-.','-'], colors=['fessa1','fessa5'], yscale='log')

            # Save figure
            ax.figure.savefig(os.path.join(ae_output_path,f'autoencoder_loss_{tries}.png'), dpi=300)

            # Close figure
            ax.figure.clf()

            # Get main metric: validation loss should decrease
            main_metric = metrics.metrics['valid_loss']

            # If valid_loss has not decreased, the model has not found a good solution
            if main_metric[-1] > main_metric[0]:
                logger.info('WARNING: Autoencoder has not found a good solution. Re-starting training...')
            else:
                good_sol = True

            tries += 1

        # Save model NOTE: which model is saved? the best one or the last one?
        ae_model.to_torchscript(os.path.join(ae_output_path,'autoencoder.ptc'), method='trace')

        # Load best model
        best_ae_model = ae_model.load_from_checkpoint(checkpoint.best_model_path)
        best_ae_model.to_torchscript(file_path = checkpoint.best_model_path.replace(".ckpt",".ptc"), method='trace')
        logger.info(f'Best model saved at {checkpoint.best_model_path}')

        # Evaluate model on the colvars data
        projected_features_ae = ae_model.forward_cv(torch.tensor(features_array)).detach().numpy()

        # Create CV labels
        cv_labels = [f'AE {i+1}' for i in range(cv_dimension)]

        # Create FES along the CV
        fes_ae, grid_ae, bounds_ae = common.create_fes_plot(X = projected_features_ae,
                                                    temperature=temperature,
                                                    num_bins=100,
                                                    bandwidth=0.01,
                                                    labels=cv_labels,
                                                    max_fes = max_fes,
                                                    file_path = os.path.join(ae_output_path, 'fes_autoencoder.png'))
    except Exception as e:
        logger.info(f'ERROR: Autoencoder could not be computed. Error message: {e}')
        logger.info('Skipping Autoencoder...')

    ############
    # CV: TICA #
    ############

    # Create output directory
    tica_output_path = common.create_output_folder(output_path, 'tica')

    logger.info('Calculating TICA CV...')

    # Use TICA to compute slow linear combinations of the input features
    tica = TICA(in_features = num_features, out_features=min(num_features, num_samples))

    # Try to compute TICA
    try:
        tica_eigvals, tica_eigvecs = tica.compute(data=[timelagged_dataset['data'], timelagged_dataset['data_lag']], save_params = True, remove_average = True)

        # Save TICA eigenvectors and eigenvalues
        np.savetxt(os.path.join(tica_output_path,'tica_eigvals.txt'), tica_eigvals.numpy())
        np.savetxt(os.path.join(tica_output_path,'tica_eigvecs.txt'), tica_eigvecs.numpy())

        # Save the first cv_dimension eigenvectors as CVs
        tica_cv = tica_eigvecs[:,0:cv_dimension].numpy()
        np.savetxt(os.path.join(tica_output_path,'tica.txt'), tica_cv)

        # Evaluate the CV on the colvars data
        projected_features_tica = np.matmul(features_array, tica_cv)

        # Create CV labels
        cv_labels = [f'TIC {i+1}' for i in range(cv_dimension)]

        # Create FES along the CV
        fes_tica, grid_tica, bounds_tica = common.create_fes_plot(X = projected_features_tica,
                                                        temperature=temperature,
                                                        num_bins=100,
                                                        bandwidth=0.01,
                                                        labels=cv_labels,
                                                        max_fes = max_fes,
                                                        file_path = os.path.join(tica_output_path,'fes_tica.png'))
    
    except Exception as e:
        logger.info(f'ERROR: TICA could not be computed. Error message: {e}')
        logger.info('Skipping TICA...')
    
    #################
    # CV: Deep-TICA #
    #################

    # Create output directory
    dtica_output_path = common.create_output_folder(output_path, 'deep_tica')

    logger.info('Calculating DeepTICA CV...')

    # DeepTICA settings
    nn_layers = [num_features, 15, 15, cv_dimension]
    options= {'nn': {'activation': 'shifted_softplus'} }

    good_sol = False
    max_tries = 10
    tries = 0

    # Try to train DeepTICA
    try:
        # Train until model finds a good solution
        while not good_sol and tries < max_tries:
        
            # Define model
            dtica_model = DeepTICA(nn_layers, options=options)

            # Define callbacks
            metrics = MetricsCallback()
            early_stopping = EarlyStopping(monitor="valid_loss", min_delta=1e-5, patience=10, verbose = True, mode = "min")

            # Checkpoint to save the best model
            checkpoint = ModelCheckpoint(save_top_k=1,          # number of models to save (top_k=1 means only the best one is stored)
                                        monitor="valid_loss")    # quantity to monitor
            
            # Define trainer
            trainer = lightning.Trainer(callbacks=[metrics, early_stopping, checkpoint] ,max_epochs=None, logger=None, enable_checkpointing=True, enable_model_summary=False)

            logger.info('Training DeepTICA...')
            trainer.fit(dtica_model, timelagged_datamodule)

            # Evaluate DeepTICA
            dtica_model.eval()

            # Plot eigenvalues
            ax = plot_metrics(metrics.metrics, 
                            keys=[x for x in  metrics.metrics.keys() if 'valid_eigval' in x],
                            yscale='linear')

            # Save figure
            ax.figure.savefig(os.path.join(dtica_output_path,f'deeptica_eigvals_{tries}.png'), dpi=300)

            # Close figure
            ax.figure.clf()

            # Plot loss: - squared sum of the eigenvalues
            ax = plot_metrics(metrics.metrics,
                            keys=['valid_loss'],
                            yscale='linear')
            
            # Save figure
            ax.figure.savefig(os.path.join(dtica_output_path,f'deeptica_loss_{tries}.png'), dpi=300)

            # Close figure
            ax.figure.clf()

            # Get main metric: valid_eigval_1 should increase 
            main_metric = metrics.metrics['valid_eigval_1']

            # If valid_loss has not increased, the model has not found a good solution
            if main_metric[-1] < main_metric[0]:
                logger.info('WARNING: DeepTICA has not found a good solution. Re-starting training...')
            else:
                good_sol = True

            tries += 1

        # Save model
        dtica_model.to_torchscript(os.path.join(dtica_output_path,'deeptica.ptc'), method='trace')

        # Evaluate model on the colvars data
        projected_features_dtica = dtica_model.forward_cv(torch.tensor(features_array)).detach().numpy()

        # Create CV labels
        cv_labels = [f'DeepTIC {i+1}' for i in range(cv_dimension)]

        # Create FES along the CV
        fes_dtica, grid_dtica, bounds_dtica = common.create_fes_plot(X = projected_features_dtica,
                                                            temperature=temperature,
                                                            num_bins=100,
                                                            bandwidth=0.01,
                                                            labels=cv_labels,
                                                            max_fes = max_fes,
                                                            file_path = os.path.join(dtica_output_path,'fes_deeptica.png'))
    
    except Exception as e:
        logger.info(f'ERROR: DeepTICA could not be computed. Error message: {e}')
        logger.info('Skipping DeepTICA...')

    # Move log file to output folder
    shutil.move(LOG_FILENAME, os.path.join(output_path, LOG_FILENAME))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("CV trainer with mlcolvar")

    parser.add_argument("--colvars", type=str, help="Path to the colvars file", required=True)
    parser.add_argument("--features", type=str, help="Path to a file containing the features that should be used (these are used if the path is given)", required=False)
    parser.add_argument("--feat_regex", type=str, help="Regular expression to select the features that should be used. All are used if no regex is given. E.g.: '*.x|*.y' would select g.x, e.x and t.y.", required=False)
    parser.add_argument("--cv_dimension", type=int, default=1, help="Dimension of the CVs", required=False)
    parser.add_argument("--max_fes", type=float, default=25, help="Maximum value of the FES", required=False)
    parser.add_argument("--temperature", type=float, default=300, help="Temperature in Kelvin", required=False)
    parser.add_argument("--output", type=str, default='cv_output', help="Output directory", required=False)
    
    args = parser.parse_args()

    train_cvs(args.colvars, args.features, args.feat_regex, args.cv_dimension, args.max_fes, args.temperature, args.output)