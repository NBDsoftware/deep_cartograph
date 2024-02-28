import torch, lightning
import logging.config
import numpy as np
import argparse
import shutil
import os

from mlcolvar.data import DictModule
from mlcolvar.core.stats import TICA, PCA
from mlcolvar.cvs import AutoEncoderCV, DeepTICA
from mlcolvar.utils.io import  create_dataset_from_files
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.timelagged import create_timelagged_dataset
from mlcolvar.utils.plot import plot_metrics
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import utils

# Set up logging
logging.config.fileConfig('/home/pnavarro/repos/CV_learning/mlcolvar/log_config/configuration.ini') 
LOG_FILENAME = 'cv_training.log'                            # Should coincide with the name in the configuration file

logger = logging.getLogger(__name__)

logger.info("Collective variable training: Alanine dipeptide \n")
logger.info("=============================================== \n")
logger.info("This script trains collective variables using the mlcolvar library. \n")
logger.info("Assumes the input colvars file is from a simulation of alanine dipeptide containing phi and psi torsion angles. \n")
logger.info("The following CVs are trained: \n")
logger.info(" - PCA")
logger.info(" - Autoencoder")
logger.info(" - TICA")
logger.info(" - DeepTICA \n")

########
# MAIN #
########

def main(colvars_path, ops_path, cv_dimension, max_fes, temperature, output_path):
    """
    Main function of the script. It can be used to run the script from the command line.

    Parameters
    ----------

    colvars_path : str
        Path to the colvars file with the input data

    ops_path : str
        Path to a file containing the order parameters that should be used (all are used if no file is given)

    cv_dimension : int
        Dimension of the CVs to train

    max_fes : float
        maximum value of the FES in the plots
    
    temperature : float
        Temperature in Kelvin of the simulation that generated the data

    output_path : str
        Path where the output files are saved
    """
    
    # Create output directory
    utils.create_output_folder(output_path)

    #################
    # INPUT COLVARS #
    #################

    logger.info('Loading input colvars...')

    # If order parameters are given, load the list and use it to filter the input data
    if ops_path is not None:
        used_ops = np.loadtxt(ops_path, dtype=str)
        logger.info(f' Using order parameters: {used_ops}')
        filter_dict = dict(items=used_ops)
    else:
        filter_dict = dict(regex='ha_')

    # Build dataset
    features_dataset, colvars_dataframe = create_dataset_from_files([colvars_path], 
                                            create_labels=False,
                                            filter_args=filter_dict, # select input descriptors using .filter method of Pandas dataframes
                                            return_dataframe=True)         # return also the dataframe of the loaded files (not only the input data)

    # Plot the FES in the phi-psi space
    fes, grid, bounds = utils.create_fes_plot(X = colvars_dataframe.loc[:,['phi','psi']].to_numpy(), 
                                        temperature=temperature,
                                        num_bins=100,
                                        bandwidth=0.1,
                                        labels=[r'$\phi$',r'$\psi$'],
                                        max_fes = max_fes+5,
                                        file_path = os.path.join(output_path,'fes_phi-psi.png'))

    # Filter dataframe to keep only features
    features_dataframe = colvars_dataframe.filter(**filter_dict)

    # Save input feature names
    feature_names = features_dataframe.columns
    utils.save_list(feature_names, os.path.join(output_path,'feature_names.txt'))

    # Transform to array
    features_array = features_dataframe.to_numpy()

    # Transform features from numpy.float64 to double
    features_array = features_array.astype(np.float32)

    # Print number of pairwise distances and samples
    num_features = features_dataframe.shape[1]
    num_samples = features_dataframe.shape[0]
    logger.info(f' Number of samples: {num_samples}')
    logger.info(f' Number of features: {num_features}')

    # Build datamodule
    datamodule = DictModule(features_dataset, lengths =[0.8 ,0.2])

    # Build time-lagged dataset (composed by pairs of configs at time t, t+lag). lag = 10 -> 10*100*2 fs -> 2 ps
    timelagged_dataset = create_timelagged_dataset(features_dataframe, lag_time=10)

    # Build time-lagged datamodule
    timelagged_datamodule = DictModule(timelagged_dataset, lengths =[0.8 ,0.2])


    ###########
    # CV: PCA #
    ###########

    # Create output directory
    pca_output_path = utils.create_output_folder(output_path, 'pca')

    logger.info('Calculating PCA CV...')

    # Use PCA to compute high variance linear combinations of the input features
    pca = PCA(in_features = num_features, out_features=min(num_features, num_samples))
    pca_eigvals, pca_eigvecs = pca.compute(X=torch.tensor(features_dataframe.to_numpy()))

    # Save the first cv_dimension eigenvectors as CVs 
    pca_cv = pca_eigvecs[:,0:cv_dimension].numpy()
    np.savetxt(os.path.join(pca_output_path,'pca_cv.txt'), pca_cv)

    # Evaluate the CV on the colvars data
    projected_features_pca = np.matmul(features_array, pca_cv)

    # Create CV labels 
    cv_labels = [f'PC {i+1}' for i in range(cv_dimension)]

    # Create CV plots
    utils.create_cv_plot(fes, grid, 
                cv = projected_features_pca, 
                x = colvars_dataframe['phi'].to_numpy(), 
                y = colvars_dataframe['psi'].to_numpy(), 
                labels=[r'$\phi$',r'$\psi$'],
                cv_labels = cv_labels, 
                max_fes = max_fes, file_path = os.path.join(pca_output_path,'cv_pca.png'))

    # Create FES along the CV
    fes_pca, grid_pca, bounds_pca = utils.create_fes_plot(X = projected_features_pca,
                                                    temperature=temperature,
                                                    num_bins=100,
                                                    bandwidth=0.01,
                                                    labels=cv_labels,
                                                    max_fes = max_fes,
                                                    file_path = os.path.join(pca_output_path,'fes_pca.png'))

    ###################
    # CV: Autoencoder #
    ###################

    # Create output directory
    ae_output_path = utils.create_output_folder(output_path, 'autoencoder')

    logger.info('Calculating Autoencoder CV...')

    # Autoencoder settings - NOTE: If all the features have the same units, should we normalize? - by default it standardizes the input
    encoder_layers = [num_features, 15, 15, cv_dimension]
    nn_args = {'activation': 'shifted_softplus'}
    options= {'encoder': nn_args, 'decoder': nn_args}

    good_sol = False
    max_tries = 10
    tries = 0

    # Train until model finds a good solution
    while not good_sol and tries < max_tries:
    
        # Define model
        ae_model = AutoEncoderCV(encoder_layers, options=options)  

        # Define callbacks
        metrics = MetricsCallback()
        early_stopping = EarlyStopping(monitor="valid_loss", min_delta=1e-5, patience=10)

        # Define trainer
        trainer = lightning.Trainer(callbacks=[metrics, early_stopping], max_epochs=None, logger=None, enable_checkpointing=False)

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

    # Save model
    ae_model.to_torchscript(os.path.join(ae_output_path,'autoencoder.ptc'), method='trace')

    # Evaluate model on the colvars data
    projected_features_ae = ae_model.forward_cv(torch.tensor(features_array)).detach().numpy()

    # Create CV labels
    cv_labels = [f'AE {i+1}' for i in range(cv_dimension)]

    # Create CV plot
    utils.create_cv_plot(fes, grid, 
                cv = projected_features_ae, 
                x = colvars_dataframe['phi'].to_numpy(), 
                y = colvars_dataframe['psi'].to_numpy(), 
                labels = [r'$\phi$',r'$\psi$'],
                cv_labels = cv_labels,
                max_fes = max_fes, file_path = os.path.join(ae_output_path, 'cv_autoencoder.png'))

    # Create FES along the CV
    fes_ae, grid_ae, bounds_ae = utils.create_fes_plot(X = projected_features_ae,
                                                temperature=temperature,
                                                num_bins=100,
                                                bandwidth=0.01,
                                                labels=cv_labels,
                                                max_fes = max_fes,
                                                file_path = os.path.join(ae_output_path, 'fes_autoencoder.png'))

    ############
    # CV: TICA #
    ############

    # Create output directory
    tica_output_path = utils.create_output_folder(output_path, 'tica')

    logger.info('Calculating TICA CV...')

    # Use TICA to compute slow linear combinations of the input features
    tica = TICA(in_features = num_features, out_features=min(num_features, num_samples))
    tica_eigvals, tica_eigvecs = tica.compute(data=[timelagged_dataset['data'], timelagged_dataset['data_lag']], save_params = True, remove_average = True)

    # Save TICA eigenvectors and eigenvalues
    np.savetxt(os.path.join(tica_output_path,'tica_eigvals.txt'), tica_eigvals.numpy())
    np.savetxt(os.path.join(tica_output_path,'tica_eigvecs.txt'), tica_eigvecs.numpy())

    # Save the first cv_dimension eigenvectors as CVs
    tica_cv = tica_eigvecs[:,0:cv_dimension].numpy()
    np.savetxt(os.path.join(tica_output_path,'tica_cv.txt'), tica_cv)

    # Evaluate the CV on the colvars data
    projected_features_tica = np.matmul(features_array, tica_cv)

    # Create CV labels
    cv_labels = [f'TIC {i+1}' for i in range(cv_dimension)]

    # Create CV plot
    utils.create_cv_plot(fes, grid, 
                cv = projected_features_tica, 
                x = colvars_dataframe['phi'].to_numpy(), 
                y = colvars_dataframe['psi'].to_numpy(), 
                labels=[r'$\phi$',r'$\psi$'],
                cv_labels = cv_labels,
                max_fes = max_fes, file_path = os.path.join(tica_output_path,'cv_tica.png'))

    # Create FES along the CV
    fes_tica, grid_tica, bounds_tica = utils.create_fes_plot(X = projected_features_tica,
                                                       temperature=temperature,
                                                       num_bins=100,
                                                       bandwidth=0.01,
                                                       labels=cv_labels,
                                                       max_fes = max_fes,
                                                       file_path = os.path.join(tica_output_path,'fes_tica.png'))
    
    #################
    # CV: Deep-TICA #
    #################

    # Create output directory
    dtica_output_path = utils.create_output_folder(output_path, 'deep_tica')

    logger.info('Calculating DeepTICA CV...')

    # DeepTICA settings
    nn_layers = [num_features, 15, 15, cv_dimension]
    options= {'nn': {'activation': 'shifted_softplus'} }

    good_sol = False
    max_tries = 10
    tries = 0

    # Train until model finds a good solution
    while not good_sol and tries < max_tries:
    
        # Define model
        dtica_model = DeepTICA(nn_layers, options=options)

        # Define callbacks
        metrics = MetricsCallback()
        early_stopping = EarlyStopping(monitor="valid_loss", min_delta=1e-5, patience=10)

        # Define trainer
        trainer = lightning.Trainer(callbacks=[metrics, early_stopping],max_epochs=None, logger=None, enable_checkpointing=False, enable_model_summary=False)

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

    # Create CV plot
    utils.create_cv_plot(fes, grid, 
                cv = projected_features_dtica, 
                x = colvars_dataframe['phi'].to_numpy(), 
                y = colvars_dataframe['psi'].to_numpy(), 
                labels=[r'$\phi$',r'$\psi$'],
                cv_labels = cv_labels,
                max_fes = max_fes, file_path = os.path.join(dtica_output_path,'cv_deeptica.png'))

    # Create FES along the CV
    fes_dtica, grid_dtica, bounds_dtica = utils.create_fes_plot(X = projected_features_dtica,
                                                          temperature=temperature,
                                                          num_bins=100,
                                                          bandwidth=0.01,
                                                          labels=cv_labels,
                                                          max_fes = max_fes,
                                                          file_path = os.path.join(dtica_output_path,'fes_deeptica.png'))

    # Move log file to output folder
    shutil.move(LOG_FILENAME, os.path.join(output_path, LOG_FILENAME))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("CV trainer with mlcolvar")

    parser.add_argument("--colvars", type=str, help="Path to the colvars file", required=True)
    parser.add_argument("--order_parameters", type=str, help="Path to a file containing the order parameters that should be used (all are used if no file is given)", required=False)
    parser.add_argument("--cv_dimension", type=int, default=1, help="Dimension of the CVs", required=False)
    parser.add_argument("--max_fes", type=float, default=25, help="Maximum value of the FES", required=False)
    parser.add_argument("--temperature", type=float, default=300, help="Temperature in Kelvin", required=False)
    parser.add_argument("--output", type=str, default='cv_output', help="Output directory", required=False)
    

    args = parser.parse_args()

    main(args.colvars, args.order_parameters, args.cv_dimension, args.max_fes, args.temperature, args.output)