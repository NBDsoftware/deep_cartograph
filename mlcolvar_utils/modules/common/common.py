import os
import sys
import yaml
import torch
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import PurePath
import matplotlib.pyplot as plt

from mlcolvar.utils.fes import compute_fes
from mlcolvar.data import DictDataset

# Set logger
logger = logging.getLogger(__name__)


# Related to FES
def create_fes_plot(X, temperature, num_bins, bandwidth, labels, max_fes, file_path):
    """
    Creates a figure of the free energy surface and saves it to a file.

    Parameters
    ----------

    X : array-like, list of arrays
        data with time series of the variables along which the FES is computed (1D or 2D)
    temperature: float
        temperature in Kelvin of the simulation that generated the data
    num_bins : int
        number of bins used for the KDE of the FES
    bandwidth : float
        bandwidth of for the KDE of the FES
    labels : list of str
        labels of the variables along which the FES is computed
    max_fes : float
        maximum value of the FES
    file_path : str
        File path where the figure is saved

    Returns
    -------
    fes : array-like
        Free energy surface
    grid : array-like
        Grid of the free energy surface
    bounds : array-like
        Bounds of the free energy surface
    """

    logger.info(f'Computing FES(' + ','.join(labels) + ')...')

    # Dimensions of the input data
    num_variables = X.shape[1]
    
    # Create figure
    fig, ax = plt.subplots()

    # Compute the FES along the given variables
    fes, grid, bounds, error = compute_fes(X, temp=temperature, ax=ax, plot=True, 
                                           plot_max_fes = max_fes, backend="KDEpy",
                                           num_samples=num_bins, bandwidth=bandwidth,
                                           eps=1e-10)

    # Set axis labels
    ax.set_xlabel(labels[0])

    if len(labels) > 1:
        ax.set_ylabel(labels[1]) 

    # Set axis limits
    if num_variables == 1:
        ax.set_ylim(0, max_fes)
        ax.set_xlim(bounds[0], bounds[1])
    elif num_variables == 2:
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xlim(bounds[0][0], bounds[0][1])

    # Save figure
    fig.savefig(file_path, dpi=300)

    return fes, grid, bounds

def create_cv_plot(fes, grid, cv, x, y, labels, cv_labels, max_fes, file_path):
    """
    Creates a figure with the value of the CV for each point (x,y) of the grid,
    adds the fes as a contour plot and saves it to a file.

    This can be used to project CVs on the FES whenever the CV is not a function of the FES variables. 
    
    Parameters
    ----------

    fes : array-like
        Free energy surface
    grid : array-like
        Grid of the free energy surface
    cv : array-like
        CV values
    x : array-like
        Values of the first variable of the FES
    y : array-like
        Values of the second variable of the FES
    labels : list of str
        labels of the variables of the FES
    cv_labels: list of str
        labels of the CVs
    max_fes : float
        maximum value of the FES
    file_path : str
        File path where the figure is saved
    """

    def add_fes_contour(fes, grid, ax, max_fes):
        """
        Adds a contour plot of the free energy surface to a given axis.
        
        Parameters
        ----------

        fes : array-like
            Free energy surface
        grid : array-like
            Grid of the free energy surface
        ax : matplotlib axis
            Axis where the contour plot is added
        max_fes : float
            maximum value of the FES
        """

        # Add contour plot 
        ax.contour(grid[0], grid[1], fes, levels=np.linspace(0, max_fes, 10), colors='black', linestyles='dashed', linewidths=0.5)

        return

    # Find the dimension of the CV
    cv_dimension = cv.shape[1]

    for component in range(cv_dimension):

        # Create figure
        fig, ax = plt.subplots()

        # Add contour plot of the FES
        add_fes_contour(fes, grid, ax, max_fes)

        # Add scatter plot of the CV
        ax.scatter(x, y, c=cv[:,component], cmap='viridis', s=1)

        # Add colorbar
        fig.colorbar(ax.collections[1], ax=ax)

        # Set axis labels
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        # Set title
        ax.set_title(cv_labels[component])

        # Modify file path to save the figure
        new_file_path = file_path.replace('.png',f'_{component+1}.png')

        # Save figure
        fig.savefig(new_file_path, dpi=300)

        # Close figure
        fig.clf()

    return


# General utils
def create_output_folder(output_path, folder_name = None):
    """
    Creates the output path if it does not exist.
    Creates a folder inside the output path if folder_name is not None.
    Return the path of the created folder or None if folder_name is None.

    Parameters
    ----------

    output_path : str
        Path where the output files are saved
    folder_name : str
        Name of the folder to be created

    Returns
    -------

    output_path : str or None
        Path of the created folder or None if folder_name is None.
    """

    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create folder inside the output directory
    if folder_name is not None:
        folder_path = os.path.join(output_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    return folder_path if folder_name is not None else None

def files_exist(*file_path):
    '''
    Returns true if all files exist

    Inputs
    ------

        file_path  (str): variable number of paths to files including filename
    
    Output
    ------

        exist (bool): True if they all exist. False if any of them doesnt exist
    '''

    exist = True

    for path in file_path:

        exist = exist and os.path.isfile(path)
            
    return exist


    """
    Function to read parameters from the configuration file. Exits if configuration file is not found.
    """
    
    # Read global parameters from configuration file
    if files_exist(configuration_path):
        with open(configuration_path) as config_file:
            global_parameters = yaml.load(config_file, Loader = yaml.FullLoader)
    else:
        logger.error(f"Configuration file: {configuration_path} not found")
        sys.exit()
    
    return global_parameters

def get_global_parameters(configuration_path: str, output_folder: str = None) -> dict:
    """
    Function to read global parameters from configuration file. Exits if configuration file is not found.
    It also copies a backup of the configuration file to the output folder.

    Inputs
    ------

        configuration_path (str): Path to configuration file
        output_folder      (str): Path to output folder

    Outputs
    -------
        
        global_parameters (dict): Dictionary with global parameters
    """
    
    # Read global parameters from configuration file
    if files_exist(configuration_path):
        with open(configuration_path) as config_file:
            global_parameters = yaml.load(config_file, Loader = yaml.FullLoader)
    else:
        logger.error("Configuration file not found")
        sys.exit(1)
    
    # Copy configuration file to output folder
    if output_folder is not None:
        bck_configuration_path = get_unique_path(os.path.join(output_folder, "input_bck.yml"))
        shutil.copyfile(configuration_path, bck_configuration_path)
    
    return global_parameters


# Features utils
def get_filter_dict(features_path: str, feat_regex: str):
    """
    Create the filter dictionary to select the features to use. Either from
    a list of features or a regex.

    Parameters
    ----------
    features_path : str
        Path to the file containing a list of features
    feat_regex : str
        Regular expression to select the features
    
    Returns
    -------
    
    filter_dict : dict
        Dictionary with the filter to select the features
    """

    if features_path is not None:
        # Features path is given, load the list of features and use it to create the filter dictionary
        used_features = np.loadtxt(features_path, dtype=str)
        logger.info(f' Using features in {features_path}')
        filter_dict = dict(items=used_features)
    else:
        # Features path is not given
        if feat_regex is not None:
            # Regex is given, use it to create the filter dictionary
            logger.info(f' Using regex to select features: {feat_regex}')
            filter_dict = dict(regex=feat_regex)
        else:
            # No features path or regex is given, use default filter
            logger.info(' Using all features except time and *.bias columns')
            filter_dict = dict(regex='^(?!.*time)^(?!.*bias)')
    
    return filter_dict


# Related to i/o
def create_dataset_from_dataframe(df: pd.DataFrame, filter_args: dict = None, verbose: bool = True):
    """
    Initialize a dataset from a dataframe. Suitable for supervised/unsupervised tasks.

    Parameters
    ----------
    filter_args: dict, optional
        Dictionary of arguments which are passed to df.filter() to select descriptors (keys: items, like, regex), by default None
        Note that 'time' and '*.bias' columns are always discarded.
    verbose : bool, optional
        Print info on the datasets, by default True
    create_labels: bool, optional
        Assign a label to each file, default True if more than a file is given, otherwise False
    kwargs : optional
        args passed to mlcolvar.utils.io.load_dataframe

    Returns
    -------
    torch.Dataset
        Torch labeled dataset of the given data
    """

    # filter inputs
    df_data = df.filter(**filter_args) if filter_args is not None else df.copy()
    df_data = df_data.filter(regex="^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)")

    if verbose:
        print(f"\n - Loaded dataframe {df.shape}:", list(df.columns))
        print(f" - Descriptors {df_data.shape}:", list(df_data.columns))

    # create DictDataset
    dictionary = {"data": torch.Tensor(df_data.values)}

    dataset = DictDataset(dictionary)

    return dataset

def save_list(list_to_save: list, path_to_save: str) -> None:
    """
    Function that saves a list to a file.

    Inputs
    ------

        list_to_save: List to save
        path_to_save: Path to the file where the list will be saved
    """

    # Open the file
    with open(path_to_save, 'w') as file_to_save:

        # Save the list
        for element in list_to_save:
            file_to_save.write(f'{element}\n')

    return

def get_unique_path(path: str):
    """
    Returns a unique path. 
    If it exists, it creates a new path with a suffix number.
    If it does not exist, it returns the original path.
    If the path is a file, it returns the path to the file with a suffix number, respecting the extension.

    Inputs
    ----------

        path           (str): original path
    
    Returns
    -------
        
        unique_path    (str): unique path
    """

    # Convert to PurePath object
    pure_path = PurePath(path)

    # If path exists
    if os.path.exists(path):

        # Get parent path
        parent_path = pure_path.parent

        # If path is a file
        if os.path.isfile(path):

            # Find the suffix and the stem
            stem = pure_path.stem
            suffix = pure_path.suffix
            
            # While the path exists
            i = 1
            while os.path.exists(path):

                # Create new path
                path = os.path.join(parent_path, stem + "_" + str(i) + suffix)
                i += 1
            
            # Return unique path
            return path
        
        # If path is a folder
        elif os.path.isdir(path):

            # Find the name
            name = pure_path.name

            # While the path exists
            i = 1
            while os.path.exists(path):

                # Create new path
                path = os.path.join(parent_path, name + "_" + str(i))
                i += 1
            
            # Return unique path
            return path
    
    # If path does not exist
    else:

        # Return original path
        return path