# Import modules
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex

from mlcolvar.utils.fes import compute_fes

# Set logger
logger = logging.getLogger(__name__)

def plot_fes(X: np.ndarray, X_ref: np.ndarray, labels: List[str], settings: Dict, output_path: str):
    """
    Creates a figure of the free energy surface and saves it to a file.

    Parameters
    ----------

        X:           data with time series of the variables along which the FES is computed (1D or 2D)
        X_ref:       data with the reference values of the variables along which the FES is computed
        settings:    dictionary with the settings of the FES plot
        labels:      labels of the variables along which the FES is computed
        output_path: path where the outputs are saved
    """

    if settings.get('compute', True):

        # Create fes folder inside the output path
        output_path = os.path.join(output_path, 'fes')
        os.makedirs(output_path, exist_ok=True)

        cv_dimension = X.shape[1]

        if cv_dimension > 2:
            logger.warning('The FES can only be plotted for 1D or 2D CVs.')
            return

        logger.info(f'Computing FES(' + ','.join(labels) + ')...')

        # Find settings
        temperature = settings.get('temperature', 300)
        max_fes = settings.get('max_fes', 10)
        num_bins = settings.get('num_bins', 100)
        num_blocks = settings.get('num_blocks', 10)
        bandwidth = settings.get('bandwidth', 0.1)
        min_block_size = 20

        # Number of samples for the FES
        num_samples = X.shape[0]

        # Find block size
        block_size = int(num_samples/num_blocks)

        # If the block size is too small, reduce the number of blocks and issue a warning
        if block_size < min_block_size:
            num_blocks = int(num_samples/min_block_size)
            block_size = min_block_size
            logger.warning(f"Block size too small. Reducing the number of blocks to {num_blocks}")
        
        # Create figure
        fig, ax = plt.subplots()

        # Compute the FES along the given variables
        fes, grid, bounds, error = compute_fes(X, temp = temperature, ax = ax, plot = True, 
                                            plot_max_fes = max_fes, backend = "KDEpy",
                                            num_samples = num_bins, bandwidth = bandwidth,
                                            blocks = num_blocks, eps = 1e-10, bounds = find_limits(X, X_ref))
        
        # Save the FE values, the grid, the bounds and the error
        if settings.get('save', False):
            np.save(os.path.join(output_path, 'fes.npy'), fes)
            np.save(os.path.join(output_path, 'grid.npy'), grid)
            np.save(os.path.join(output_path, 'bounds.npy'), bounds)
            np.save(os.path.join(output_path, 'error.npy'), error)

        # Add reference data to the FES plot
        if X_ref is not None:

            # If the reference data is 2D
            if X_ref.shape[1] == 2:
                
                # Add as a scatter plot
                ax.scatter(X_ref[:,0], X_ref[:,1], c='black', s=5, label='Reference data')

            # If the reference data is 1D
            elif X_ref.shape[1] == 1:

                # Add as a histogram
                ax.hist(X_ref, bins=num_bins, color='red', alpha=0.5, density=True, label='Reference data')

        # Set axis labels
        ax.set_xlabel(labels[0])

        if len(labels) > 1:
            ax.set_ylabel(labels[1]) 

        # Enforce FES limit if needed (max_fes defined and 1D FES)
        if max_fes and cv_dimension == 1:
            ax.set_ylim(0, max_fes)

        # Enforce CV limits
        if cv_dimension == 1:
            ax.set_xlim(bounds)
        elif cv_dimension == 2:
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])

        ax.legend()

        # Save figure
        fig.savefig(os.path.join(output_path, 'fes.png'), dpi=300)

    return

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

def plot_projected_trajectory(data_df: pd.DataFrame, axis_labels: List[str], cmap_label: str, settings: Dict, file_path: str) -> None:
    """
    Create a scatter plot of a trajectory projected on a 2D space defined by the CVs given in labels. 
    The color of the markers is given by the value of cmap_label
    Adds the histograms of the data along each axis and save the figure to a file.

    Inputs
    ------

        data:       data with the trajectory projected on the 2D space
        labels:     labels of the CVs used to project the trajectory
        settings:   dictionary with the settings of the plot
        file_path:  path where the figure will be saved
    """

    if settings.get('plot', True):

        logger.info(f'Creating projected trajectory plot...')

        marker_size = settings.get('marker_size', 10)
        alpha = settings.get('alpha', 0.5)
        num_bins = settings.get('num_bins', 50)
        bw_adjust = settings.get('bandwidth', 0.5)
        cmap = settings.get('cmap', 'viridis')
        use_legend = settings.get('use_legend', False)
        if use_legend:
            legend = 'full' # Show all the labels, otherwise it will show only some representative labels
        else:
            legend = False

        # Create a JointGrid object with a colormap
        ax = sns.JointGrid(data=data_df, x=axis_labels[0], y=axis_labels[1])

        # Create a scatter plot of the data, color-coded by the order of the data points, modify the markers size
        scatter = ax.plot_joint(
            sns.scatterplot, 
            data=data_df,
            hue=cmap_label, 
            palette=cmap, 
            alpha=alpha, 
            edgecolor=".2", 
            linewidth=.5,
            legend=legend,
            s=marker_size)
        
        scatter.set_axis_labels(axis_labels[0], axis_labels[1])

        # Marginal histograms
        ax.plot_marginals(sns.histplot, kde=True, bins=num_bins, kde_kws = {'bw_adjust': bw_adjust})

        plt.tight_layout()

        # Save the figure
        plt.savefig(file_path, dpi=300)

def find_limits(X: np.ndarray, X_ref: np.ndarray) -> List:
    """
    Find the limits of the axis for the FES plot.

    Inputs
    ------

        X:             data with the time series of the variables along which the FES is computed
        X_ref:         data with the reference values of the variables along which the FES is computed

    Returns
    -------

        limits:        List with the limits of the axis for the FES plot
    """

    # Dimensions of the input data
    fes_dimension = X.shape[1]
    
    # If there is no reference data, set X_ref to X
    if X_ref is None:
        X_ref = X

    # Find the limits of the axis
    offset = 1e-3
    if fes_dimension == 1:
        limits = (min(np.min(X), np.min(X_ref))-offset, max(np.max(X), np.max(X_ref))+offset)
    else:
        limits = []
        for i in range(fes_dimension):
            min_x = min(np.min(X[:, i]), np.min(X_ref[:, i]))
            max_x = max(np.max(X[:, i]), np.max(X_ref[:, i]))
            limits.append((min_x-offset, max_x+offset))

    return limits