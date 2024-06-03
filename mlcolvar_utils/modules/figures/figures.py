# Import modules
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mlcolvar.utils.fes import compute_fes

# Set logger
logger = logging.getLogger(__name__)

def plot_fes(X: np.ndarray, labels: List[str], settings: Dict, file_path: str):
    """
    Creates a figure of the free energy surface and saves it to a file.

    Parameters
    ----------

    X:           data with time series of the variables along which the FES is computed (1D or 2D)
    settings:    dictionary with the settings of the FES plot
    labels:      labels of the variables along which the FES is computed
    file_path:   file path where the figure is saved
    """

    if settings.get('plot', True):

        logger.info(f'Computing FES(' + ','.join(labels) + ')...')

        # Find settings
        temperature = settings.get('temperature', 300)
        max_fes = settings.get('max_fes', 10)
        num_bins = settings.get('num_bins', 100)
        bandwidth = settings.get('bandwidth', 0.1)

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

def plot_projected_trajectory(data: np.ndarray, labels: List[str], settings: Dict, file_path: str) -> None:
    """
    Create a scatter plot of a trajectory projected on a 2D space defined by the CVs given in labels. 
    The color of the markers is given by the order of the data points.
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

        alpha = settings.get('alpha', 0.5)
        num_bins = settings.get('num_bins', 50)
        bw_adjust = settings.get('bandwidth', 0.5)
        cmap = settings.get('cmap', 'viridis')

        # Create a pandas DataFrame from the data and the labels
        data_df = pd.DataFrame(data, columns=labels)

        # Add a column with the order of the data points
        data_df['order'] = np.arange(data_df.shape[0])

        # Create a JointGrid object with a colormap
        ax = sns.JointGrid(data=data_df, x=labels[0], y=labels[1])

        # Create a scatter plot of the data, color-coded by the order of the data points
        scatter = ax.plot_joint(
            sns.scatterplot, 
            data=data_df,
            hue='order', 
            palette=cmap, 
            alpha=alpha, 
            edgecolor=".2", 
            linewidth=.5,
            legend=False)
        
        scatter.set_axis_labels(labels[0], labels[1])

        # Marginal histograms
        ax.plot_marginals(sns.histplot, kde=True, bins=num_bins, kde_kws = {'bw_adjust': bw_adjust})
        
        plt.tight_layout()

        # Save the figure
        plt.savefig(file_path, dpi=300)