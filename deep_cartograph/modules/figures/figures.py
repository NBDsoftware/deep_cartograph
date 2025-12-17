# Import modules
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Union, Optional, Literal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex

# Import local modules
from deep_cartograph.yaml_schemas.train_colvars import FesFigure 
from deep_cartograph.modules.common import package_is_installed

# Set logger
logger = logging.getLogger(__name__)

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
# Assuming other necessary imports (logger, package_is_installed, FesFigure, get_ranges) exist in your environment

def plot_fes(
    data: np.ndarray, 
    cv_labels: List[str],  
    settings: Dict, 
    output_path: str, 
    num_blocks: int = 1,
    sup_data: Optional[List[np.ndarray]] = None, 
    sup_data_labels: Optional[List[str]] = None,
    legend_cutoff: int = 10
    ):
    """
    Creates a figure of the free energy surface and saves it to a file.
    If len(sup_data) > legend_cutoff, the legend is saved to a separate 'fes_legend.png' file.

    Parameters
    ----------
        data:             data with time series of the variables along which the FES is computed (1D or 2D)
        cv_labels:        labels of the variables along which the FES is computed
        settings:         dictionary with the settings of the FES plot
        output_path:      path where the outputs are saved
        num_blocks:       number of blocks to use for the FES computation
        sup_data:         supplementary data to plot alongside the main data
        sup_data_labels:  labels of the supplementary data
        legend_cutoff:    maximum number of items allowed in the main plot legend before splitting
    """
    
    if not package_is_installed('mlcolvar', 'torch'):
        logger.debug('mlcolvar and torch are not installed. Skipping FES plot.')
        return
    
    import mlcolvar.utils.plot 
    from mlcolvar.utils.fes import compute_fes

    # Find settings
    font_size = 12
    tick_size = 10

    # Validate the settings
    settings = FesFigure(**settings).model_dump()

    if settings['compute']:
        
        cv_dimension = len(cv_labels)

        logger.info(f'Computing FES(' + ', '.join(cv_labels) + ')...')

        # Find settings
        temperature = settings['temperature']
        max_fes = settings['max_fes']
        num_bins = settings['num_bins']
        bandwidth = settings['bandwidth']
        min_block_size = 100

        # Number of samples for the FES
        num_samples = data.shape[0]

        # Find block size
        block_size = int(num_samples/num_blocks)

        # If the block size is too small, reduce the number of blocks and issue a warning
        if block_size < min_block_size:
            old_num_blocks = num_blocks
            num_blocks = max(1, int(num_samples/min_block_size))
            block_size = min_block_size
            logger.warning(f"Block size too small with {old_num_blocks} blocks and {num_samples} samples. Reducing the number of blocks to {num_blocks}")
        
        # Create figure
        fig, ax = plt.subplots()

        # Compute and plot the 1D or 2D FES along the given variables
        fes, fes_grid, fes_bounds, fes_error = compute_fes(data, temp = temperature, ax = ax, plot = True, 
                                            plot_max_fes = max_fes, backend = "KDEpy",
                                            num_samples = num_bins, bandwidth = bandwidth,
                                            blocks = num_blocks, eps = 1e-10, bounds = get_ranges(data))
        
        # Save the FE values, the grid, the bounds and the error
        if settings.get('save', False):
            np.save(os.path.join(output_path, 'fes.npy'), fes)
            np.save(os.path.join(output_path, 'fes_grid.npy'), fes_grid)
            np.save(os.path.join(output_path, 'fes_bounds.npy'), fes_bounds)
            np.save(os.path.join(output_path, 'fes_error.npy'), fes_error)

        # Add reference data to the FES plot
        if sup_data is not None:
            
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 
                    'P', '*', 'h', 'H',"D","d", '8'] 
            
            for i, sup_data_i in enumerate(sup_data):

                label = sup_data_labels[i] if sup_data_labels else ''
                
                # Cycle through markers
                marker = markers[i % len(markers)]

                # If the reference data is 1D -> histogram
                if sup_data_i.ndim == 1:
                    ax.hist(sup_data_i, bins=num_bins, alpha=0.5, density=True, label=label)
                # If the reference data is 2D -> scatter plot
                else: 
                    if sup_data_i.shape[1] == 2:
                        ax.scatter(sup_data_i[:,0], sup_data_i[:,1], s=8, label=label, marker=marker, alpha=1)
                    else:
                        logger.warning(f"Supplementary data {i} has {sup_data_i.shape[1]} dimensions > 2 dimensions. Skipping scatter plot of this data.")
                        continue

        # Set axis labels
        ax.set_xlabel(cv_labels[0], fontsize = font_size)

        if len(cv_labels) > 1:
            ax.set_ylabel(cv_labels[1]) 

        # Enforce FES limit if needed (max_fes defined and 1D FES)
        if max_fes and cv_dimension == 1:
            ax.set_ylim(0, max_fes)

        # Find the range of the data
        data_range = get_ranges(data, sup_data)
        
        # Enforce CV limits, mix data ranges and -1, 1 limits
        if cv_dimension == 1:
            ax.set_xlim(min(data_range[0], -1), max(data_range[1], 1))
        elif cv_dimension == 2:
            ax.set_xlim(min(data_range[0][0], -1), max(data_range[0][1], 1))
            ax.set_ylim(min(data_range[1][0], -1), max(data_range[1][1], 1))

        if sup_data_labels:
            num_items = len(sup_data) if sup_data else 0
            
            # Case 1: Few items, plot legend inside the main figure
            if num_items <= legend_cutoff:
                ax.legend(fontsize = font_size, framealpha=0.5)
            
            # Case 2: Many items, save legend to separate file
            else:
                logger.info(f"Sup_data count ({num_items}) exceeds cutoff ({legend_cutoff}). Saving legend to separate file.")
                
                # Extract handles and labels from the main plot
                handles, labels = ax.get_legend_handles_labels()
                
                if handles:
                    # Create a new figure just for the legend
                    # Dynamic height based on number of items
                    fig_leg = plt.figure(figsize=(4, 0.3 * num_items + 1)) 
                    
                    # Add legend to the new figure
                    fig_leg.legend(handles, labels, loc='center', fontsize=font_size, frameon=False)
                    
                    # Turn off axes so only the legend text/symbols appear
                    plt.axis('off') 
                    
                    # Save the legend figure
                    fig_leg.savefig(os.path.join(output_path, 'fes_legend.png'), bbox_inches='tight', dpi=300)
                    
                    # Close the legend figure to free memory
                    plt.close(fig_leg)

        # Set tick size
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        # Save main figure
        fig.savefig(os.path.join(output_path, 'fes.png'), dpi=300)

        # Close figure
        fig.clf()
        plt.close(fig) # Ensure matplotlib figure is fully closed

    return

def create_cv_plot(fes, fes_grid, cv, x, y, labels, cv_labels, max_fes, file_path):
    """
    Creates a figure with the value of the CV for each point (x,y) of the grid,
    adds the fes as a contour plot and saves it to a file.

    This can be used to project CVs on the FES whenever the CV is not a function of the FES variables. 
    
    Parameters
    ----------

    fes : array-like
        Free energy surface
    fes_grid : array-like
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

    def add_fes_contour(fes, fes_grid, ax, max_fes):
        """
        Adds a contour plot of the free energy surface to a given axis.
        
        Parameters
        ----------

        fes : array-like
            Free energy surface
        fes_grid : array-like
            Grid of the free energy surface
        ax : matplotlib axis
            Axis where the contour plot is added
        max_fes : float
            maximum value of the FES
        """

        # Add contour plot 
        ax.contour(fes_grid[0], fes_grid[1], fes, levels=np.linspace(0, max_fes, 10), colors='black', linestyles='dashed', linewidths=0.5)

        return

    # Find the dimension of the CV
    cv_dimension = cv.shape[1]

    for component in range(cv_dimension):

        # Create figure
        fig, ax = plt.subplots()

        # Add contour plot of the FES
        add_fes_contour(fes, fes_grid, ax, max_fes)

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

def clusters_scatter_plot(data: pd.DataFrame, column_labels: List[str], cluster_label: str, settings: Dict, file_path: str, cluster_colors: List) -> None:
    """
    Create a scatter plot of a trajectory projected on a 2D space defined by the CVs given in labels. 
    The color of the markers is given by the cluster.
    Adds the histograms of the data along each axis and save the figure to a file.

    Inputs
    ------

        data:            data with the trajectory projected on the 2D space
        column_labels:   labels of the CVs used to project the trajectory
        cluster_label:   label of the data used to color the markers
        settings:        dictionary with the settings of the plot
        file_path:       path where the figure will be saved
        cluster_colors:  list with the RGB colors to use for each cluster
    """
    
    if settings.get('plot', True):

        marker_size = settings.get('marker_size', 10)
        alpha = settings.get('alpha', 0.5)
        num_bins = settings.get('num_bins', 50)
        bw_adjust = settings.get('bandwidth', 0.5)

        # Sort data so that cluster -1 is plotted first
        sorted_data = data.sort_values(by=cluster_label, ascending=True)

        # Create a JointGrid object with a colormap
        ax = sns.JointGrid(data=sorted_data, x=column_labels[0], y=column_labels[1])

        # Create a scatter plot of the data, color-coded by the cluster
        scatter = ax.plot_joint(
            sns.scatterplot, 
            data=sorted_data,
            hue=cluster_label, 
            palette=cluster_colors, 
            alpha=alpha, 
            edgecolor=".2", 
            linewidth=.5,
            legend='full',
            s=marker_size)
        
        scatter.set_axis_labels(column_labels[0], column_labels[1])
        
        # Get range of the data
        data_range = get_ranges(data[[column_labels[0], column_labels[1]]].to_numpy())

        ax.ax_joint.set_xlim(min(data_range[0][0], -1), max(data_range[0][1], 1))
        ax.ax_joint.set_ylim(min(data_range[1][0], -1), max(data_range[1][1], 1))

        # Marginal histograms
        ax.plot_marginals(sns.histplot, kde=True, bins=num_bins, kde_kws = {'bw_adjust': bw_adjust})

        # Apply tight layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(file_path, dpi=300)

        # Close the figure
        plt.close()

def gradient_scatter_plot(data: pd.DataFrame, column_labels: List[str], color_label: str, settings: Dict, file_path: str) -> None:
    """
    Create a scatter plot of the column_labels columns of the data DataFrame. 
    The color of the markers is a gradient of the colormap given in settings and determined by the color_label column.
    The plot is saved to a file.

    Inputs
    ------

        data:         data with the trajectory projected on the 2D space
        column_labels:     labels of the CVs used to project the trajectory
        color_label:     label of the data used to color the markers
        settings:        dictionary with the settings of the plot
        file_path:       path where the figure will be saved
    """

    if settings.get('plot', True):

        cmap = settings.get('cmap')
        marker_size = settings.get('marker_size')
        alpha = settings.get('alpha')

        # Create a figure
        fig, ax = plt.subplots()

        # Create a scatter plot color-coded by the order
        ax.scatter(data[column_labels[0]], data[column_labels[1]], c=data[color_label], cmap=cmap,
                   s=marker_size, alpha=alpha, edgecolor=".2", linewidth=.5)

        # Add color bar
        norm = plt.Normalize(data[color_label].min(), data[color_label].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.set_label('Frame num.')

        # Set axis labels
        ax.set_xlabel(column_labels[0])
        ax.set_ylabel(column_labels[1])
        
        # Get range of the data
        data_range = get_ranges(data[[column_labels[0], column_labels[1]]].to_numpy())
        
        # Enforce CV limits, mix data ranges and -1, 1 limits
        ax.set_xlim(min(data_range[0][0], -1), max(data_range[0][1], 1))
        ax.set_ylim(min(data_range[1][0], -1), max(data_range[1][1], 1))
        
        # Apply tight layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(file_path, dpi=300)

        # Close the figure
        plt.close()

def get_ranges(X: np.ndarray, X_ref: Union[List[np.ndarray], None] = None) -> List:
    """
    Find the range of the data along each dimension.

    Inputs
    ------

        X:             array with the main data
        X_ref:         array with the reference data

    Returns
    -------

        ranges:       list with the range of the data along each dimension +/- 0.5% of the range
    """

    # Check dimension of the data
    if X.ndim == 1:
        data_dimension = 1
    else:
        data_dimension = X.shape[1]

    if data_dimension == 1:

        # Find the range of the main data
        data_range = (np.min(X), np.max(X))

        # If reference data is provided
        if X_ref is not None:

            # Update limits with the range of the reference data
            for X_ref_i in X_ref:

                min_x = np.min(X_ref_i)
                max_x = np.max(X_ref_i)

                if min_x < data_range[0]:
                    data_range = (min_x, data_range[1])

                if max_x > data_range[1]:
                    data_range = (data_range[0], max_x)

        # Define an offset as 5% of the range
        offset = 0.005*(data_range[1]-data_range[0])

        # Add the offset to the limits
        data_range = (data_range[0]-offset, data_range[1]+offset)

    else:

        data_range = []

        for i in range(data_dimension):
            
            # Find the range of the main data
            limit_i = (np.min(X[:, i]), np.max(X[:, i]))

            # If reference data is provided
            if X_ref is not None:

                # Update limits with the range of the reference data
                for X_ref_i in X_ref:

                    min_x = np.min(X_ref_i[:, i])
                    max_x = np.max(X_ref_i[:, i])

                    if min_x < limit_i[0]:
                        limit_i = (min_x, limit_i[1])

                    if max_x > limit_i[1]:
                        limit_i = (limit_i[0], max_x)

            # Define an offset as 5% of the range
            offset = 0.05*(limit_i[1]-limit_i[0])

            # Add the offset to the limits
            data_range.append((limit_i[0]-offset, limit_i[1]+offset))

    return data_range

def plot_clusters_size(cluster_labels: pd.Series, colors: List, output_folder: str):
    """
    Plot barplot with the number of members for each cluster.
    
    Inputs
    ------

        cluster_labels   : List with cluster ID for each frame
        colors           : List with the RGB colors to use for each cluster
        output_folder    : Path to the output folder
    """

    # Find settings
    font_size = 12
    tick_size = 10

    cluster_sizes = []
    clusters = np.sort(np.unique(cluster_labels))

    # Iterate over the clusters
    for cluster in clusters:

        # Count the number of frames in each cluster
        cluster_sizes.append(np.count_nonzero(cluster_labels == cluster))

    # Width of the bars
    width = 0.7

    # Number of clusters
    num_clusters = len(clusters)

    # Transfor the RGB values to hex
    colors = [rgb2hex(color) for color in colors]

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the barplot
    bp = ax.bar(clusters, cluster_sizes, width, color=colors, label="Cluster size")

    # add value on top of bars
    for rect in bp:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.05,
                '%d' % int(height),
                ha='center', va='bottom', fontsize=font_size)

    # Put labels and title
    plt.xlabel("Clusters", fontsize=font_size)
    plt.ylabel("Number of members", fontsize=font_size)
    plt.title("Distribution within clusters", fontsize=font_size)

    # Other formatting
    plt.xticks(np.array(clusters) + (width / 2), clusters)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_folder, 'clusters_size.png'), dpi=300, bbox_inches='tight')

    # Close the figure
    plt.close()

def generate_cmap(num_colors: int, base_colormap: str):
    """
    Generate a color map with num_colors colors from a base colormap.
    
    Inputs
    ------

        num_colors (int): The number of colors to generate.
        base_colormap (str): The name of the base colormap to use.

    Output
    ------
        
        cmap (ListedColormap): The generated color map.
    """
    
    # Extract the desired number of colors from the base cmap in RGBA format
    colors = generate_colors(num_colors, base_colormap)
    
    # Create a ListedColormap object using the extracted colors
    cmap = ListedColormap(colors)
    
    return cmap  

def generate_colors(num_colors: int, base_colormap: str) -> list:
    """
    Generate a list of colors from a base colormap.

    Inputs
    ------

        num_colors (int): The number of colors to generate.
        base_colormap (str): The name of the base colormap to use.
    
    Output
    ------

        colors (List): A list of colors in the RGB(A) format (0-1 range for each channel: (Red, Green, Blue, Alpha)).
    """

    # Get the base cmap
    base_cmap = plt.cm.get_cmap(base_colormap)
    
    # Extract the desired number of colors from the base cmap
    colors = base_cmap(np.linspace(0, 1, num_colors))
    
    return colors

def plot_data(y_data: Dict[str, np.array], x_data: Dict[str, np.array], title: str, y_label: str, x_label: str, figure_path: str):
    """
    Plot each array in y_data vs the corresponding array in x_data and save the figure to a file.
    
    Inputs
    ------
    
        y_data:     dictionary with the y data to plot
        x_data:     dictionary with the x values for each y data
        title:      title of the plot
        y_label:    label of the y axis
        x_label:    label of the x axis
        figure_path: path where the figure will be saved 
    """
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Increase the figure size
    fig.set_size_inches(10, 5)

    x_limits = []
    y_limits = []
    
    # For each column in data
    for key in y_data.keys():
        
        # Find the corresponding x data
        x_array = x_data.get(key)
        if x_array is None:
            raise ValueError(f"No x values provided for {key}")
        
        # Plot the data
        ax.plot(x_array, y_data[key], label=key, linewidth=1)
        
        # Get limits
        x_limits.append(np.min(x_array))
        x_limits.append(np.max(x_array))
        y_limits.append(np.min(y_data[key]))
        y_limits.append(np.max(y_data[key]))
    
    # Set axis limits
    ax.set_xlim(min(x_limits), max(x_limits))
    ax.set_ylim(0, max(y_limits)+max(y_limits)*0.05)
    
    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Set title
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Save figure
    fig.savefig(figure_path, dpi=300)
    
    # Close figure
    plt.close()

    return

def plot_sensitivity_results(results: Dict[str, np.array], 
                             modes: Literal['barh', 'violin', 'scatter'], 
                             output_folder: str):
    """
    Plot the sensitivity analysis results.

    Inputs
    ------

        results:             dictionary with the sensitivity analysis results
        modes:               list of modes to plot the sensitivity analysis results 
        output_folder:       path to the output folder where the figures will be saved
    """
    
    from mlcolvar.explain import plot_sensitivity
    
    # Plot sensitivity with different modes
    
    for mode in modes:
        fig,ax = plt.subplots(figsize=(5, 10))
        plot_sensitivity(results, mode=mode, per_class=False, max_features=20, ax=ax)
        ax.set_title(f'plot mode = {mode}')
        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, f'top_features_{mode}.png'), dpi=300)
        plt.close(fig)
    
    # Plot the histogram of the sensitivity values
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(results['sensitivity'], bins=40, kde=True, ax=ax, log_scale=[True, False])
    ax.set_title('Sensitivity histogram')
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, 'sensitivity_histogram.png'), dpi=300)
    plt.close(fig)
    
    return