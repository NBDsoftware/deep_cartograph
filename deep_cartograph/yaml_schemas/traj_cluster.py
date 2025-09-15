from pydantic import BaseModel
from typing import List, Union, Literal

class Figures(BaseModel):
    # Plot the Projected Clustered Trajectory
    plot: bool = True
    # Number of bins for the Kernel Density Estimation of the Projected Clustered Trajectory
    num_bins: int = 100
    # Bandwidth for the Kernel Density Estimation of the Projected Clustered Trajectory
    bandwidth: float = 0.25
    # Transparency of the points in the Projected Clustered Trajectory
    alpha: float = 0.8
    # Colormap for the Projected Clustered Trajectory
    cmap: str = "turbo"
    # Size of the markers in the Projected Clustered Trajectory
    marker_size: int = 5
    
class TrajClusterSchema(BaseModel):

    # Whether to run the clustering or not
    run: bool = True
    # Output mode for the clustering results
    output_structures: Literal['centroids', 'all', 'none'] = 'centroids'
    # Clustering algorithm to use
    algorithm: Literal["kmeans", "hdbscan", "hierarchical"] = "hierarchical"
    # Whether to search for the optimal number of clusters inside the search_interval or not (only for hierarchical and kmeans)
    opt_num_clusters: bool = True
    # Range of number of clusters to search for the optimal number of clusters (only for hierarchical and kmeans)
    search_interval: List[int] = [3, 10]
    # Number of clusters to use (only for hierarchical and kmeans and if opt_num_clusters is false)
    num_clusters: int = 10
    # Linkage criterion to use ('ward', 'single', 'average', 'complete') (only for hierarchical)
    linkage: str = "complete"
    # Number of times the k-means algorithm is run with different centroid seeds (only for kmeans)
    n_init: int = 20
    # Minimum number of samples in a group for that group to be considered a cluster; groupings smaller than this size will be left as noise (only for hdbscan)
    min_cluster_size: int = 5
    # A limit to the size of clusters returned by the "eom" cluster selection algorithm, no limit if None (only for hdbscan)
    max_cluster_size: Union[int, None] = None
    # Number of samples in a neighborhood for a point to be considered as a core point (only for hdbscan)
    min_samples: int = 3
    # A distance threshold. Clusters below this value will be merged (only for hdbscan)
    cluster_selection_epsilon: float = 0
    # The method used to select clusters from the condensed tree."eom" selects the most persistent cluster while “leaf” provides the most fine grained and homogeneous ones
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    # Settings for figures
    figures: Figures = Figures()

