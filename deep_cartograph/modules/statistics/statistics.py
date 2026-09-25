"""
Statistics helpers: clustering of samples (k-means, HDBSCAN, hierarchical)
and simple per-feature statistics used to filter features.
"""

# Import modules
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Literal, List
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set logger
logger = logging.getLogger(__name__)

# Clustering
def optimize_clustering(features: np.ndarray, settings: Dict):
    """
    Cluster the samples, choosing the number of clusters automatically.

    For kmeans and hierarchical, the data is clustered once for each number of clusters
    in the search interval, and the number with the best combined score* is kept.
    HDBSCAN already finds the number of clusters by itself, so it is run only once.

    * The score is the equal weight combination of three scores, each min-max
      normalized over the search interval:

        - Average silhouette score: how close each sample is to its own cluster compared
          to other clusters. Ranges from -1 to 1, higher is better.
        - Calinski-Harabasz score: ratio of between-cluster to within-cluster dispersion.
          Higher is better.
        - Davies-Bouldin score: average similarity of each cluster with its most similar
          cluster. Lower is better (so it is subtracted).

    Parameters
    ----------

    features : np.ndarray
        Matrix with the features of each sample (n_samples x n_features)
    settings : Dict
        Clustering settings. 'algorithm' must be 'kmeans', 'hierarchical' or 'hdbscan'.
        For kmeans and hierarchical, 'search_interval' gives the [min, max] number of
        clusters to try (default [2, 15]). See cluster_data() for the other keys.
        This dictionary is modified in place.

    Returns
    -------

    cluster_labels : np.ndarray
        Cluster assignment for each sample
    centroids : np.ndarray
        Centroids of the clusters
    """

    if settings['algorithm'] == 'kmeans' or settings['algorithm'] == 'hierarchical':

        search_interval = settings.get('search_interval', [2, 15])
        num_clusters = range(search_interval[0], search_interval[1]+1)
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        silhouette_scores = []
        results = []

        # Iterate over interval of clusters
        for N in num_clusters:

            # Set number of clusters to look for
            settings['num_clusters'] = N

            # Cluster data
            cluster_labels, centroids = cluster_data(features, settings)

            # Compute scores
            calinski_harabasz_scores.append(calinski_harabasz_score(features, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(features, cluster_labels))
            silhouette_scores.append(silhouette_score(features, cluster_labels))
            
            # Log
            logger.debug(f"Calinski-Harabasz score: {round(calinski_harabasz_scores[-1],3)}")
            logger.debug(f"Davies-Bouldin score: {round(davies_bouldin_scores[-1],3)}")
            logger.debug(f"Average silhouette score: {round(silhouette_scores[-1],3)}")
            
            # Save results
            results.append((cluster_labels, centroids))

        # Normalize scores
        calinski_harabasz_scores = (calinski_harabasz_scores - np.min(calinski_harabasz_scores)) / (np.max(calinski_harabasz_scores) - np.min(calinski_harabasz_scores))
        davies_bouldin_scores = (davies_bouldin_scores - np.min(davies_bouldin_scores)) / (np.max(davies_bouldin_scores) - np.min(davies_bouldin_scores))
        silhouette_scores = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))

        # Combine scores
        score = (np.array(calinski_harabasz_scores) - np.array(davies_bouldin_scores) + np.array(silhouette_scores)) / 3

        # Find best number of clusters
        best_N = num_clusters[np.argmax(score)]

        # Log
        logger.info(f"Best number of clusters: {best_N}")

        # Save best results
        cluster_labels, centroids = results[np.argmax(score)]
    
    elif settings['algorithm'] == 'hdbscan':

        # Cluster data
        cluster_labels, centroids = cluster_data(features, settings)
     
    if len(centroids) == 0:
        logger.warning("No clusters found using the provided settings. Try different settings or a different algorithm")

    return cluster_labels, centroids

def cluster_data(features: np.ndarray, 
                 settings: Dict, 
                 initial_centroids: np.ndarray = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster the samples using the algorithm and settings given in the settings dictionary.

    Missing settings are filled in with defaults (the dictionary is modified in place):
    algorithm='kmeans', num_clusters=10, n_init=10, min_cluster_size=10% of the samples,
    min_samples=0.1% of the samples (at least 1), cluster_selection_epsilon=0,
    linkage='complete', max_cluster_size=None, cluster_selection_method='eom'.

    Parameters
    ----------

    features : np.ndarray
        Matrix with the features of each sample (n_samples x n_features)
    settings : Dict
        Clustering settings. 'algorithm' can be 'kmeans', 'hdbscan' or 'hierarchical'.
    initial_centroids : np.ndarray, optional
        Initial centroids for k-means. If given, it sets the number of clusters
        (overrides num_clusters). Ignored by the other algorithms.

    Returns
    -------

    cluster_labels : np.ndarray
        Cluster assignment for each sample
    centroids : np.ndarray
        Centroids of the clusters

    Raises
    ------

    Exception
        If the algorithm is not implemented
    """

    # Set default values for clustering settings
    settings['algorithm'] = settings.get('algorithm', 'kmeans')
    settings['num_clusters'] = settings.get('num_clusters', 10)
    settings['n_init'] = settings.get('n_init', 10)
    settings['min_cluster_size'] = settings.get('min_cluster_size', int(0.1 * features.shape[0]))  # 10% of the number of samples
    settings['min_samples'] = settings.get('min_samples',  max(int(0.001 * features.shape[0]), 1)) # 0.1% of the number of samples, at least 1
    settings['cluster_selection_epsilon'] = settings.get('cluster_selection_epsilon', 0)
    settings['linkage'] = settings.get('linkage', 'complete')
    settings['max_cluster_size'] = settings.get('max_cluster_size')
    settings['cluster_selection_method'] = settings.get('cluster_selection_method', 'eom')

    if settings['algorithm'] == 'kmeans':
        cluster_labels, centroids = kmeans_clustering(features, settings['num_clusters'], settings['n_init'], initial_centroids)
    
    elif settings['algorithm'] == 'hdbscan':
        cluster_labels, centroids = hdbscan_clustering(features, settings['min_cluster_size'], settings['max_cluster_size'], settings['min_samples'], 
                                                       settings['cluster_selection_epsilon'], settings['cluster_selection_method'])
    
    elif settings['algorithm'] == 'hierarchical':
        cluster_labels, centroids = hierarchical_clustering(features, None, settings['num_clusters'], settings['linkage'])
                
    else:
        raise Exception(f"clustering algorithm {settings['algorithm']} not implemented")

    return cluster_labels, centroids

def kmeans_clustering(feature_matrix: np.ndarray,
                      num_clusters: int, 
                      n_init: int, 
                      initial_centroids: np.ndarray = None
                      ) -> np.ndarray:

    """
    Cluster the frames of the simulation with k-means, using the Euclidean distance between features.

    Parameters
    ----------

    feature_matrix : np.ndarray
        Matrix with the features of each frame (n_frames x n_features)
    num_clusters : int
        Number of clusters to find
    n_init : int
        Number of k-means runs with different centroid seeds. The best result is kept.
    initial_centroids : np.ndarray, optional
        Initial centroids. If given, it sets the number of clusters (overrides num_clusters).
        If None, 'k-means++' initialization is used.

    Returns
    -------

    clusters : np.ndarray
        Cluster assignment for each frame
    centroids : np.ndarray
        Centroids of the clusters
    """

    # Log
    logger.debug("Clustering frames with kmeans...")

    if initial_centroids is None:
        initial_centroids = 'k-means++'
    else:
        num_clusters = initial_centroids.shape[0]

    logger.debug("Number of clusters: {}".format(num_clusters))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, init=initial_centroids, n_init=n_init)

    # Cluster frames
    clusters = kmeans.fit_predict(feature_matrix)

    # Find centroids
    centroids = kmeans.cluster_centers_

    return clusters, centroids

def hdbscan_clustering(feature_matrix: np.array, 
                       min_cluster_size: Optional[int] = 5, 
                       max_cluster_size: Optional[int] = None, 
                       min_samples: Optional[int] = None, 
                       cluster_selection_epsilon: Optional[float] = None, 
                       cluster_selection_method: Literal["eom", "leaf"] = "eom"
                       ) -> Tuple[np.array, np.array]:
    """
    Cluster the frames of the simulation with HDBSCAN, using the Euclidean distance between features.

    HDBSCAN is a density-based method: it finds the number of clusters by itself and
    labels frames that don't belong to any cluster as noise (label -1). The number of
    parallel jobs is taken from the SLURM environment variables, if set.

    Parameters
    ----------

    feature_matrix : np.array
        Matrix with the features of each frame (n_frames x n_features)
    min_cluster_size : int, optional
        Minimum number of samples in a group for that group to be considered a cluster.
        Smaller groups are left as noise. Default is 5.
    max_cluster_size : int, optional
        Maximum size of the clusters returned by the "eom" selection method. No limit if None.
    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered a core point.
        If None, it is set to min_cluster_size.
    cluster_selection_epsilon : float, optional
        Distance threshold. Clusters closer than this value are merged.
    cluster_selection_method : {"eom", "leaf"}, optional
        Method used to select clusters from the condensed tree. Default is "eom".

    Returns
    -------

    clusters : np.array
        Cluster assignment for each frame (-1 means noise)
    centroids : np.array
        Estimated centroid of each cluster (noise excluded)
    """

    # Find number of CPU cores requested in SLURM
    n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    # Find number of tasks requested in SLURM
    n_tasks = int(os.environ.get('SLURM_NTASKS', 1))

    # Find number of concurrent processes to use with joblib
    n_jobs = n_cores * n_tasks

    # Log
    logger.debug("Clustering frames with HDBSCAN...")
    logger.debug("min_cluster_size: {}".format(min_cluster_size))
    logger.debug("min_samples: {}".format(min_samples))
    logger.debug("cluster_selection_epsilon: {}".format(cluster_selection_epsilon))
    logger.debug("max_cluster_size: {}".format(max_cluster_size))
    logger.debug("n_jobs: {}".format(n_jobs))

    # If n_jobs = 1, use None
    if n_jobs == 1:
        n_jobs = None

    # Initialize HDBSCAN object
    hdb = HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples, n_jobs = n_jobs, store_centers = "centroid",
                  cluster_selection_epsilon = cluster_selection_epsilon, max_cluster_size = max_cluster_size,
                  cluster_selection_method = cluster_selection_method, allow_single_cluster=False)
    
    # "pip install hdbscan" version 
    # hdb = HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples,
    #               cluster_selection_epsilon = cluster_selection_epsilon, max_cluster_size = max_cluster_size)
    
    # Cluster samples
    hdb.fit(feature_matrix)
    
    # Find labels
    clusters = hdb.labels_

    # Find unique clusters
    unique_clusters = np.unique(clusters)

    # Log number of clusters
    logger.debug(f"Number of clusters (including noise cluster): {len(unique_clusters)}")

    # Eliminate -1 (noise) from unique clusters
    unique_clusters = unique_clusters[unique_clusters != -1]

    # "pip install hdbscan" version
    # Initialize centroids
    # centroids = np.zeros((len(unique_clusters), feature_matrix.shape[1]))
    # Iterate over unique clusters
    # for i in unique_clusters:
            # Find centroid
            # centroids[i,:] = hdb.weighted_cluster_centroid(i)

    centroids = hdb.centroids_

    return clusters, centroids

def hierarchical_clustering(feature_matrix: np.array, 
                            cutoff: Optional[float], 
                            num_clusters: Optional[int] = None, 
                            linkage: Literal["ward", "complete", "average", "single"] = 'complete'
                            ) -> Tuple[np.array, np.array]:
    """
    Cluster points with agglomerative (hierarchical) clustering, using the Euclidean distance between features.

    Give either cutoff or num_clusters, not both.

    Parameters
    ----------

    feature_matrix : np.array
        Matrix with the features of each point (e.g. descriptors of a frame of an MD simulation)
    cutoff : Optional[float]
        Distance threshold above which clusters are not merged
    num_clusters : int, optional
        Number of clusters to find
    linkage : {"ward", "complete", "average", "single"}, optional
        Linkage criterion used to merge clusters. Default is 'complete'.

    Returns
    -------

    clusters : np.array
        Cluster assignment for each point
    centroids : np.array
        Centroid of each cluster, computed as the mean of the features of its points

    Raises
    ------

    Exception
        If both or neither of cutoff and num_clusters are given
    """

    # Log
    logger.debug("Clustering frames with Hierarchical...")
    logger.debug("cutoff: {}".format(cutoff))
    logger.debug("num_clusters: {}".format(num_clusters))
    logger.debug("linkage: {}".format(linkage))

    # Check if cutoff or num_clusters is provided
    if cutoff is None and num_clusters is None:
        raise Exception("  Either cutoff or num_clusters must be provided")
    elif cutoff is not None and num_clusters is not None:
        raise Exception("  Only one of cutoff or num_clusters must be provided")
    
    # Initialize hierarchical clustering object
    hc = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=cutoff, linkage=linkage)

    # Cluster frames
    clustered_points = hc.fit_predict(feature_matrix)

    # Initialize centroids
    centroids = np.zeros((len(np.unique(clustered_points)), feature_matrix.shape[1]))

    # Iterate over clusters
    for i in range(len(np.unique(clustered_points))):
            
            # Find frames in cluster
            frames = np.where(clustered_points == i)[0]
    
            # Find centroid
            centroids[i,:] = np.mean(feature_matrix[frames,:], axis=0)

    return clustered_points, centroids

def find_centroids(data: pd.DataFrame,
                   centroids: np.array, 
                   clustering_features: list
                   ) -> pd.DataFrame:
    """
    Find the closest sample to each centroid and mark it in a new boolean column named 'centroid'.

    The input dataframe is modified in place. Exits the program if the centroids
    and the clustering features have different dimensions.

    Parameters
    ----------

    data : pd.DataFrame
        Data with the features of the samples and possibly other columns
    centroids : np.array
        Estimated centroid of each cluster
    clustering_features : list
        Column names in data that correspond to the features used for clustering

    Returns
    -------

    data : pd.DataFrame
        Input dataframe with the extra 'centroid' column, or an empty dataframe if there are no centroids
    """

    # Make sure there are centroids
    if len(centroids) == 0:
        logger.warning("No centroids found")
        return pd.DataFrame()
    
    # Make sure the centroids have the same dimension as the clustering features
    if len(centroids[0]) != len(clustering_features):
        logger.error("  The dimension of the centroids is not the same as the dimension of the used features for clustering.\n")
        logger.error(f"  Centroids dimension: {len(centroids[0])}, clustering features dimension: {len(clustering_features)}\n")
        sys.exit(1)

    # Add a centroid column to the dataframe marking the samples that are centroids
    data['centroid'] = False

    # Find the closest sample to each centroid
    for centroid in centroids:

        # Find closest sample to centroid
        distances = np.linalg.norm(data.loc[:, clustering_features].values - centroid, axis=1)
        closest_sample_index = np.argmin(distances)

        # Mark the closest sample as a centroid
        data.at[closest_sample_index, 'centroid'] = True

    return data

# Feature statistics
def difference_filter(features_df: pd.DataFrame) -> List[bool]:
    """
    Check if each feature changes enough across samples, using a fixed threshold per feature type.

    The feature type is read from the prefix of the name (the part before the first '-').
    The range of values (max - min) of each feature is compared with a threshold:

        - 'sin'/'cos' features: the angle is rebuilt from both components and its range
          must be at least pi/8 rad (22.5 degrees).
        - 'tor' features (torsion angles): range of at least pi/8 rad.
        - 'coord' features: the x, y, z coordinates of each atom are taken together and
          the largest distance between any two samples must be at least 0.2 nm.
        - Other features: range of at least 0.2 (e.g. 0.2 nm for distances).

    Parameters
    ----------

    features_df : pd.DataFrame
        DataFrame with the time series of the features (one column per feature)

    Returns
    -------

    List[bool]
        One value per column, in the same order. True if the feature changes enough.
        Empty list if the dataframe is empty.
    """
    from scipy.spatial import distance_matrix
    
    angle_threshold = np.pi / 8  # 22.5 degrees
    distance_threshold = 0.2     # 0.2 nm or 2 Angstroms
    
    # Check the dataframe is not empty
    if features_df.empty:
        logger.warning("Features dataframe is empty. Returning empty list.")
        return []
    
    atoms_touched = set()  # Keep track of atoms already processed for coordinate features
    
    feature_names = list(features_df.columns)
    results = pd.DataFrame(columns=['name', 'above_threshold'])
    results['name'] = feature_names
    for name in feature_names:
        
        feature_parts = name.split('-')
        if not len(feature_parts) > 1:
            logger.error(f"Feature name {name} does not contain a '-' character. Skipping this feature.")
            continue
        feature_type = feature_parts[0]
        
        if 'sin' == feature_type:
            # Deal here with sin/cosine features
            cosine_name = name.replace('sin', 'cos')
            if cosine_name in feature_names:
                # Get the samples for the sine and cosine components
                sine_timeseries = features_df[name].to_numpy()
                cosine_timeseries = features_df[cosine_name].to_numpy()
                
                # Compute the angle for each sample between 0 and 2pi
                angles = np.arctan2(sine_timeseries, cosine_timeseries) + np.pi  # Shift to [0, 2pi]
                
                # Get the maximum difference between all samples in radians
                delta = np.abs(np.max(angles) - np.min(angles))
            else:
                logger.warning(f"Cosine component {cosine_name} not found for sine component {name}. Check the compute features step. Skipping this feature.")
                delta = 10  # Large difference to skip this feature

            # Check if the angle difference is big enough
            results.loc[results['name'] == name, 'above_threshold'] = delta >= angle_threshold
            results.loc[results['name'] == cosine_name, 'above_threshold'] = delta >= angle_threshold

        elif 'cos' == feature_type:
            continue  # Skip cosine components, they are handled with the sine components 
        
        elif 'tor' == feature_type:
            # Deal here with torsion angles
            torsion_timeseries = features_df[name].to_numpy()
            delta = np.max(torsion_timeseries) - np.min(torsion_timeseries)
            results.loc[results['name'] == name, 'above_threshold'] = delta >= angle_threshold
        elif 'coord' == feature_type:
            # Deal here with coordinates
            coordinate_definition = feature_parts[1].split('.')
            atom_definition = coordinate_definition[0]
            
            # Do this once for each atom
            if atom_definition not in atoms_touched:
                atoms_touched.add(atom_definition)
                
                # Get the names for the x, y, z coordinate features of the atom
                x_name = f"coord-{atom_definition}.x"
                y_name = f"coord-{atom_definition}.y"
                z_name = f"coord-{atom_definition}.z"
                
                # For each axis, get the time series or add a zero array if the axis is not present
                x_timeseries = features_df[x_name].to_numpy() if x_name in feature_names else np.zeros(features_df.shape[0])
                y_timeseries = features_df[y_name].to_numpy() if y_name in feature_names else np.zeros(features_df.shape[0])
                z_timeseries = features_df[z_name].to_numpy() if z_name in feature_names else np.zeros(features_df.shape[0])
                
                # Construct the matrix of (x, y, z) coordinates for each sample
                coordinates = np.vstack((x_timeseries, y_timeseries, z_timeseries)).T
                
                # Compute the pairwise distance matrix between all samples
                dist_matrix = distance_matrix(coordinates, coordinates)
                
                # Get the maximum distance between all samples
                delta = np.max(dist_matrix)
                
                # Check if the distance is big enough
                is_big_enough = delta >= distance_threshold
                results.loc[results['name'] == x_name, 'above_threshold'] = is_big_enough
                results.loc[results['name'] == y_name, 'above_threshold'] = is_big_enough
                results.loc[results['name'] == z_name, 'above_threshold'] = is_big_enough
                 
        else:
            # Deal with other features
            feature_timeseries = features_df[name].to_numpy()
            delta = np.abs(np.max(feature_timeseries) - np.min(feature_timeseries))
            results.loc[results['name'] == name, 'above_threshold'] = delta >= distance_threshold 
            
    return results['above_threshold'].tolist()

def min_value_filter(features_df: pd.DataFrame, threshold: float) -> List[bool]:
    """
    Check if the minimum value of each feature across samples is at or below a threshold.

    Parameters
    ----------

    features_df : pd.DataFrame
        DataFrame with the time series of the features
    threshold : float
        Threshold for the minimum value

    Returns
    -------

    results : List[bool]
        One value per feature. True if its minimum value is at or below the threshold.
    """
    
    feature_names = list(features_df.columns)
    results = []
    for name in feature_names:
        min_value = np.min(features_df[name].to_numpy())
        results.append(min_value <= threshold)
    return results
    

def shannon_entropy(features_df: pd.DataFrame) -> List[float]:
    """
    Compute the Shannon entropy of the distribution of each feature.

    Entropy measures how spread out a distribution is: it is higher when values are
    spread over many bins and lower when they are concentrated in a few. Here it is
    computed on a 100-bin histogram of each feature (spanning its own min to max), in bits.

    Parameters
    ----------

    features_df : pd.DataFrame
        DataFrame with the time series of the features

    Returns
    -------

    feature_entropies : List[float]
        Shannon entropy of each feature, rounded to 3 decimals
    """

    from scipy.stats import entropy

    # Iterate over the features
    feature_entropies = []
    feature_names = list(features_df.columns)
    for name in feature_names:
        
        # Compute the histogram of the feature
        hist, bin_edges = np.histogram(features_df[name].to_numpy(), bins=100, density=True)
        prob_distribution = hist * np.diff(bin_edges)

        # Compute and append the entropy to the list
        feature_entropies.append(round(entropy(prob_distribution, base=2), 3))
        
    return feature_entropies

def standard_deviation(features_df: pd.DataFrame) -> List[float]:
    """
    Compute the standard deviation of each feature.

    Parameters
    ----------

    features_df : pd.DataFrame
        DataFrame with the time series of the features

    Returns
    -------

    feature_stds : List[float]
        Standard deviation of each feature, rounded to 3 decimals
    """

    # Iterate over the features
    feature_stds = []
    feature_names = list(features_df.columns)
    for name in feature_names:

        # Compute and append the std to the list
        feature_stds.append(round(np.std(features_df[name].to_numpy()), 3))

    return feature_stds

def dip_test(features_df: pd.DataFrame) -> List[float]:
    """
    Compute the p-value of Hartigan's dip test for each feature.

    Hartigan's dip test checks whether a distribution has a single peak (unimodal).
    A small p-value is evidence that the feature has more than one peak (multimodal).

    Parameters
    ----------

    features_df : pd.DataFrame
        DataFrame with the time series of the features

    Returns
    -------

    hdt_pvalues : List[float]
        p-value of Hartigan's dip test for each feature
    """
    
    from diptest import diptest
    
    # Iterate over the features
    hdt_pvalues = []
    feature_names = list(features_df.columns)
    for name in feature_names:
        
        # Compute the p-value of the Hartigan Dip test
        hdt_pvalue = diptest(np.array(features_df[name].to_numpy()))[1]

        # Append the p-value to the list
        hdt_pvalues.append(hdt_pvalue)

    # Return the list of p-values
    return hdt_pvalues