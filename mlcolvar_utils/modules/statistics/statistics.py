# Import modules
import os
import logging
import numpy as np
from typing import List, Dict
from sklearn.decomposition import PCA
# from hdbscan import HDBSCAN - pip install hdbscan
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set logger
logger = logging.getLogger(__name__)

# Define functions
def optimize_clustering(features: np.ndarray, settings: Dict):
    """
    Optimize the hyper-parameters of the clustering algorithm. For kmeans and hierarchical, the optimization 
    is done computing a combined score* for each number of clusters and selecting the best scoring number of clusters.

    HDBSCAN already has a built-in mechanism to find the best number of clusters.

    * The score is the equal weight max-min normalized combination of:

        - Average silhouette score 
        - Calinski-Harabasz score
        - Davies-Bouldin score
        
    Average silhouette score: a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
    The silhouette ranges from -1 to 1, the higher the value, the better the clustering.

    Calinski-Harabasz score: a measure of how dense the clusters are and how well separated they are from each other. 
    The score is defined as the ratio of the sum of between-clusters dispersion and of within-cluster dispersion. 
    The higher the value, the better the clustering.

    Davies-Bouldin score: defined as the average similarity measure of each cluster with its most similar cluster, 
    where similarity is the ratio of within-cluster distances to between-cluster distances. The lower the value,
    the better the clustering.

    Inputs
    ------

        features:          matrix with the features of each sample
        settings:          dictionary with the settings for the clustering
    
    Outputs
    -------

        cluster_labels: array with the cluster assignment for each sample
        centroids:      array with the centroids of the clusters
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
            logger.debug(f"Calinski-Harabasz score: {calinski_harabasz_scores[-1]}")
            logger.debug(f"Davies-Bouldin score: {davies_bouldin_scores[-1]}")
            logger.debug(f"Average silhouette score: {silhouette_scores[-1]}")
            
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
    
    return cluster_labels, centroids

def cluster_data(features: np.ndarray, settings: Dict, initial_centroids: np.ndarray = None) -> np.ndarray:
    """
    Cluster the data in features using the clustering settings provided in the settings dictionary.

    Inputs
    ------

        features:          matrix with the features of each sample
        settings:          dictionary with the settings for the clustering
        initial_centroids: array with the initial centroids for the k-means algorithm (supersedes num_clusters)
    
    Outputs
    -------

        cluster_labels:  array with the cluster assignment for each sample
        centroids:       array with the centroids of the clusters
    """

    # Set default values for clustering settings
    settings['algorithm'] = settings.get('algorithm', 'kmeans')
    settings['num_clusters'] = settings.get('num_clusters', 10)
    settings['n_init'] = settings.get('n_init', 10)
    settings['min_cluster_size'] = settings.get('min_cluster_size', int(0.1 * features.shape[0]))  # 10% of the number of samples
    settings['min_samples'] = settings.get('min_samples',  max(int(0.001 * features.shape[0]), 1)) # 0.1% of the number of samples, at least 1
    settings['cluster_selection_epsilon'] = settings.get('cluster_selection_epsilon', 0)
    settings['linkage'] = settings.get('linkage', 'complete')

    if settings['algorithm'] == 'kmeans':
        cluster_labels, centroids = kmeans_clustering(features, settings['num_clusters'], settings['n_init'], initial_centroids)
    
    elif settings['algorithm'] == 'hdbscan':
        cluster_labels, centroids = hdbscan_clustering(features, settings['min_cluster_size'], settings['min_samples'], settings['cluster_selection_epsilon'])
    
    elif settings['algorithm'] == 'hierarchical':
        cluster_labels, centroids = hierarchical_clustering(features, None, settings['num_clusters'], settings['linkage'])
                
    else:
        raise Exception(f"clustering algorithm {settings['algorithm']} not implemented")

    return cluster_labels, centroids

def kmeans_clustering(feature_matrix: np.ndarray, num_clusters: int, n_init: int, initial_centroids: np.ndarray = None) -> np.ndarray:

    """
    Cluster the frames of the simulation based on the euclidian distance between features. The clustering is performed
    using the k-means algorithm.

    Inputs
    ------

        feature_matrix:    matrix with the features of each frame of the simulation
        num_clusters:      number of clusters to be used in the k-means algorithm
        n_init:            number of times the k-means algorithm will be run with different centroid seeds, the best result will be kept
        initial_centroids: array with the initial centroids for the k-means algorithm (supersedes num_clusters)
    
    Outputs
    -------

        clusters:    array with the cluster assignment for each frame of the simulation
        centroids:   array with the centroids of the clusters
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

def hdbscan_clustering(feature_matrix, min_cluster_size, min_samples, cluster_selection_epsilon):
    """
    Cluster the frames of the simulation based on the euclidian distance between features. The clustering is performed
    using the HDBSCAN algorithm.

    Inputs
    ------

        feature_matrix      (numpy array): matrix with the features of each frame of the simulation
        min_cluster_size            (int): minimum number of samples in a group for that group to be considered a cluster; 
                                           groupings smaller than this size will be left as noise 
        min_samples                 (int): number of samples in a neighborhood for a point to be considered as a core point
        cluster_selection_epsilon (float):  a distance threshold. Clusters below this value will be merged.
    
    Outputs
    -------

        clusters (numpy array): array with the cluster assignment for each frame of the simulation
        centroids (numpy array): array with the estimated centroids for each cluster
    """

    # Find max cluster size as 50% of the number of samples
    max_cluster_size = int(0.5 * feature_matrix.shape[0])

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
                  cluster_selection_epsilon = cluster_selection_epsilon, max_cluster_size = max_cluster_size)
    
    # pip install hdbscan version
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

    # pip install hdbscan version
    # Initialize centroids
    # centroids = np.zeros((len(unique_clusters), feature_matrix.shape[1]))
    # Iterate over unique clusters
    # for i in unique_clusters:
            # Find centroid
            # centroids[i,:] = hdb.weighted_cluster_centroid(i)

    centroids = hdb.centroids_

    return clusters, centroids

def hierarchical_clustering(feature_matrix, cutoff, num_clusters = None, linkage = 'complete'):
    """
    Cluster points based on the euclidian distance between features. The clustering is performed
    using the hierarchical clustering algorithm.

    Inputs
    ------

        feature_matrix (numpy array): matrix with the features of each point (e.g. descriptors of a frame of an MD simulation)
        cutoff               (float): cutoff distance for the clustering algorithm
        num_clusters           (int): number of clusters to be found.
        linkage                (str): linkage criterion to be used in the clustering algorithm

    Outputs
    -------

        clusters (numpy array): array with the cluster assignment for each pointt
        centroids (numpy array): array with the estimated centroid for each cluster as the mean value of the features of the points in the cluster
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