"""AMINO: generating a minimally redundant set of order parameters through
clustering of mutual information based distances. Method by Ravindra, Smith,
and Tiwary. Code maintained by Ravindra and Smith.

This is the parallelized kernel density estimation version.

Read and cite the following when using this method:
https://pubs.rsc.org/--/content/articlehtml/2020/me/c9me00115h
"""
import os
import sys
import copy
import time
import numpy as np
import logging
import dask
import dask.multiprocessing
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from dask.diagnostics import ProgressBar
dask.config.set(scheduler='processes')

# Local imports
from deep_cartograph.modules.common import common

# Set up logging
logger = logging.getLogger(__name__)

class OrderParameter:
    """Order Parameter (OP) class - stores OP name and trajectory
    
    Attributes
    ----------
    name : str
        Name of OP.
        
    traj : list of floats or np.array or None (if we just need the names for clustering and we use precomputed distances)
        Trajectory of OP values. This will be normalized to have std = 1.
        
    """

    # name should be unique to the Order Parameter being defined
    def __init__(self, name, traj = None):

        self.name = name

        if traj is not None:
            self.traj = np.array(traj, dtype = np.float32).reshape([-1,1])/np.std(traj)
        else:
            self.traj = None

    def __eq__(self, other):
        return self.name == other.name

    # This is needed for the sets construction used in clustering
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)


class Memoizer:
    """Memoizes distance computation between OP's to prevent re-calculations.
    
    Attributes
    ----------
    bins : int
        Number of values along each axis for the joint probability. 
        The probability will be a bins x bins grid.
        
    bandwidth : float
        Bandwidth parameter for kernel denensity estimation.
        
    kernel : str
        Kernel name for kernel density estimation.
        
    """

    def __init__(self, bins, bandwidth, kernel, weights=None):
        self.memo = {}
        self.bins = bins
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights

    # Binning two OP's in 2D space
    def d2_bin(self, x, y):
        """ Calculate a joint probability distribution for two trajectories.
        
        Parameters
        ----------
        x : np.array
            Trajectory of first OP.
            
        y : np.array
            Trajectory of second OP.
            
        Returns
        -------
        p : np.array
            self.bins by self.bins array of joint probabilities from KDE.
            
        """
        
        KD = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel)
        KD.fit(np.column_stack((x,y)), sample_weight=self.weights)
        grid1 = np.linspace(np.min(x),np.max(x),self.bins)
        grid2 = np.linspace(np.min(y),np.max(y),self.bins)
        mesh = np.meshgrid(grid1,grid2)
        data = np.column_stack((mesh[0].reshape(-1,1),mesh[1].reshape(-1,1)))
        samp = KD.score_samples(data)
        samp = samp.reshape(self.bins,self.bins)
        p = np.exp(samp)/np.sum(np.exp(samp))

        return p

    def distance(self, OP1, OP2):
        """Returns the mutual information-based distance between two OPs ()
        Calls distance calculation in parallel or returns saved values.
        
        Parameters
        ----------
        OP1 : OrderParameter
            The first order parameter for distance calculation.
            
        OP2 : OrderParameter
            The second order parameter for distance calculation.
            
        Returns
        -------
        output : float
            The mutual information distance.
            
        label (index1 or False) : str or bool
            The label for saving the distance in the memoizer.
            The name of the OP pair if not memoized or False if memoized.
            
        """

        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1)
        if memo_val == None: 
            memo_val = self.memo.get(index2)
        if memo_val != None:
            return memo_val, False

        x = OP1.traj
        y = OP2.traj
        
        output = dask.delayed(self.dist_calc)(x,y)

        return output, index1

    def dist_calc(self, x, y):
        """Calculates the distance between two trajectories.
        This is called when distances are not memoized.
        
        Parameters
        ----------
        x : np.array
            First trajectory.
            
        y : np.array
            Second trajectory.
            
        Returns
        -------
        output : float
            Calculated mutual information-based distance. D(X;Y) = 1 - I(X;Y)/H(X;Y)
        
        """
        p_xy = self.d2_bin(x, y)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        p_x_times_p_y = np.tensordot(p_x, p_y, axes = 0)
        info = np.sum(p_xy * np.ma.log(np.ma.divide(p_xy, p_x_times_p_y))) # Mutual information I(X;Y)
        entropy = np.sum(-1 * p_xy * np.ma.log(p_xy)) # Joint entropy H(X;Y)

        output = max(0.0, (1 - (info / entropy))) # Distance D(X;Y) = 1 - I(X;Y)/H(X;Y)
        return output
                
    def dist_matrix(self, group1, group2):
        """Calculates all distances between two groups of OPs. 
        
        Parameters
        ----------
        group1 : list of OrderParameters
            First group of OPs.
            
        group2 : list of OrderParameters
            Second group of OPs.
            
        Returns
        -------
        tmps : list of lists of floats
            Matrix containing distances between the groups.
            
        """
        
        tmps = []
        for i in group2:
            tmps.append([])
            for j in group1:
                mi, label = self.distance(i, j)
                tmps[-1].append(mi)
        return tmps
        

# Dissimilarity Matrix (DM) construction
class DissimilarityMatrix:
    """Matrix containing distances for initial centroid determination.
    
    Attributes
    ----------
    size : int
        Maximum number of OPs contained.
    
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    """

    def __init__(self, size, mut):
        self.size = size
        self.matrix = [[] for i in range(size)]
        self.mut = mut
        self.OPs = []
        
    def add_OP(self, OP):
        """Adds OPs to the matrix if they should be added.
        OPs should be added when there are fewer OPs than self.size
        or if they can increase the geometric mean of distances
        by being swapped with a different OP.
        
        Parameters
        ----------
        OP : OrderParameter
            OP to potentially be added.
            
        Returns
        -------
        None
            self.matrix is updated directly without a return.
            
        """
        
        if len(self.OPs) == self.size: # matrix is full, check for swaps
            mut_info = []
            existing = []
            for i in range(len(self.OPs)):
                mi, label = self.mut.distance(self.OPs[i], OP)
                mut_info.append(mi)
                product = 1
                for j in range(len(self.OPs)):
                    if not i == j:
                        product = product * self.matrix[i][j]
                existing.append(product)
            update = False
            difference = None
            for i in range(len(self.OPs)):
                candidate_info = 1
                for j in range(len(self.OPs)):
                    if not i == j:
                        candidate_info = candidate_info * mut_info[j]
                if candidate_info > existing[i]:
                    update = True
                    if difference == None:
                        difference = candidate_info - existing[i]
                        old_OP = i
                    else:
                        if (candidate_info - existing[i]) > difference:
                            difference = candidate_info - existing[i]
                            old_OP = i
            if update == True: # swapping out an OP
                mi, label = self.mut.distance(OP, OP)
                mut_info[old_OP] = mi
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else: # adding an OP when there are fewer than self.size
            distances = []
            for i in range(len(self.OPs)):
                mi,label = self.mut.distance(OP, self.OPs[i])
                distances.append(mi)
            for i in range(len(self.OPs)):
                mut_info = distances[i]
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            mi, label = self.mut.distance(OP, OP)
            #mi = dask.compute(mi)
            self.matrix[len(self.OPs)].append(mi)
            self.OPs.append(OP)
           
            
def distortion(centers, ops, mut):
    """Computes the distortion between a set of centroids and OPs.
    When multiple centroids are used, the minimum distortion grouping
    will be used to calculate the total distortion.
    
    Parameters
    ----------
    centers : list of OrderParameters
        Cluster centroids.
        
    ops : list of OrderParameters
        All OPs belonging to the centroid's clusters.
        
    Returns
    -------
    float
        Minimum total distortion given the centroids and OPs.
    
    """
    
    tmps = mut.dist_matrix(centers, ops)
    min_vals = np.min(tmps,axis=1)
    dis = np.sum(min_vals**2)
    return 1 + (dis ** (0.5))

def grouping(centers, ops, mut):
    """Assigns OPs to minimum distortion clusters.
    
    Parameters
    ----------
    centers : list of OrderParameters
        Cluster centroids.
        
    ops : list of OrderParameters
        All OPs to be assigned to clusters.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    groups : list of lists of OrderParameters
        One list for each centroid containing the associated OPs.
    
    """
    
    groups = [[] for i in range(len(centers))]
    tmps = mut.dist_matrix(centers, ops)  
    assignment = np.argmin(tmps,axis=1)
     
    for i in range(len(assignment)):
        groups[assignment[i]].append(ops[i])
    return groups

def group_evaluation(ops, mut):
    """Calculates the centroid minimizing distortion for a set of OPs.
    
    Parameters
    ----------
    ops : list of OrderParameters
        Set of OPs for centroid calculation.
    
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    center : OrderParameter
        The OP that is the minimum distortion centroid.
        
    """

    # If there is only one OP, it is the centroid
    if len(ops) == 1:
        return ops[0]
    
    # If there are no OPs, return None
    if len(ops) == 0:
        return None
    
    center = ops[0]
    min_distortion = distortion([ops[0]], ops, mut)
    for i in ops:
        tmp = distortion([i], ops, mut)
        if tmp < min_distortion:
            center = i
            min_distortion = tmp
    return center

def read_matrix(input_distance_matrix, input_op_labels, mut):
    """ Reads in a pre-calculated distance matrix and corresponding OP labels.
    The distance matrix is stored in the mut.memo dictionary with the corresponding
    OP labels as keys.
    
    Parameters
    ----------
    
    input_distance_matrix : numpy array
        The distance matrix.
    
    input_op_labels : list of strings
        The labels corresponding to the order parameters in the distance matrix.
    
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
    
    Returns
    -------
    None
        The distance matrix is stored in mut.memo.
    
    """

    # Iterate through the the OP labels 
    for i in range(len(input_op_labels)):

        # Iterate through all the possible combinations of OP labels
        for j in range(i, len(input_op_labels)):

            index1 = input_op_labels[i] + " " + input_op_labels[j]
            index2 = input_op_labels[j] + " " + input_op_labels[i]

            # If neither index is in the memo, add the distance to the memo
            if index1 not in mut.memo and index2 not in mut.memo:
                mut.memo[index1] = input_distance_matrix[i][j]
            
def compute_matrix(ops, mut):
    """Calculates all the OP distances in parallel.
    Used before any of the clustering to maximize the
    number of distances calculated at once.
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    None
        Stores all values in mut.memo and does not return a value.
        
    """
    
    index_mat = np.ones((len(ops),len(ops)))
    pairs = np.argwhere(np.triu(index_mat)==1)
    distances = []
    labels = []

    for pair in pairs:
        mi, label = mut.distance(ops[pair[0]], ops[pair[1]])
        distances.append(mi)
        labels.append(label)
    with ProgressBar():
        distances = dask.compute(*distances)

    for i in range(len(labels)):
        mut.memo[labels[i]] = distances[i]
        
def get_matrix(ops, mut):
    """Get a matrix containing the distance between all OPs.
    This will most commonly be used after clustering is completed
    to observe the distances used for clustering.
    
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    dist_mat : np.array
        Distances for all pairs of OPs.
        
    """
    
    dist_mat = np.zeros((len(ops),len(ops)))
    pairs = np.argwhere(dist_mat == 0)
    
    

    for pair in pairs:
        mi, label = mut.distance(ops[pair[0]], ops[pair[1]])
        dist_mat[pair[0],pair[1]] = mi
        
    return dist_mat

def cluster(ops, seeds, mut):
    """Clusters OPs starting with centroids from a DissimilarityMatrix.
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    seeds : list of OrderParameters
        Starting centroids from a DissimilarityMatrix.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    centers : list of OPs
        Final centroids after clustering.

    empty_cluster : bool
        True if there is an empty cluster, False otherwise.
        
    """
    empty_cluster = False

    old_centers = []
    centers = copy.deepcopy(seeds)

    while (set(centers) != set(old_centers)):

        old_centers = copy.deepcopy(centers)
        centers = []
        groups = grouping(old_centers, ops, mut)
            
        for i in range(len(groups)):

            # Check for empty clusters
            if len(groups[i]) == 0:
                logger.error(f'Empty cluster found. This is likely due to a small size of OPs batch ({len(ops)}) compared to the maximum number of independent OPs that can be found ({len(seeds)}).')

                # Set empty_cluster to True
                empty_cluster = True

                # Find number of non-empty clusters
                non_empty = 0
                for j in range(len(groups)):
                    if len(groups[j]) != 0:
                        non_empty += 1
                
                # Suggest a different maximum number of independent OPs to the user
                logger.error(f'The number of non-empty clusters is {non_empty}, try setting the maximum number of independent OPs to {non_empty} or less.')

                break

            result = group_evaluation(groups[i], mut)
            centers.append(result)

    return centers, empty_cluster

def set_bandwidth(ops, kernel, weights):
    """Calculates the bandwidth consistent with Scott and Silverman's
    rules of thumb for bandwidth selection.
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    kernel : str
        Kernel name for kernel density estimation.
        
    weights : list of floats or numpy array
        The weights associated with each data point after reweighting an enhanced 
        sampling trajectory.
        
    Returns
    -------
    bandwidth : float
        Bandwidth from the rules of thumb (they're the same for 2D KDE).
        
    """
    
    if kernel == 'epanechnikov':
        bw_constant = 2.2
    else:
        bw_constant = 1

    if type(weights) == type(None):
        n = np.shape(ops[0].traj)[0]
    else:
        weights = np.array(weights)
        n = np.sum(weights)**2 / np.sum(weights**2)
    bandwidth = bw_constant*n**(-1/6)

    logger.info('      Selected bandwidth: ' + str(bandwidth)+ '\n')

    return bandwidth
        
def starting_centroids(all_ops, num_clusters, mut):
    """Makes a DissimilarityMatrix and scans all OPs forward and backward
    adding OPs to have maximum separation between starting centroids.
    
    Parameters
    ----------
    all_ops : list of OrderParameters
        All OPs.
        
    num_clusters : int
        Number of starting centroids.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    matrix : DissimilarityMatrix
        The DissimilarityMatrix with the starting centroids.
        
    """
    matrix = DissimilarityMatrix(num_clusters, mut)
    for i in all_ops:
        matrix.add_OP(i)
    for i in all_ops[::-1]:
        matrix.add_OP(i)
        
    return matrix

def k_clusters(all_ops, num_clusters, mut):
    """Clusters the OPs using k-means clustering.
    Calculates the total distortion for the clustering.
    
    all_ops : list of OrderParameters
        All OPs.
        
    num_clusters : int
        The number of clusters to be calculated.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    centroids : list of OrderParameters
        Centroids resulting from clustering.
        
    disto : float
        The total distortion from all centroids.
    
    empty_cluster : bool
        True if there is an empty cluster, False otherwise.
    """

    # DM construction
    matrix = starting_centroids(all_ops, num_clusters, mut)

    # Clustering
    seed = []
    for i in matrix.OPs:
        seed.append(i)

    centroids, empty_cluster = cluster(all_ops, seed, mut)
    disto = distortion(centroids, all_ops, mut)

    return centroids, disto, empty_cluster

def num_clust(distortion_list, num_clusters_list):
    """Calculates the optimal number of clusters given
    the distortion for each k-clustering.
    
    Parameters
    ----------
    distortion_list : list of floats
        The total distortion for each k-clustering.
        
    num_clusters_list : list of ints
        The number of clusters associated with the distortions.
        
    Returns
    -------
    num_ops : int
        The optimal number of clusters/centroids.
        
    """
    
    num_ops = 0
    all_jumps = []

    for dim in range(1,11):
        neg_expo = np.array(distortion_list) ** (-0.5 * dim)
        jumps = []
        for i in range(len(neg_expo) - 1):
            jumps.append(neg_expo[i] - neg_expo[i + 1])
        all_jumps.append(jumps)

        min_index = 0
        for i in range(len(jumps)):
            if jumps[i] > jumps[min_index]:
                min_index = i
        if num_clusters_list[min_index] > num_ops:
            num_ops = num_clusters_list[min_index]
        
        return num_ops

def plot_distance_matrix(distance_matrix, op_labels, figure_path):
    """Plots the mutual information matrix as a heatmap. Make the size of the 
    figure vary depending on the number of OPs. 
    
    Parameters
    ----------
    distance_matrix : numpy array
        The mutual information matrix.
        
    op_labels : list of strings
        The labels for the OPs.
        
    mut_info_image_path : string
        The path to save the mutual information matrix image.
        
    """
        
    # Set the size of the figure

    if len(op_labels) > 300:
        fig_size = (40, 40)
    elif len(op_labels) > 100:
        fig_size = (30, 30)
    elif len(op_labels) > 50:
        fig_size = (20, 20)
    elif len(op_labels) > 25:
        fig_size = (15, 15)
    else:
        fig_size = (10, 10)
    
    # Plot the mutual information matrix as a heatmap 
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(distance_matrix, cmap='viridis', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(op_labels)))
    ax.set_yticks(np.arange(len(op_labels)))
    ax.set_xticklabels(op_labels)
    ax.set_yticklabels(op_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations, only if the number of OPs is less than 20
    if len(op_labels) < 20:
        for i in range(len(op_labels)):
            for j in range(len(op_labels)):
                ax.text(j, i, round(distance_matrix[i, j],2),
                            ha="center", va="center", color="w")
    
    # Add a colorbar 
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title("Distance Matrix")
    fig.tight_layout()
    plt.savefig(figure_path)
    plt.close()


# Clustering: find the optimal number of clusters and the centroids
def find_ops(input_distance_matrix=None, input_op_labels=None, all_ops=None, 
             max_outputs=20, min_outputs=0, bins=20, output_dir = None, bandwidth=None, kernel='epanechnikov',
             distortion_filename=None, return_memo=False,  weights=None):
    """Main function performing clustering and finding the optimal number of OPs.
    
    Parameters
    ----------

    input_distance_matrix : numpy array 
        Array with pre-computed distance matrix 

    input_op_labels : list 
        Strings with OP labels 

    all_ops : list of OrderParameters
        All OPs for clustering.
        
    max_outputs : int
        The maximum number of clusters/centroids.
    
    min_outputs : int
        The minimum number of clusters/centroids.
        
    bins : int or None
        Number of values along each axis for the joint probability. 
        The probability will be a bins x bins grid.
        If None this is set with a rule of thumb.
    
    output_dir : str or None 
        The path to the output directory.
        
    bandwidth : float or None
        Bandwidth parameter for kernel density estimation.
        If None this is set with a rule of thumb.
        
    kernel : str
        Kernel name for kernel density estimation.
        It is recommended to use either epanechnikov (parabolic) or gaussian.
        These are currently the only two implemented in bandwidth rule of thumb.
        
    distortion_filename : str or None
        The filename to save distortion jumps.
        
    return_memo: bool
        Option to also return the memoizer used.
        
    weights : list of floats or numpy array
        The weights associated with each data point after re-weighting an enhanced 
        sampling trajectory.

    Returns
    -------
    list of OPs
        The centroids for the optimal clustering.
        
    mut: Memoizer (only with return_memo)
        The Memoizer used to calculate the distances used in clustering.
        
    """

    # Measure time for debugging purposes
    clustering_start_time = time.time()
    
    # Initialize parameters
    if kernel == 'parabolic':
        kernel = 'epanechnikov'
    if bandwidth == None and all_ops is not None:
        bandwidth = set_bandwidth(all_ops, kernel, weights)
    if bins == None and all_ops is not None:
        bins = np.ceil(np.sqrt(len(all_ops[0].traj)))
    
    # Initialize memoizer
    mut = Memoizer(bins, bandwidth, kernel, weights)

    num_clusters_list = []
    distortion_list = []
    op_dict = {}
    
    # Obtain the mutual information-based distance matrix
    if input_distance_matrix is None:
        logger.info('       Calculating all pairwise distances...')
        compute_matrix(all_ops, mut)
    else:
        logger.info('       Using pre-calculated distance matrix...')
        read_matrix(input_distance_matrix, input_op_labels, mut)
        all_ops = [OrderParameter(op_name, None) for op_name in input_op_labels]

    # Cluster with different number of centroids and calculate the distortion for each clustering
    for num_clusters in range(min_outputs, max_outputs + 1):

        logger.info("       Clustering with " + str(num_clusters) + " centroids...")

        ops_centroids, disto, empty_cluster = k_clusters(all_ops, num_clusters, mut)

        # If there is an empty cluster, stop clustering and return the results
        if empty_cluster:
            logger.info('       Empty cluster found. Stopping clustering and returning the results.')
            break

        num_clusters_list.append(num_clusters)
        op_dict[num_clusters] = ops_centroids
        distortion_list.append(disto)

    # Determine optimal number of clusters that minimizes the distortion
    num_ops = num_clust(distortion_list, num_clusters_list)

    # Save output
    if output_dir is not None:

        np.save(os.path.join(output_dir, 'distortion.npy'), distortion_list[::-1])
        
        # Get the mutual information-based distance matrix for the old ops
        complete_distance_matrix = get_matrix(all_ops, mut)

        # Get the mutual information-based distance matrix for the new ops
        solution_distance_matrix = get_matrix(op_dict[num_ops], mut)

        # Find list of names for the old ops
        all_op_names = [op.name for op in all_ops]

        # Find list of names for the new ops
        solution_names = [op.name for op in op_dict[num_ops]]

        # Plot the complete mutual information-based distance matrix 
        plot_distance_matrix(complete_distance_matrix, all_op_names, os.path.join(output_dir, 'dist_matrix_complete.png'))
        
        # Save the complete mutual information-based distance matrix
        np.save(os.path.join(output_dir, 'dist_matrix_complete.npy'), complete_distance_matrix)

        # Save the labels for the complete mutual information-based distance matrix in a text file
        with open(os.path.join(output_dir, 'dist_matrix_complete_labels.txt'), 'w') as f:
            for item in all_op_names:
                f.write(f"{item}")

        # Plot the mutual information-based distance matrix for the new ops
        plot_distance_matrix(solution_distance_matrix, solution_names, os.path.join(output_dir, 'dist_matrix_filtered.png'))
    
    if return_memo:
        return op_dict[num_ops], mut

    # Log time for debugging purposes
    clustering_end_time = time.time()
    logger.debug(f'    Time to find independent order parameters: {clustering_end_time - clustering_start_time:.2f} s')
        
    return op_dict[num_ops]

def find_ops_from_matrix(input_distance_matrix, input_op_labels, max_outputs=20, distance_figure_path = None, distortion_filename=None, return_memo=False):
    """Function performing clustering and finding the optimal number of OPs starting from the pre-calculated distance matrix.
    
    Parameters
    ----------

    input_distance_matrix : numpy array 
        Array with distance matrix 
 
    input_op_labels : list 
        Strings with OP labels 
        
    max_outputs : int
        The maximum number of clusters/centroids.
    
    distance_figure_path : str or None 
        The path to save the figure of the distance matrix.
    
    distortion_filename : str or None
        The filename to save distortion jumps.
    
    return_memo: bool
        Option to also return the memoizer used.
        
    Returns
    -------
    list of OPs
        The centroids for the optimal clustering.
        
    mut: Memoizer (only with return_memo)
        The Memoizer used to calculate the distances used in clustering.
        
    """

    # Measure time for debugging purposes
    clustering_start_time = time.time()

    # Not used - dummy values
    weights = None
    bandwidth = 0.2
    kernel = 'epanechnikov'
    bins = 20
    
    mut = Memoizer(bins, bandwidth, kernel, weights)
    distortion_array = []
    num_array = []
    op_dict = {}
    
    logger.info('       Using pre-calculated distance matrix...')
    read_matrix(input_distance_matrix, input_op_labels, mut)

    all_ops = [OrderParameter(op_name, None) for op_name in input_op_labels]

    # This loops through each number of clusters
    while (max_outputs > 0):

        logger.info("       Clustering with " + str(max_outputs) + " centroids...")
        
        tmp_ops, disto = k_clusters(all_ops, max_outputs, mut)

        num_array.append(max_outputs)
        op_dict[max_outputs] = tmp_ops
        distortion_array.append(disto)
        
        max_outputs = max_outputs - 1

    if not distortion_filename == None:
        np.save(distortion_filename, distortion_array[::-1])
        
    # Determining number of clusters
    num_ops = num_clust(distortion_array, num_array)

    if distance_figure_path:
        
        # Get the mutual information-based distance matrix for the old ops
        complete_distance_matrix = get_matrix(all_ops, mut)

        # Get the mutual information-based distance matrix for the new ops
        solution_distance_matrix = get_matrix(op_dict[num_ops], mut)

        # Find list of names for the old ops
        all_op_names = [op.name for op in all_ops]

        # Find list of names for the new ops
        solution_names = [op.name for op in op_dict[num_ops]]

        # Remove the .png extension from the distance matrix path if it exists
        distance_figure_path = distance_figure_path.replace('.png', '')

        # Plot the complete mutual information-based distance matrix 
        plot_distance_matrix(complete_distance_matrix, all_op_names, distance_figure_path + '_complete.png')
        
        # Save the complete mutual information-based distance matrix
        np.save(distance_figure_path + '_complete.npy', complete_distance_matrix)

        # Save the labels for the complete mutual information-based distance matrix in a text file
        with open(distance_figure_path + '_complete_labels.txt', 'w') as f:
            for item in all_op_names:
                f.write(f"{item}")

        # Plot the mutual information-based distance matrix for the new ops
        plot_distance_matrix(solution_distance_matrix, solution_names, distance_figure_path + '_solution.png')
    
    if return_memo:
        return op_dict[num_ops], mut

    # Log time for debugging purposes
    clustering_end_time = time.time()
    logger.debug(f'    Time to find independent order parameters: {clustering_end_time - clustering_start_time:.2f} s')
        
    return op_dict[num_ops]

# Compute the mutual information-based distance matrix
def find_matrix(all_ops, bins, bandwidth, kernel, distance_figure_path):

    """
    Find the distance matrix between all order parameters

    Parameters
    ----------
    all_ops : list of OrderParameters
        All OPs to compute the pairwise distance for.

    bins: int
        Number of values along each axis for the joint probability. 
        The probability will be a bins x bins grid.
        If None this is set with a rule of thumb.
    
    bandwidth : float or None
        Bandwidth parameter for kernel density estimation.
        If None this is set with a rule of thumb.
        
    kernel : str
        Kernel name for kernel density estimation.
        It is recommended to use either epanechnikov (parabolic) or gaussian.
        These are currently the only two implemented in bandwidth rule of thumb.

    distance_figure_path : str
        The path to save the figure of the distance matrix.
    """
    
    if kernel == 'parabolic':
        kernel = 'epanechnikov'
    if bandwidth == None:
        bandwidth = set_bandwidth(all_ops, kernel, None)
    
    mut = Memoizer(bins, bandwidth, kernel)

    if bins == None:
        bins = np.ceil(np.sqrt(len(all_ops[0].traj)))
        
    logger.info('       Calculating all pairwise distances...')
    compute_matrix(all_ops, mut)

    # Get the mutual information-based distance matrix for the old ops
    complete_distance_matrix = get_matrix(all_ops, mut)

    # Find list of names for the old ops
    all_op_names = [op.name for op in all_ops]

    # Plot the complete mutual information-based distance matrix 
    plot_distance_matrix(complete_distance_matrix, all_op_names, distance_figure_path)

# Apply the AMINO clustering algorithm to the time series of a set of order parameters contained in a colvars file
def amino(ops_subset: list, colvar_path: str, output_dir: str, amino_settings: dict, sampling_settings: dict):

    if amino_settings.get('run_amino', True) is False:
        logger.info('Skipping AMINO clustering.')
        return ops_subset
    else:
        logger.info('Running AMINO clustering.')
    
    # Read amino settings
    max_independent_ops = amino_settings.get('max_independent_ops', 50)
    min_independent_ops = amino_settings.get('min_independent_ops', 10)
    ops_batch_size = amino_settings['ops_batch_size']
    num_bins = amino_settings.get('num_bins', 50)
    bandwidth = amino_settings.get('bandwidth', 0.1)

    # Read stratified sampling settings
    num_samples = sampling_settings.get('num_samples', None)
    total_num_samples = sampling_settings.get('total_num_samples', None)
    relaxation_time = sampling_settings.get('relaxation_time', 1)

    # Determine the order parameter batch size  
    if ops_batch_size is None:
        ops_batch_size = len(ops_subset)   
    else:
        ops_batch_size = min(int(ops_batch_size), len(ops_subset))

        # Check ops_batch_size is greater than max_independent_ops
        if ops_batch_size <= max_independent_ops:
            logger.error(f'ops_batch_size must be greater than max_independent_ops. ops_batch_size = {ops_batch_size} and max_independent_ops = {max_independent_ops}.')
            sys.exit(1)

    logger.info(f'Size of order parameters set: {len(ops_subset)}.')
    logger.info(f'Batch size of order parameters: {ops_batch_size}.')
    logger.info(f'Maximum number of independent order parameters: {min(max_independent_ops, len(ops_subset)),}.')
    logger.info(f'Minimum number of independent order parameters: {min_independent_ops}.')
    logger.info(f'Number of bins to estimate the joint probability density: {num_bins}.') 
    
    # Determine if stratified sampling will be used
    if num_samples is None:
        logger.info('Using all samples, no stratified sampling will be used.')
        stratified_samples = None
    else:
        num_samples = int(num_samples)
        total_num_samples = int(total_num_samples)

        if relaxation_time:
            relaxation_time = int(relaxation_time)
        else:
            relaxation_time = 1
        logger.info(f'Relaxation time: {relaxation_time} samples.')

        # Samples the time series data using a stratified sampling approach
        stratified_samples = common.stratified_sampling(num_samples, total_num_samples, relaxation_time)

    # Determine initial list of batches to analyze
    ops_batches = common.divide_into_batches(ops_subset, ops_batch_size)

    # Loop until ops_subset is smaller or equal than requested ops batch size
    while len(ops_subset) > ops_batch_size:

        logger.info(f'   Current size of OPs subset:  {len(ops_subset)}')

        # Initialize the new list of independent order parameters
        new_ops_subset = []

        # Apply AMINO to each batch of order parameters
        for batch_index, ops_batch in enumerate(ops_batches):

            logger.info(f'      Applying AMINO to batch {batch_index + 1} of {len(ops_batches)}.')
            logger.debug(f'      Order parameters in the batch: {ops_batch}')

            # Read the time series data of the order parameters in the batch 
            ops_batch_data = common.read_colvars(colvar_path, ops_batch, stratified_samples)

            # Build the list of order parameters objects from their names and time series data
            OPS_batch = [OrderParameter(op_name, ops_batch_data[op_name]) for op_name in ops_batch]

            # Find the independent order parameters
            independent_ops = find_ops(all_ops = OPS_batch, 
                                                  max_outputs = min(max_independent_ops, len(ops_batch)), 
                                                  min_outputs = min_independent_ops,
                                                  bins = num_bins,
                                                  bandwidth=bandwidth)

            independent_op_names = [op.name for op in independent_ops]

            # Log the results
            logger.debug(f'      Found {len(independent_ops)} independent order parameters.')
            logger.debug(f'      Independent order parameters in the batch:{independent_op_names}')

            # Save the results
            new_ops_subset.extend(independent_op_names)

        # Find the next batch of order parameters to analyze
        ops_batches = common.divide_into_batches(new_ops_subset, ops_batch_size)

        # Check if the size of the subset is the same.
        if len(new_ops_subset) == len(ops_subset):
            logger.info(f'  The size of the subset of order parameters is the same as the previous iteration ({len(new_ops_subset)}).')
            break

        # Update the list of independent order parameters
        ops_subset = new_ops_subset

    # Read the time series data of the order parameters in the batch
    ops_batch_data = common.read_colvars(colvar_path, ops_subset, stratified_samples)
    
    # Build the list of order parameters objects from their names and time series data
    subset_ops = [OrderParameter(op_name, ops_batch_data[op_name]) for op_name in ops_subset]
  
    # Find the independent order parameters
    independent_ops = find_ops(all_ops = subset_ops, 
                                max_outputs = min(max_independent_ops, len(ops_subset)), 
                                min_outputs = min_independent_ops, 
                                bins = num_bins, 
                                output_dir = output_dir,
                                bandwidth=bandwidth)

    independent_op_names = [op.name for op in independent_ops]
    independent_op_names.sort()

    return independent_op_names



