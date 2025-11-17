# Import necessary modules
import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional

# Local imports
from deep_cartograph.modules.md import md
from deep_cartograph.modules.figures import figures
from deep_cartograph.modules.statistics import statistics
from deep_cartograph.yaml_schemas.traj_cluster import TrajClusterSchema
from deep_cartograph.modules.common import validate_configuration, files_exist, merge_configurations 

# Set logger
logger = logging.getLogger(__name__)

class TrajClusterWorkflow:
    """
    Class to cluster trajectories based on collective variables.
    """
    def __init__(self, 
                configuration: Dict, 
                cv_traj_paths: List[str],
                trajectories: Optional[List[str]] = None,
                topologies: Optional[List[str]] = None,
                sup_cv_traj_paths: Optional[List[str]] = None,
                sup_trajectories: Optional[List[str]] = None,
                sup_topologies: Optional[List[str]] = None,
                frames_per_sample: Optional[int] = 1,
                output_folder: str = 'traj_cluster'):
        """
        Initializes the TrajClusterWorkflow class.
        
        The class runs the traj_cluster workflow, which consists of:

            1. Cluster the collective variable data from 'cv_traj_paths'.
            2. Assign cluster labels to collective variable data in 'sup_cv_traj_paths' if provided. 
               Looking at cluster of closest point in 'cv_traj_paths'.
            3. Extract representative structures from trajectories in 'trajectories' and 'sup_trajectories' 
               based on the clusters.
        """
        
        # Set output folder
        self.output_folder: str = output_folder
        
        # Configuration related attributes
        self.configuration: Dict = validate_configuration(configuration, TrajClusterSchema, output_folder)
        self.figures_configuration: Dict = self.configuration['figures']
        
        # Input related attributes
        self.cv_traj_paths: List[str] = cv_traj_paths
        self.trajectories: Union[str, None] = trajectories
        self.topologies: Union[str, None] = topologies
        self.sup_cv_traj_paths: Union[List[str], None] = sup_cv_traj_paths
        self.sup_trajectories: Union[List[str], None] = sup_trajectories
        self.sup_topologies: Union[List[str], None] = sup_topologies
        
        if not frames_per_sample:
            self.frames_per_sample = 1
        else:
            self.frames_per_sample: float = frames_per_sample
        
        # Validate inputs existence
        self._validate_files()

        # CV related attributes
        self.cv_dimension: int = None
        self.cv_labels: List[str] = None
        
    def _validate_files(self):
        """Checks if provided input files exist."""
        
        for path in self.cv_traj_paths:
            if not files_exist(path):
                logger.error(f"Collective variable trajectory {path} does not exist. Exiting...")
                sys.exit(1)

        if self.trajectories:
            for path in self.trajectories:
                if not files_exist(path):
                    logger.error(f"Trajectory {path} does not exist. Exiting...")
                    sys.exit(1)
        
            if self.topologies:
                for path in self.topologies:
                    if not files_exist(path):
                        logger.error(f"Topology file {path} does not exist. Exiting...")
                        sys.exit(1)
            else:
                logger.error("Trajectory files provided but no topology file. Exiting...")
                sys.exit(1)
            
            if len(self.trajectories) != len(self.topologies):
                logger.error("Different number of trajectory and topology files provided. Exiting...")
                sys.exit(1)
            
            if len(self.trajectories) != len(self.cv_traj_paths):
                logger.error("Different number of trajectory and colvars files provided. Exiting...")
                sys.exit(1)

        if self.sup_cv_traj_paths:
            
            for path in self.sup_cv_traj_paths:
                if not files_exist(path):
                    logger.error(f"Supplementary collective variable trajectory {path} does not exist. Exiting...")
                    sys.exit(1)
                    
            if self.sup_trajectories:
                for path in self.sup_trajectories:
                    if not files_exist(path):
                        logger.error(f"Supplementary trajectory {path} does not exist. Exiting...")
                        sys.exit(1)
                
                if self.sup_topologies:
                    for path in self.sup_topologies:
                        if not files_exist(path):
                            logger.error(f"Supplementary topology file {path} does not exist. Exiting...")
                            sys.exit(1)
                else:
                    logger.error("Supplementary trajectory files provided but no topology file. Exiting...")
                    sys.exit(1)
                
                if len(self.sup_trajectories) != len(self.sup_topologies):
                    logger.error("Different number of supplementary trajectory and topology files provided. Exiting...")
                    sys.exit(1)
                
                if len(self.sup_trajectories) != len(self.sup_cv_traj_paths):
                    logger.error("Different number of supplementary trajectory and colvars files provided. Exiting...")
                    sys.exit(1)
            else:
                logger.error("Supplementary colvars files provided but no trajectory files. Exiting...")
                sys.exit(1)
                     
    def extract_centroids(self, data: pd.DataFrame):    
        """ 
        Extract PDB for centroids from the trajectories based on the traj_df.
        """
        
        logger.info("Extracting centroids from the trajectories...")
        
        # Keep only the rows that are centroids
        centroids_data = data[data['centroid'] == True]
        
        # For each centroid, find the trajectory and frame
        for index, row in centroids_data.iterrows():
            traj_index = row['traj_label']
            frame = row['frame']
            cluster_label = row['cluster']
            
            logger.info(f"Extracting centroid for cluster {cluster_label} from trajectory {Path(self.trajectories[traj_index]).name} at frame {frame}.")
            
            centroids_folder = os.path.join(self.output_folder, "centroids")
            if not os.path.exists(centroids_folder):
                os.makedirs(centroids_folder)
                
            centroid_path = os.path.join(centroids_folder, f"cluster_{cluster_label}.pdb")
            
            # Extract the frame from the trajectory
            md.extract_PDB(trajectory_path = self.trajectories[traj_index],
                           topology_path = self.topologies[traj_index],
                           pdb_frame = frame,
                           pdb_path = centroid_path)
        
    def extract_cluster_ensembles(self, data: pd.DataFrame, output_folder: str):
        """
        Extract all cluster ensembles from the trajectories based on the traj_df.
        """     
        
        logger.info("Extracting cluster ensembles from the trajectories...")
        
        # For each trajectory
        for traj_index in range(len(self.cv_traj_paths)):
            
            traj_data = data[data['traj_label'] == traj_index]
            trajectory_path = self.trajectories[traj_index]
            topology_path = self.topologies[traj_index]
            
            # For each cluster in the trajectory
            for cluster_label in traj_data['cluster'].unique():
                
                cluster_data = traj_data[traj_data['cluster'] == cluster_label]
                frames = cluster_data['frame'].tolist()
                
                ensemble_path = os.path.join(output_folder, f"cluster_{cluster_label}.xtc")
                md.extract_XTC(trajectory_path = trajectory_path,
                               topology_path = topology_path,
                               traj_frames = frames,
                               new_traj_path = ensemble_path)
    
    def read_cv_traj_data(self, paths: List[str]) -> pd.DataFrame:
        """
        Reads collective variable trajectory data from given paths into a single DataFrame.
        """
        data = []
        for traj_index, traj_path in enumerate(paths):
            df = pd.read_csv(traj_path)
            df['traj_label'] = traj_index
            data.append(df)
        return pd.concat(data, ignore_index=True)
   
    def assign_closest_cluster(self, new_data: pd.DataFrame, clusters_data: pd.DataFrame) -> np.ndarray:
        """
        Assign cluster labels to new cv data based on closest point in cv space from clusters data.

        Parameters
        ----------
        
        new_data : pd.DataFrame
            DataFrame containing new samples to assign cluster labels to.
            
        clusters_data : pd.DataFrame
            DataFrame containing original samples with cluster labels.
        
        Returns
        -------
        
        np.ndarray
            Array of cluster labels assigned to new_data samples.
        """
        
        from sklearn.neighbors import NearestNeighbors
        
        # Fit NearestNeighbors on original data
        nbrs = NearestNeighbors(n_neighbors=1).fit(clusters_data[self.cv_labels].to_numpy())
        
        # Find the nearest neighbor in clusters_data for each point in new_data
        distances, indices = nbrs.kneighbors(new_data[self.cv_labels].to_numpy())
        
        # Assign cluster labels based on nearest neighbor
        cluster_labels = clusters_data.iloc[indices.flatten()]['cluster'].values

        return cluster_labels
    
    def run(self) -> Dict[str, List[str]]:
        """
        Run the traj_cluster workflow.
        
        Returns
        -------
        
        Dict[str, List[str]]
            A dictionary where keys are the names of the trajectories from cv_traj_paths and values are
            lists of paths to the clustered trajectories in the CV space for each cv trajectory file.
        """
        
        output_paths: Dict[str, List[str]] = {}
        
        logger.info("Starting traj_cluster workflow...")
        
        # Read all the csv files in self.cv_traj_paths into a single dataframe
        # include a column named 'traj_label' to identify the trajectory
        cv_data = self.read_cv_traj_data(self.cv_traj_paths)

        # Update CV info from dataframe shape and labels
        self.cv_dimension = cv_data.shape[1] - 1       
        self.cv_labels = cv_data.columns[:-1].tolist()
                
        # Cluster the cv data
        cluster_labels, centroids = statistics.optimize_clustering(cv_data[self.cv_labels].to_numpy(), self.configuration)
        
        # Add cluster labels 
        cv_data['cluster'] = cluster_labels

        # Find closest samples to centroids among input samples, add centroid column
        cv_data = statistics.find_centroids(cv_data, centroids, self.cv_labels)

        # Generate color map for the clusters
        num_clusters = len(np.unique(cluster_labels))
        cluster_colors = figures.generate_colors(num_clusters, self.figures_configuration['cmap'])
        
        # Add a column with the frame number taking into account self.frames_per_sample
        frames = []
        for traj_index in range(len(self.cv_traj_paths)):
            n_samples = cv_data[cv_data['traj_label'] == traj_index].shape[0]
            frames.extend(np.arange(0, n_samples * self.frames_per_sample, self.frames_per_sample))
        cv_data['frame'] = frames
        
        # Plot cluster sizes
        figures.plot_clusters_size(cluster_labels, cluster_colors, self.output_folder)
        
        # If the user requested output structures = centroids or all, extract centroids from the trajectories
        if self.configuration['output_structures'] in ['centroids', 'all']:
            if self.trajectories and self.topologies:
                self.extract_centroids(cv_data)
            else:
                logger.warning("Trajectory and/or topology files not provided. Skipping extraction of centroids from the trajectory.")
        
        # For each trajectory
        for traj_index in range(len(self.cv_traj_paths)):
            
            # Create output folder for this trajectory -> /traj_cluster/cv_name/traj_name
            if self.trajectories:
                traj_name = Path(self.trajectories[traj_index]).stem
            else:
                traj_name = f"traj_{traj_index}"
            traj_output_folder = os.path.join(self.output_folder, traj_name)
            os.makedirs(traj_output_folder, exist_ok=True)
            
            # Save the cv trajectory with cluster labels
            traj_df = cv_data[cv_data['traj_label'] == traj_index]
            projected_traj_path = os.path.join(traj_output_folder, 'projected_trajectory.csv')
            traj_df.to_csv(projected_traj_path, index=False)
            logger.debug(f"Saved projected trajectory with cluster labels to {projected_traj_path}")
            output_paths[traj_name] = [projected_traj_path]
            
            # Plot the cv trajectory coloring by cluster
            scatter_plot_path = os.path.join(traj_output_folder, "trajectory_clustered.png")
            figures.clusters_scatter_plot(
                data = traj_df,
                column_labels=self.cv_labels,
                cluster_label = 'cluster',
                settings = self.figures_configuration,
                file_path = scatter_plot_path,
                cluster_colors = cluster_colors       
            )
            logger.debug(f"Saved clustered trajectory plot to {scatter_plot_path}")
            
            # If the user requested output structures = all, extract all frames from the trajectory for each cluster
            if self.configuration['output_structures'] == 'all':
                if self.trajectories and self.topologies:
                    self.extract_cluster_ensembles(traj_df, traj_output_folder)
                else:
                    logger.warning("Trajectory and/or topology files not provided. Skipping extraction of cluster ensembles from the trajectory.")
                    
        # If supplementary trajectories are provided, assign cluster labels to them
        if self.sup_cv_traj_paths:
            
            logger.info("Assigning clusters to supplementary collective variable trajectories...")
            
            # Read all the csv files in self.sup_cv_traj_paths into a single dataframe
            # include a column named 'traj_label' to identify the trajectory
            sup_cv_data = self.read_cv_traj_data(self.sup_cv_traj_paths)
            
            # Check that the dimensionality of the supplementary cv data matches the original cv data
            if sup_cv_data.shape[1] - 1 != self.cv_dimension:
                logger.error("Dimensionality of supplementary collective variable data does not match the original data. Exiting...")
                sys.exit(1)
            
            # Assign cluster labels based on closest point in original cv_data
            sup_cluster_labels = self.assign_closest_cluster(sup_cv_data, cv_data)
            
            # Add cluster labels to the supplementary cv data
            sup_cv_data['cluster'] = sup_cluster_labels
            
            # For each supplementary trajectory
            for traj_index in range(len(self.sup_cv_traj_paths)):
                
                # Create output folder for this trajectory -> /traj_cluster/cv_name/traj_name
                if self.sup_trajectories:
                    traj_name = f"sup_{Path(self.sup_trajectories[traj_index]).stem}"
                else:
                    traj_name = f"sup_traj_{traj_index}"
                traj_output_folder = os.path.join(self.output_folder, traj_name)
                os.makedirs(traj_output_folder, exist_ok=True)
                
                # Save the supplementary cv trajectory with cluster labels
                traj_df = sup_cv_data[sup_cv_data['traj_label'] == traj_index]
                projected_traj_path = os.path.join(traj_output_folder, 'projected_trajectory.csv')
                traj_df.to_csv(projected_traj_path, index=False)
                logger.debug(f"Saved projected supplementary trajectory with cluster labels to {projected_traj_path}")
                output_paths[traj_name] = [projected_traj_path]
                
                # Plot the supplementary cv trajectory coloring by cluster
                scatter_plot_path = os.path.join(traj_output_folder, "trajectory_clustered.png")
                figures.clusters_scatter_plot(
                    data = traj_df,
                    column_labels=self.cv_labels,
                    cluster_label = 'cluster',
                    settings = self.figures_configuration,
                    file_path = scatter_plot_path,
                    cluster_colors = cluster_colors       
                )
                logger.debug(f"Saved clustered supplementary trajectory plot to {scatter_plot_path}")

        return output_paths
