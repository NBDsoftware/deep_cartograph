# Import necessary modules
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Literal

# Local imports
from deep_cartograph.modules.md import md
from deep_cartograph.modules.figures import figures
from deep_cartograph.modules.statistics import statistics
from deep_cartograph.yaml_schemas.train_colvars import TrainColvarsSchema
from deep_cartograph.modules.common import package_is_installed, validate_configuration, files_exist, merge_configurations 

from deep_cartograph.tools.train_colvars.cv_calculator import cv_calculators_map

# Set logger
logger = logging.getLogger(__name__)

class TrainColvarsWorkflow:
    """
    Class to train collective variables from colvars files.
    """
    def __init__(self, 
                 configuration: Dict, 
                 training_colvars_paths: List[str], 
                 feature_constraints: Union[List[str], str] = None,
                 sup_colvars_paths: Union[List[str], None] = None, 
                 sup_labels: Union[List[str], None] = None,
                 cv_dimension: Union[int, None] = None,
                 cvs: List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']] = None,
                 trajectory_paths: Union[List[str], None] = None,
                 topology_paths: Union[List[str], None] = None,
                 ref_topology_path: Union[str, None] = None,  
                 samples_per_frame: Union[float, None] = 1,
                 output_folder: str = 'train_colvars'):
        """
        Initializes the TrainColvarsWorkflow class.
        
        The class runs the train_colvars workflow, which consists of:
        
            1. Use training_colvars_paths files to compute the collective variables.
            2. Plotting the FES of each colvars file in the CV space. 
            3. Plotting the projected features onto the CV space colored by order.
            4. Clustering the input samples according to the distance in the CV space.
            5. Plotting the projected features onto the CV space colored by cluster.
            6. Extracting clusters from the trajectory.
        """
        
        # Set output folder
        self.output_folder: str = output_folder
        
        # Configuration related attributes
        self.configuration: Dict = validate_configuration(configuration, TrainColvarsSchema, output_folder)
        self.figures_configuration: Dict = self.configuration['figures']
        self.clustering_configuration: Dict = self.configuration['clustering']
                
        # Input related attributes
        self.training_colvars_paths: List[str] = training_colvars_paths
        self.feature_constraints: Union[List[str], str] = feature_constraints
        self.sup_colvars_paths: Union[List[str], None] = sup_colvars_paths
        self.trajectory_paths: Union[str, None] = trajectory_paths
        self.topology_paths: Union[str, None] = topology_paths
        self.ref_topology_path: Union[str, None] = ref_topology_path
        
        if self.topology_paths:
            if self.ref_topology_path is None:
                self.ref_topology_path = self.topology_paths[0]
        
        if not samples_per_frame:
            self.samples_per_frame = 1
        else:
            self.samples_per_frame: float = samples_per_frame
            
        self.sup_labels: Union[List[str], None] = sup_labels
        
        # Validate inputs existence
        self._validate_files()

        # CV related attributes
        self.cvs: List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']] = cvs if cvs else self.configuration['cvs']
        self.cv_dimension: int = cv_dimension
        
    def _validate_files(self):
        """Checks if provided input files exist."""
        
        for path in self.training_colvars_paths:
            if not files_exist(path):
                logger.error(f"Colvars file {path} does not exist. Exiting...")
                sys.exit(1)
            
        if self.sup_colvars_paths: 
            for path in self.sup_colvars_paths:
                if not files_exist(path):
                    logger.error(f"Reference colvars file {path} does not exist. Exiting...")
                    sys.exit(1)
                
        if self.trajectory_paths:
            for path in self.trajectory_paths:
                if not files_exist(path):
                    logger.error(f"Trajectory file {path} does not exist. Exiting...")
                    sys.exit(1)
        
            if self.topology_paths:
                for path in self.topology_paths:
                    if not files_exist(path):
                        logger.error(f"Topology file {path} does not exist. Exiting...")
                        sys.exit(1)
                
                if self.ref_topology_path:
                    if not files_exist(self.ref_topology_path):
                        logger.error(f"Reference topology file {self.ref_topology_path} does not exist. Exiting...")
                        sys.exit(1)
            else:
                logger.error("Trajectory file provided but no topology file. Exiting...")
                sys.exit(1)
            
            if len(self.trajectory_paths) != len(self.topology_paths):
                logger.error("Different number of trajectory and topology files provided. Exiting...")
                sys.exit(1)
            
            if len(self.trajectory_paths) != len(self.training_colvars_paths):
                logger.error("Different number of trajectory and colvars files provided. Exiting...")
                sys.exit(1)
          
    def run(self):
        """
        Run the train_colvars workflow.
        """
        
        logger.info(f"Collective variables to compute: {self.cvs}")
        
        # For each requested collective variable
        for cv in self.cvs:
            
            if cv != "pca" and not package_is_installed('mlcolvar', 'torch', 'lightning'):
                logger.warning(f"Missing packages for {cv}. Skipping this CV.")
                continue
            
            cv_output_folder = os.path.join(self.output_folder, cv)
            
            # Merge common and cv-specific configuration for this cv 
            common_configuration = self.configuration['common']
            cv_specific_configuration = self.configuration.get(cv, {})
            cv_configuration = merge_configurations(common_configuration, cv_specific_configuration)
            
            # Construct the corresponding CV calculator NOTE: we need here the sup topology as well
            args = {
                'training_colvars_paths': self.training_colvars_paths,
                'configuration': cv_configuration,
                'topology_paths': self.topology_paths,
                'ref_topology_path': self.ref_topology_path,
                'feature_constraints': self.feature_constraints,
                'sup_colvars_paths': self.sup_colvars_paths,
                'output_path': self.output_folder
            }
            cv_calculator = cv_calculators_map[cv](**args)
            
            # Run the CV calculator - obtain a DictDataset with the projected colvars
            projected_data_df = cv_calculator.run(self.cv_dimension) # NOTE: These should contain the labels as well
                
            # Obtain the projected supplementary data if any NOTE: there is a type mismatch here
            projected_sup_data_df = cv_calculator.get_projected_sup_data()
            if projected_sup_data_df is not None:
                X_ref = [projected_sup_data_df[projected_sup_data_df['label'] == index] for index in range(len(self.sup_colvars_paths))]
                X_ref = [df.drop('label', axis=1).to_numpy() for df in X_ref]
                X_ref_labels = [Path(path).stem for path in self.sup_colvars_paths]
            else:
                X_ref = None
                X_ref_labels = None
                
            # If the DictDataset is not empty, proceed with the workflow
            if projected_data_df is not None:
                
                # Iterate over the training data
                for file_index in range(len(self.training_colvars_paths)):
                    
                    # Get the colvars, trajectory and topology files
                    colvars = self.training_colvars_paths[file_index]
                    trajectory = self.trajectory_paths[file_index] if self.trajectory_paths else None
                    topology = self.topology_paths[file_index] if self.topology_paths else None

                    # Log
                    logger.info(f"Projecting colvars file: {colvars}")
                    logger.info(f"Corresponding trajectory file: {trajectory}")
                    logger.info(f"Corresponding topology file: {topology}")
                    
                    # Output folder for the current trajectory
                    traj_output_folder = os.path.join(cv_output_folder, Path(trajectory).stem)
                    os.makedirs(traj_output_folder, exist_ok=True)
                    
                    # Get the projected data for this colvars file
                    projected_colvars_df = projected_data_df[projected_data_df['label'] == file_index]
                    projected_colvars_df.drop('label', axis=1, inplace=True)
                
                    # Plot FES of the CV space - add ref data if available
                    figures.plot_fes(
                        X = projected_colvars_df.to_numpy(),
                        cv_labels = cv_calculator.get_labels(),
                        X_ref = X_ref,
                        X_ref_labels = X_ref_labels,
                        settings = self.figures_configuration['fes'],
                        output_path = traj_output_folder)

                    # If clustering is enabled
                    if self.clustering_configuration['run']:
                        
                        # Cluster the input samples
                        cluster_labels, centroids = statistics.optimize_clustering(projected_colvars_df.to_numpy(), self.clustering_configuration)
                        
                        # Add a column with the order of the data points
                        projected_colvars_df['order'] = np.arange(projected_colvars_df.shape[0])
                        
                        # Add cluster labels to the projected input dataframe
                        projected_colvars_df['cluster'] = cluster_labels
                        
                        # Find centroids among input samples
                        if len(centroids) > 0:
                            centroids_df = statistics.find_centroids(projected_colvars_df, centroids, cv_calculator.get_labels())
                            
                        # Generate color map for the clusters
                        num_clusters = len(np.unique(cluster_labels))
                        cluster_colors = figures.generate_colors(num_clusters, self.figures_configuration['traj_projection']['cmap'])

                        # Plot cluster sizes 
                        figures.plot_clusters_size(cluster_labels, cluster_colors, traj_output_folder)
                        
                        # Extract clusters from the trajectory
                        if (None not in [trajectory, topology]) and (len(centroids) > 0):
                            md.extract_clusters_from_traj(trajectory_path = trajectory, 
                                                        topology_path = topology,
                                                        traj_df = projected_colvars_df,
                                                        samples_per_frame = self.samples_per_frame, 
                                                        centroids_df = centroids_df,
                                                        cluster_label = 'cluster',
                                                        frame_label = 'order', 
                                                        output_folder = os.path.join(traj_output_folder, 'clusters'))
                        elif len(centroids) == 0:
                            logger.warning("No centroids found. Skipping extraction of clusters from the trajectory.")
                        else:
                            logger.warning("Trajectory and/or topology files not provided. Skipping extraction of clusters from the trajectory.")
                            logger.debug(f"Trajectory: {trajectory}")
                            logger.debug(f"Topology: {topology}")
                    
                    # Add a column with the order of the data points
                    projected_colvars_df['order'] = np.arange(projected_colvars_df.shape[0])
                        
                    # 2D plots of the input data projected onto the CV space
                    if cv_calculator.get_cv_dimension() == 2:
                        
                        # Colored by order
                        figures.gradient_scatter_plot(
                            data = projected_colvars_df,
                            column_labels = cv_calculator.get_labels(),
                            color_label = 'order',
                            settings = self.figures_configuration['traj_projection'],
                            file_path = os.path.join(traj_output_folder,'trajectory.png'))
                    
                        # If clustering is enabled
                        if self.clustering_configuration['run']:
                            
                            # Colored by cluster
                            figures.clusters_scatter_plot(
                                data = projected_colvars_df, 
                                column_labels = cv_calculator.get_labels(),
                                cluster_label = 'cluster', 
                                settings = self.figures_configuration['traj_projection'], 
                                file_path = os.path.join(traj_output_folder,'trajectory_clustered.png'),
                                cluster_colors = cluster_colors)
                    
                    # Erase the order column
                    projected_colvars_df.drop('order', axis=1, inplace=True)
                    
                    # Save the projected input data
                    projected_colvars_df.to_csv(os.path.join(traj_output_folder,'projected_trajectory.csv'), index=False, float_format='%.4f')
            
            else:
                logger.warning(f"Projected colvars dataframe is empty for {cv}. Skipping this CV.")
                continue