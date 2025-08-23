# Import necessary modules
import os
import sys
import copy
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Literal, Optional

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
                 train_colvars_paths: List[str], 
                 train_trajectory_paths: Union[List[str], None] = None,
                 train_topology_paths: Union[List[str], None] = None,
                 ref_topology_path: Union[str, None] = None,  
                 feature_constraints: Union[List[str], str] = None,
                 sup_colvars_paths: Union[List[str], None] = None, 
                 sup_topology_paths: Union[List[str], None] = None,
                 sup_labels: Union[List[str], None] = None,
                 cv_dimension: Union[int, None] = None,
                 cvs: List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']] = None,
                 samples_per_frame: Union[float, None] = 1,
                 output_folder: str = 'train_colvars'):
        """
        Initializes the TrainColvarsWorkflow class.
        
        The class runs the train_colvars workflow, which consists of:
        
            1. Use train_colvars_paths files to compute the collective variables.
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
        self.train_colvars_paths: List[str] = train_colvars_paths
        self.train_trajectory_paths: Union[str, None] = train_trajectory_paths
        self.train_topology_paths: Union[str, None] = train_topology_paths
        self.ref_topology_path: Union[str, None] = ref_topology_path
        self.feature_constraints: Union[List[str], str] = feature_constraints
        self.sup_colvars_paths: Union[List[str], None] = sup_colvars_paths
        self.sup_topology_paths: Union[List[str], None] = sup_topology_paths
        
        if self.train_topology_paths:
            if self.ref_topology_path is None:
                self.ref_topology_path = self.train_topology_paths[0]
        
        if not samples_per_frame:
            self.samples_per_frame = 1
        else:
            self.samples_per_frame: float = samples_per_frame
            
        self.sup_labels: Union[List[str], None] = sup_labels
        
        # Validate inputs existence
        self._validate_files()

        # CV related attributes
        self.cvs_list: List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']] = cvs if cvs else self.configuration['cvs']
        self.cv_dimension: int = cv_dimension
        self.cv_labels: List[str] = None
        self.cv_type: str = None
        
    def _validate_files(self):
        """Checks if provided input files exist."""
        
        for path in self.train_colvars_paths:
            if not files_exist(path):
                logger.error(f"Colvars file {path} does not exist. Exiting...")
                sys.exit(1)
            
        if self.sup_colvars_paths: 
            for path in self.sup_colvars_paths:
                if not files_exist(path):
                    logger.error(f"Reference colvars file {path} does not exist. Exiting...")
                    sys.exit(1)
                    
        if self.sup_topology_paths:
            for path in self.sup_topology_paths:
                if not files_exist(path):
                    logger.error(f"Supplementary topology file {path} does not exist. Exiting...")
                    sys.exit(1)
                
        if self.train_trajectory_paths:
            for path in self.train_trajectory_paths:
                if not files_exist(path):
                    logger.error(f"Trajectory file {path} does not exist. Exiting...")
                    sys.exit(1)
        
            if self.train_topology_paths:
                for path in self.train_topology_paths:
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
            
            if len(self.train_trajectory_paths) != len(self.train_topology_paths):
                logger.error("Different number of trajectory and topology files provided. Exiting...")
                sys.exit(1)
            
            if len(self.train_trajectory_paths) != len(self.train_colvars_paths):
                logger.error("Different number of trajectory and colvars files provided. Exiting...")
                sys.exit(1)
         
    def create_fes_plots(self, 
                         data_df: pd.DataFrame, 
                         output_folder: str,
                         sup_data: Optional[List[np.ndarray]] = None
        ):
        """ 
        Create all the required FES plots
        
        1D plots for each components
        
        2D plots for each pair of components
        
        Inputs
        ------
        
        data_df: 
            Main data to compute the FES
            
        output_folder:
            Directory where the FES will be saved
            
        sup_data:
            Optional supplementary data to compute the FES and show it alongside the main data
        """
        
        # 1D plots for each component
        # Iterate over the dimensions of the projected data
        for dimension in range(self.cv_dimension):

            data_i = data_df.iloc[:,dimension]
            label_i = [self.cv_labels[dimension]]
            sup_data_i = [x[:,dimension] for x in sup_data] if sup_data else None
            
            fes_output_folder = os.path.join(output_folder, f'fes_{self.cv_type}_{dimension+1}')
            os.makedirs(fes_output_folder, exist_ok=True)
            
            # Plot FES
            figures.plot_fes(
                data = data_i.to_numpy(),
                cv_labels = label_i,
                settings = self.figures_configuration['fes'],
                output_path = fes_output_folder,
                num_blocks = 100,  
                sup_data = sup_data_i,
                sup_data_labels = self.sup_labels)
        
        if self.cv_dimension > 1:
            
            # Generate all possible 2D plots
            for i in range(0, self.cv_dimension-1):
                for j in range(i+1, self.cv_dimension):
                    
                    data_ij = data_df.iloc[:, [i,j]]
                    label_ij = [self.cv_labels[i], self.cv_labels[j]]
                    sup_data_ij = [x[:,[i,j]] for x in sup_data] if sup_data else None
    
                    fes_output_folder = os.path.join(output_folder, f'fes_{self.cv_type}_{i+1}_{j+1}')
                    os.makedirs(fes_output_folder, exist_ok=True)
                    
                    # Plot FES
                    figures.plot_fes(
                        data = data_ij.to_numpy(),
                        cv_labels = label_ij,
                        settings = self.figures_configuration['fes'],
                        output_path = fes_output_folder,
                        num_blocks = 1,
                        sup_data = sup_data_ij,
                        sup_data_labels = self.sup_labels)
                                    
    def run(self):
        """
        Run the train_colvars workflow.
        """
        
        logger.info(f"Collective variables to compute: {self.cvs_list}")
        
        # For each requested collective variable
        for cv_name in self.cvs_list:
            
            if cv_name != "pca" and not package_is_installed('mlcolvar', 'torch', 'lightning'):
                logger.warning(f"Missing packages for {cv_name}. Skipping this CV.")
                continue
            
            cv_output_folder = os.path.join(self.output_folder, cv_name)
            
            # Merge common and CV-specific configurations
            merged_configuration = merge_configurations(self.configuration['common'], self.configuration.get(cv_name, {}))
            
            # Construct the corresponding CV calculator
            args = {
                'configuration': copy.deepcopy(merged_configuration),
                'train_colvars_paths': self.train_colvars_paths,
                'train_topology_paths': self.train_topology_paths,
                'ref_topology_path': self.ref_topology_path,
                'feature_constraints': self.feature_constraints,
                'sup_colvars_paths': self.sup_colvars_paths,
                'sup_topology_paths': self.sup_topology_paths,
                'output_path': self.output_folder
            }
            cv_calculator = cv_calculators_map[cv_name](**args)
            
            # Run the CV calculator - obtain a dataframe with the projected training data
            projected_train_df = cv_calculator.run(self.cv_dimension)
                
            # Obtain the projected supplementary data if any 
            projected_sup_df = cv_calculator.get_projected_sup_data()
            
            # Update CV info
            self.cv_dimension = cv_calculator.get_cv_dimension()
            self.cv_labels = cv_calculator.get_labels()
            self.cv_type = cv_calculator.get_cv_type()
                
            if projected_train_df is not None:
                
                # Iterate over the trajectories used for training
                for traj_index in range(len(self.train_colvars_paths)):
                    
                    # Get the colvars, trajectory and topology files
                    colvars = self.train_colvars_paths[traj_index]
                    trajectory = self.train_trajectory_paths[traj_index] if self.train_trajectory_paths else None
                    topology = self.train_topology_paths[traj_index] if self.train_topology_paths else None

                    # Log
                    logger.info(f"Projecting colvars file: {colvars}")
                    logger.info(f"Corresponding trajectory file: {trajectory}")
                    logger.info(f"Corresponding topology file: {topology}")
                    
                    # Output folder for the current trajectory
                    traj_output_folder = os.path.join(cv_output_folder, Path(trajectory).stem)
                    os.makedirs(traj_output_folder, exist_ok=True)
                    
                    # Create plumed inputs for this CV and topology
                    cv_calculator.write_plumed_input(topology, traj_output_folder)
                    
                    # Get the projected data for this colvars file
                    projected_train_df_i = projected_train_df[projected_train_df['label'] == traj_index]
                    projected_train_df_i.drop('label', axis=1, inplace=True)
                    
                    # Get the supplementary data if any
                    sup_data = None
                    if projected_sup_df is not None:
                        sup_data = [projected_sup_df[projected_sup_df['label'] == index] for index in range(len(self.sup_colvars_paths))]
                        sup_data = [df.drop('label', axis=1).to_numpy() for df in sup_data]
                    
                    self.create_fes_plots(
                        data_df = projected_train_df_i,
                        output_folder = traj_output_folder,
                        sup_data = sup_data
                    )

                    # If clustering is enabled
                    if self.clustering_configuration['run']:
                        
                        # Cluster the input samples
                        cluster_labels, centroids = statistics.optimize_clustering(projected_train_df_i.to_numpy(), self.clustering_configuration)
                        
                        # Add a column with the order of the data points
                        projected_train_df_i['order'] = np.arange(projected_train_df_i.shape[0])
                        
                        # Add cluster labels to the projected input dataframe
                        projected_train_df_i['cluster'] = cluster_labels
                        
                        # Find centroids among input samples
                        if len(centroids) > 0:
                            centroids_df = statistics.find_centroids(projected_train_df_i, centroids, cv_calculator.get_labels())
                            
                        # Generate color map for the clusters
                        num_clusters = len(np.unique(cluster_labels))
                        cluster_colors = figures.generate_colors(num_clusters, self.figures_configuration['traj_projection']['cmap'])

                        # Plot cluster sizes 
                        figures.plot_clusters_size(cluster_labels, cluster_colors, traj_output_folder)
                        
                        # Extract clusters from the trajectory
                        if (None not in [trajectory, topology]) and (len(centroids) > 0):
                            md.extract_clusters_from_traj(trajectory_path = trajectory, 
                                                        topology_path = topology,
                                                        traj_df = projected_train_df_i,
                                                        samples_per_frame = self.samples_per_frame, 
                                                        centroids_df = centroids_df,
                                                        cluster_label = 'cluster',
                                                        frame_label = 'order', 
                                                        output_structures = self.clustering_configuration['output_structures'],
                                                        output_folder = os.path.join(traj_output_folder, 'clusters'))
                        elif len(centroids) == 0:
                            logger.warning("No centroids found. Skipping extraction of clusters from the trajectory.")
                        else:
                            logger.warning("Trajectory and/or topology files not provided. Skipping extraction of clusters from the trajectory.")
                            logger.debug(f"Trajectory: {trajectory}")
                            logger.debug(f"Topology: {topology}")
                    
                    # Add a column with the order of the data points
                    projected_train_df_i['order'] = np.arange(projected_train_df_i.shape[0])
                        
                    # 2D plots of the input data projected onto the CV space
                    if cv_calculator.get_cv_dimension() == 2:
                    
                        # Colored by order
                        figures.gradient_scatter_plot(
                            data = projected_train_df_i,
                            column_labels = cv_calculator.get_labels(),
                            color_label = 'order',
                            settings = self.figures_configuration['traj_projection'],
                            file_path = os.path.join(traj_output_folder,'trajectory.png'))
                    
                        # If clustering is enabled
                        if self.clustering_configuration['run']:
                            
                            # Colored by cluster
                            figures.clusters_scatter_plot(
                                data = projected_train_df_i, 
                                column_labels = cv_calculator.get_labels(),
                                cluster_label = 'cluster', 
                                settings = self.figures_configuration['traj_projection'], 
                                file_path = os.path.join(traj_output_folder,'trajectory_clustered.png'),
                                cluster_colors = cluster_colors)
                    
                    # Erase the order column
                    projected_train_df_i.drop('order', axis=1, inplace=True)
                    
                    # Save the projected input data
                    projected_train_df_i.to_csv(os.path.join(traj_output_folder,'projected_trajectory.csv'), index=False, float_format='%.4f')

                if projected_sup_df is not None:
                    
                    # Iterate over supplementary data
                    for traj_index in range(len(self.sup_colvars_paths)):
                        
                        # Get the colvars and topology files
                        sup_colvars = self.sup_colvars_paths[traj_index]
                        sup_topology = self.sup_topology_paths[traj_index] if self.sup_topology_paths else None
                        sup_label = self.sup_labels[traj_index]
                        
                        # Log
                        logger.info(f"Supplementary colvars file: {sup_colvars}")
                        logger.info(f"Corresponding topology file: {sup_topology}")
                    
                        # Output folder for this topology
                        sup_output_folder = os.path.join(cv_output_folder, f"sup_{sup_label}")
                        os.makedirs(sup_output_folder, exist_ok=True)
                        
                        # Create plumed inputs for this CV and topology
                        cv_calculator.write_plumed_input(sup_topology, sup_output_folder)
                
                        logger.debug(f'Saving supplementary data from {sup_colvars} to {sup_output_folder}')
                        
                        # Extract the projected data for this colvars file
                        projected_sup_colvars_df = projected_sup_df[projected_sup_df["label"] == traj_index]
                        projected_sup_colvars_df.drop('label', axis=1, inplace=True)
                        
                        # Plot FES of the supplementary data in the CV space
                        figures.plot_fes(
                            data = projected_sup_colvars_df.to_numpy(),
                            cv_labels = cv_calculator.get_labels(),
                            settings = self.figures_configuration['fes'],
                            output_path = sup_output_folder)
                        
                        # Save the projected sup data
                        projected_sup_colvars_df.to_csv(os.path.join(sup_output_folder,'projected_data.csv'), index=False, float_format='%.4f')
                
            else:
                logger.warning(f"Projected colvars dataframe is empty for {cv_name}. Skipping this CV.")
                continue