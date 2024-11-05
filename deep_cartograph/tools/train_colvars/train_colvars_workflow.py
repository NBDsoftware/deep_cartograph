# Import necessary modules
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch, lightning
from pathlib import Path
from typing import List, Dict, Union, Literal

# Local imports
from deep_cartograph.modules.md import md
from deep_cartograph.modules.figures import figures
from deep_cartograph.modules.statistics import statistics
from deep_cartograph.yaml_schemas.train_colvars import TrainColvarsSchema
from deep_cartograph.modules.common import validate_configuration, create_output_folder, files_exist, merge_configurations 

from deep_cartograph.tools.train_colvars.cv_calculator import CVCalculator, cv_calculators_map

# Set logger
logger = logging.getLogger(__name__)

# NOTE: distinguish external and internal methods _bla bla

class TrainColvarsWorkflow:
    """
    Class to train collective variables from colvars files.
    """
    def __init__(self, 
                 configuration: Dict, 
                 colvars_path: str, 
                 feature_constraints: Union[List[str], str],
                 ref_colvars_path: Union[List[str], None] = None, 
                 ref_labels: Union[List[str], None] = None,
                 cv_dimension: Union[int, None] = None,
                 cvs: List[Literal['pca', 'ae', 'tica', 'deep_tica']] = None,
                 trajectory_path: Union[str, None] = None,
                 topology_path: Union[str, None] = None, 
                 output_folder: str = 'train_colvars'):
        """
        Initializes the TrainColvarsWorkflow class.
        
        The class runs the train_colvars workflow, which consists of:
        
            1. Calculating the collective variables and projecting the features onto the CV space.
            2. Plotting the FES of the CV space.
            3. Plotting the projected features onto the CV space colored by order.
            4. Clustering the input samples according to the distance in the CV space.
            5. Plotting the projected features onto the CV space colored by cluster.
            6. Extracting clusters from the trajectory.
        """
        
        # Set output folder
        self.output_folder: str = output_folder
        create_output_folder(output_folder)
        
        # Configuration related attributes
        self.configuration: Dict = validate_configuration(configuration, TrainColvarsSchema, output_folder)
        self.figures_configuration: Dict = self.configuration['figures']
        self.clustering_configuration: Dict = self.configuration['clustering']
                
        # Input related attributes
        self.colvars_path: str = colvars_path
        self.feature_constraints: Union[List[str], str] = feature_constraints
        self.ref_colvars_path: Union[List[str], None] = ref_colvars_path
        self.trajectory_path: Union[str, None] = trajectory_path
        self.topology_path: Union[str, None] = topology_path
        self.ref_labels: Union[List[str], None] = ref_labels
        
        # Validate inputs existence
        self._validate_files()

        # CV related attributes
        self.cvs: List[Literal['pca', 'ae', 'tica', 'deep_tica']] = cvs if cvs else self.configuration['cvs']
        self.cv_dimension: int = cv_dimension
        
    def _validate_files(self):
        """Checks if required input files exist."""
        
        if not files_exist(self.colvars_path):
            logger.error(f"Colvars file {self.colvars_path} does not exist. Exiting...")
            sys.exit(1)
            
        if self.ref_colvars_path: 
            if not files_exist(*self.ref_colvars_path):
                logger.error(f"Reference colvars file {self.ref_colvars_path} does not exist. Exiting...")
                sys.exit(1)
                
        if self.trajectory_path:
            if not files_exist(self.trajectory_path):
                logger.error(f"Trajectory file {self.trajectory_path} does not exist. Exiting...")
                sys.exit(1)
        
        if self.topology_path:
            if not files_exist(self.topology_path):
                logger.error(f"Topology file {self.topology_path} does not exist. Exiting...")
                sys.exit(1)
                
    def calculate_cv(self, cv: Literal['pca', 'ae', 'tica', 'deep_tica']) -> CVCalculator:
        """
        Runs the calculation of the requested collective variables using the corresponding calculator.
        
        Inputs
        ------
        
            cv:         The collective variable to calculate.
        
        Returns
        -------
        
            calculator: The CV calculator object.
        """
        
        # Merge common and cv-specific configuration for this cv 
        common_configuration = self.configuration['common']
        cv_specific_configuration = self.configuration.get(cv, {})
        cv_configuration = merge_configurations(common_configuration, cv_specific_configuration)
        
        # Construct the corresponding CV calculator
        calculator = cv_calculators_map[cv](self.colvars_path, 
                                    self.feature_constraints, 
                                    self.ref_colvars_path, 
                                    cv_configuration,
                                    self.output_folder)
        
        # Run the CV calculator
        calculator.run(self.cv_dimension)
        
        return calculator
    
    def run(self):
        """
        Run the train_colvars workflow.
        """
        
        logger.info(f"Collective variables to compute: {self.cvs}")
        
        # For each requested collective variable
        for cv in self.cvs:
            
            cv_output_folder = os.path.join(self.output_folder, cv)
            
            # Compute collective variable and project features onto the CV space
            cv_calculator = self.calculate_cv(cv)
            
            # Plot FES of the CV space - add ref data if available
            figures.plot_fes(
                X = cv_calculator.get_projected_input(),
                cv_labels = cv_calculator.get_labels(),
                X_ref = cv_calculator.get_projected_ref(),
                X_ref_labels = self.ref_labels,
                settings = self.figures_configuration,
                output_path = cv_output_folder)
            
            # Create a dataframe with the projected input data
            projected_input_df = pd.DataFrame(cv_calculator.get_projected_input(), 
                                                   columns=cv_calculator.get_labels())
            
            # Add a column with the order of the data points
            projected_input_df['order'] = np.arange(projected_input_df.shape[0])

            # If clustering is enabled
            if self.clustering_configuration['run']:
                
                # Cluster the input samples
                cluster_labels, centroids = statistics.optimize_clustering(cv_calculator.get_projected_input(), 
                                                                           self.clustering_configuration)
                
                # Add cluster labels to the projected input dataframe
                projected_input_df['cluster'] = cluster_labels
                
                # Find centroids among input samples
                if len(centroids) > 0:
                
                    centroids_df = statistics.find_centroids(projected_input_df, centroids, cv_calculator.get_labels())
                    
                # Generate color map for clusters
                num_clusters = len(np.unique(cluster_labels))
                cmap = figures.generate_cmap(num_clusters, self.figures_configuration['projected_clustered_trajectory']['cmap'])

                # Plot cluster sizes 
                figures.plot_clusters_size(cluster_labels, cmap, cv_output_folder)
                
                # Extract clusters from the trajectory
                if (None not in [self.trajectory_path, self.topology_path]) and (len(centroids) > 0):
                    md.extract_clusters_from_traj(trajectory_path = self.trajectory_path, 
                                                topology_path = self.topology_path, 
                                                traj_df = projected_input_df, 
                                                centroids_df = centroids_df,
                                                cluster_label = 'cluster',
                                                frame_label = 'order', 
                                                output_folder = os.path.join(cv_output_folder, 'clustered_traj'))
                elif len(centroids) == 0:
                    logger.warning("No centroids found. Skipping extraction of clusters from the trajectory.")
                else:
                    logger.warning("Trajectory and/or topology files not provided. Skipping extraction of clusters from the trajectory.")
                    logger.debug(f"Trajectory: {self.trajectory_path}")
                    logger.debug(f"Topology: {self.topology_path}")
            
            # 2D plots of the input data projected onto the CV space
            if cv_calculator.get_cv_dimension() == 2:
                
                # Colored by order
                figures.plot_projected_trajectory(
                    data_df = projected_input_df,
                    axis_labels = cv_calculator.get_labels(),
                    frame_label = 'order',
                    settings = self.figures_configuration['projected_trajectory'],
                    file_path = os.path.join(cv_output_folder,'trajectory.png'))
            
                # If clustering is enabled
                if self.clustering_configuration['run']:
                    
                    # Colored by cluster
                    figures.plot_clustered_trajectory(
                        data_df = projected_input_df, 
                        axis_labels = cv_calculator.get_labels(),
                        cluster_label = 'cluster', 
                        settings = self.figures_configuration['projected_clustered_trajectory'], 
                        file_path = os.path.join(cv_output_folder,'trajectory_clustered.png'),
                        cmap = cmap)
            
            # Erase the order column
            projected_input_df.drop('order', axis=1, inplace=True)
            
            # Save the projected input data
            projected_input_df.to_csv(os.path.join(cv_output_folder,'projected_trajectory.csv'), index=False, float_format='%.4f')
        
        
