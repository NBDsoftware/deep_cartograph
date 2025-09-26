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
from deep_cartograph.modules.figures import figures
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
                 train_topology_paths: Optional[List[str]] = None,
                 trajectory_names: Optional[List[str]] = None,
                 ref_topology_path: Optional[str] = None,           
                 features_list: Optional[List[str]] = None,
                 cv_dimension: Optional[int] = None,
                 cvs: Optional[List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']]] = None,
                 frames_per_sample: Optional[int] = 1,
                 output_folder: str = 'train_colvars'):
        """
        Initializes the TrainColvarsWorkflow class.
        
        The class runs the train_colvars workflow, which consists of:
        
            1. Use train_colvars_paths files to compute the collective variables.
            2. Plotting the FES of each colvars file in the CV space. 
            3. Plotting the trajectory onto the CV space colored by frame number.
            
        The output folder is organized as follows:
        
        train_colvars/
            cv_name_1/                          # e.g. pca/
                traj_data/                      # data related to the input trajectories 
                    trajectory_1/               # trajectory folder
                        fes/                    # FES plots and arrays                           
                            component_1/
                            component_1_2/
                        plumed_inputs/
                            unbiased_md.zip
                            opes.zip
                            metad.zip
                            ...
                        projected_trajectory.csv
                        trajectory.png       
                
                sensitivity_analysis/           # sensitivity analysis results        
                training/                       # Training data and model scores
                    checkpoints/
                model.zip                       # Trained model

        """
        
        # Set output folder
        self.output_folder: str = output_folder
        
        # Configuration related attributes
        self.configuration: Dict = validate_configuration(configuration, TrainColvarsSchema, output_folder)
        self.figures_configuration: Dict = self.configuration['figures']
                
        # Input related attributes
        self.train_colvars_paths: List[str] = train_colvars_paths
        self.train_topology_paths: Optional[List[str]] = train_topology_paths
        self.trajectory_names: Optional[List[str]] = trajectory_names if trajectory_names else [Path(f).stem for f in train_colvars_paths]
        self.ref_topology_path: Optional[str] = ref_topology_path
        self.features_list: Optional[List[str]] = features_list

        if self.train_topology_paths:
            if self.ref_topology_path is None:
                self.ref_topology_path = self.train_topology_paths[0]
        
        self.frames_per_sample: int = frames_per_sample if frames_per_sample else 1
        
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
                sup_data = sup_data_i)
        
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
                        sup_data = sup_data_ij)

    def get_cvs_list(self) -> List[str]:
        """
        Returns the list of collective variables to compute.
        """
        return self.cvs_list
    
    def check_cv_trajectories(self, cv_name: str) -> bool:
        """
        Check if the trajectory along the given cv has been computed.
        """
        
        # Get the trajectory paths along the given cv
        traj_paths = self.get_cv_trajectories(cv_name)

        cv_trajs_exist = files_exist(*traj_paths, verbose=False)

        return cv_trajs_exist

    def get_cv_trajectories(self, cv_name: str) -> List[str]:
        """
        Returns the list of trajectories along the given cv.
        """
        cv_output_folder = os.path.join(self.output_folder, cv_name)
        traj_data_folder = os.path.join(cv_output_folder, 'traj_data')
        
        traj_paths = []
        for traj_index in range(len(self.train_colvars_paths)):
            traj_output_folder = os.path.join(traj_data_folder, self.trajectory_names[traj_index])
            csv_path = os.path.join(traj_output_folder,'projected_trajectory.csv')
            traj_paths.append(csv_path)

        return traj_paths
    
    def run(self):
        """
        Run the train_colvars workflow.
        """
        
        logger.info(f"Collective variables to compute: {self.cvs_list}")
        
        # For each requested collective variable
        for cv_name in self.cvs_list:
            
            if cv_name != "pca" and not package_is_installed('mlcolvar', 'torch', 'lightning'):
                logger.warning(f"Missing packages for {cv_name}. Skipping this CV. If you want to use it, please install mlcolvar, torch and lightning.")
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
                'features_list': self.features_list,
                'output_path': self.output_folder
            }
            cv_calculator = cv_calculators_map[cv_name](**args)
            
            # Run the CV calculator - obtain a dataframe with the projected training data
            projected_train_df = cv_calculator.run(self.cv_dimension)
            
            # Update CV info
            self.cv_dimension = cv_calculator.get_cv_dimension()
            self.cv_labels = cv_calculator.get_labels()
            self.cv_type = cv_calculator.get_cv_type()
                
            if projected_train_df is not None:
                
                # Iterate over the trajectories used for training
                for traj_index in range(len(self.train_colvars_paths)):
                    
                    # Get the colvars, trajectory and topology files
                    colvars = self.train_colvars_paths[traj_index]
                    topology = self.train_topology_paths[traj_index] if self.train_topology_paths else None
                    traj_name = self.trajectory_names[traj_index]

                    # Log
                    logger.info(f"Processing trajectory: {traj_name}")
                    logger.info(f"Corresponding colvars file: {colvars}")
                    logger.info(f"Corresponding topology file: {topology}")
                    
                    # Output folder for the current trajectory
                    traj_output_folder = os.path.join(cv_output_folder, 'traj_data', traj_name)
                    os.makedirs(traj_output_folder, exist_ok=True)
                    
                    # Create plumed inputs for this CV and topology
                    plumed_inputs_folder = os.path.join(traj_output_folder, 'plumed_inputs')
                    os.makedirs(plumed_inputs_folder, exist_ok=True)
                    cv_calculator.write_plumed_files(topology, plumed_inputs_folder)

                    # Get the projected data for this colvars file
                    projected_train_df_i = projected_train_df[projected_train_df['traj_label'] == traj_index]
                    projected_train_df_i.drop('traj_label', axis=1, inplace=True)
                    
                    fes_output_folder = os.path.join(traj_output_folder, 'fes')
                    self.create_fes_plots(
                        data_df = projected_train_df_i,
                        output_folder = fes_output_folder
                    )
                    
                    # Add a column with the frame of each sample
                    projected_train_df_i['frame'] = np.arange(0, len(projected_train_df_i)) * self.frames_per_sample

                    # 2D plots of the input data projected onto the CV space
                    if cv_calculator.get_cv_dimension() == 2:

                        # Colored by frame
                        figures.gradient_scatter_plot(
                            data = projected_train_df_i,
                            column_labels = cv_calculator.get_labels(),
                            color_label = 'frame',
                            settings = self.figures_configuration['traj_projection'],
                            file_path = os.path.join(traj_output_folder,'trajectory.png'))

                    # Erase the frame column
                    projected_train_df_i.drop('frame', axis=1, inplace=True)
                    
                    # Save the projected input data
                    projected_train_df_i.to_csv(os.path.join(traj_output_folder,'projected_trajectory.csv'), index=False, float_format='%.4f')

            else:
                logger.warning(f"Projected colvars dataframe is empty for {cv_name}. Skipping this CV.")
                continue