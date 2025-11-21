# Import necessary modules
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# Local imports
from deep_cartograph.modules.figures import figures
from deep_cartograph.yaml_schemas.traj_projection import TrajProjectionSchema
from deep_cartograph.modules.common import validate_configuration, files_exist 

from deep_cartograph.modules.plumed.colvars import create_dataframe_from_files
from deep_cartograph.modules.cv_learning import CVCalculator

# Set logger
logger = logging.getLogger(__name__)

class TrajProjectionWorkflow:
    """
    Class to train collective variables from colvars files.
    """
    def __init__(self, 
                 configuration: Dict, 
                 colvars_paths: List[str],
                 topologies: List[str],
                 trajectory_names: List[str],
                 model_paths: List[str],
                 model_traj_paths: Optional[List[List[str]]] = None,
                 output_folder: Optional[str] = 'traj_projection'):
        """
        Initializes the TrajProjectionWorkflow class.
        
        The class runs the traj_projection workflow, which consists of:
        
            1. Load previously trained CV models from model paths
            2. Project the colvars files onto the CV space defined by each model (coloring by frame index)
            3. Project the colvars files onto the FES given by the training data in model_traj_paths

        The output folder is organized as follows:
        
        traj_projection/
            cv_name_1/                          # e.g. pca/
                traj_data/                      # data related to the input trajectories 
                    trajectory_1/               # trajectory folder
                        fes/                    # FES plots and arrays                           
                            component_1_2
                        plumed_inputs/
                            unbiased_md.zip
                            biased_md.zip
                        projected_trajectory.csv
                        trajectory.png      

        Parameters
        ----------
        configuration : Dict
            Configuration dictionary (see `default_config.yml` for more information).

        colvars_paths : List[str]
            List of paths to the colvars files containing the input data from new trajectories to project (samples of features).

        topologies : List[str]
            List of paths to topologies of new trajectories to project
            
        trajectory_names : List[str]
            List of trajectory names corresponding to the input colvars files.
        
        model_paths : List[str]
            List of paths to the pre-trained collective variable model files.
        
        model_traj_paths : Optional[List[List[str]]]
            List of paths to the projected data in the CV space of the trajectories used for 
            training the collective variable model(s). These will be used to compute the background FES.

        output_folder : Optional[str]
            Path to the output folder where results will be saved. 
        """
        
        # Set output folders
        self.parent_output_folder: str = output_folder
        
        # Configuration related attributes
        self.configuration: Dict = validate_configuration(configuration, TrajProjectionSchema, output_folder)
        self.figures_configuration: Dict = self.configuration['figures']
                
        # Input related attributes
        self.colvars_paths: List[str] = colvars_paths
        self.topologies: List[str] = topologies
        self.trajectory_names: List[str] = trajectory_names
        self.model_paths: List[str] = model_paths
        self.model_traj_paths: List[List[str]] = model_traj_paths
        
        # Attributes related to the CVs
        self.cv_name: str = None
        self.cv_dimension: int = None
        self.cv_labels: List[str] = None

        # Validate inputs existence
        self._validate_files()
        
    def _validate_files(self):
        """Checks if provided input files exist."""
        
        for path in self.colvars_paths:
            if not files_exist(path):
                logger.error(f"Colvars file {path} does not exist. Exiting...")
                sys.exit(1)
        
            if self.topologies:
                for path in self.topologies:
                    if not files_exist(path):
                        logger.error(f"Topology file {path} does not exist. Exiting...")
                        sys.exit(1)
                        
                if len(self.topologies) != len(self.colvars_paths):
                    logger.error("Number of topologies must match number of colvars files. Exiting...")
                    sys.exit(1)

                if len(self.trajectory_names) != len(self.colvars_paths):
                    logger.error("Number of trajectory names must match number of colvars files. Exiting...")
                    sys.exit(1)
            else:
                logger.warning("No topologies provided. Some CVs may not work.")
            
        if self.model_paths:
            for path in self.model_paths:
                if not files_exist(path):
                    logger.error(f"CV model file {path} does not exist. Exiting...")
                    sys.exit(1)
    
    def create_fes_plots(self, 
                         main_data: pd.DataFrame, 
                         output_folder: str,
                         sup_data: Optional[List[np.ndarray]] = None,
                         sup_data_labels: Optional[List[str]] = None   
        ):
        """
        Create all the required FES plots
        
            1D plots for each components
            2D plots for each pair of components
        
        Inputs
        ------
        
        main_data: 
            Main data to compute the background FES
            
        output_folder:
            Directory where the FES will be saved
            
        sup_data:
            Optional supplementary data to show it as a scatter plot alongside the main data

        sup_data_labels:
            Optional labels for the supplementary data
        """
        
        # 1D plots for each component
        for dimension in range(self.cv_dimension):

            data_i = main_data.iloc[:,dimension]
            label_i = [self.cv_labels[dimension]]
            sup_data_i = [x[:,dimension] for x in sup_data] if sup_data else None

            fes_output_folder = os.path.join(output_folder, f'fes_{self.cv_name}_{dimension+1}')
            os.makedirs(fes_output_folder, exist_ok=True)
            
            # Plot FES
            figures.plot_fes(
                data = data_i.to_numpy(),
                cv_labels = label_i,
                settings = self.figures_configuration['fes'],
                output_path = fes_output_folder,
                num_blocks = 100,  
                sup_data = sup_data_i,
                sup_data_labels = sup_data_labels
            )
        
        # 2D plots for each pair of components
        if self.cv_dimension > 1:
            
            for i in range(0, self.cv_dimension-1):
                for j in range(i+1, self.cv_dimension):
                    
                    data_ij = main_data.iloc[:, [i,j]]
                    label_ij = [self.cv_labels[i], self.cv_labels[j]]
                    sup_data_ij = [x[:,[i,j]] for x in sup_data] if sup_data else None
    
                    fes_output_folder = os.path.join(output_folder, f'fes_{self.cv_name}_{i+1}_{j+1}')
                    os.makedirs(fes_output_folder, exist_ok=True)
                    
                    # Plot FES
                    figures.plot_fes(
                        data = data_ij.to_numpy(),
                        cv_labels = label_ij,
                        settings = self.figures_configuration['fes'],
                        output_path = fes_output_folder,
                        num_blocks = 1,
                        sup_data = sup_data_ij,
                        sup_data_labels = sup_data_labels
                    )
    
    def run(self) -> Dict[str, List[str]]:
        """
        Run the traj_projection workflow.
        
        Returns
        -------
        
        Dict[str, List[str]]
            Dictionary with the colvars paths trajectories in the CV space of each model in model_paths.

        """

        output_cv_data: Dict[str, List[str]] = {}

        logger.info("Starting traj_projection workflow...")

        # For each model, load the model and project the input data
        for model_index in range(len(self.model_paths)):

            model_path = self.model_paths[model_index]

            # Load the CV model
            cv_calculator = CVCalculator.load(
                model_path = model_path,
                output_path = self.parent_output_folder
            )
            self.cv_name = cv_calculator.cv_name
            self.cv_dimension = cv_calculator.cv_dimension
            self.cv_labels = cv_calculator.cv_labels
            cv_output_folder = os.path.join(self.parent_output_folder, self.cv_name)
            os.makedirs(cv_output_folder, exist_ok=True)
            
            # Append output paths dictionary
            output_cv_data[self.cv_name] = {'traj_paths': []}
            output_cv_data[self.cv_name]['traj_paths'] = [os.path.join(cv_output_folder, traj_name, 'projected_trajectory.csv') for traj_name in self.trajectory_names]

            # Check if the files already exist
            if files_exist(*output_cv_data[self.cv_name]['traj_paths']):
                logger.info(f"Projected trajectory files for CV {self.cv_name} already exist. Skipping projection...")
                continue
            
            # Project the input colvars file onto the CV space
            projected_data = cv_calculator.project_colvars(
                colvars_paths = self.colvars_paths,
                topology_paths = self.topologies
            )
            
            # Return file label
            projected_data['traj_label'] = cv_calculator.projection_data_labels

            # Divide into the different colvars paths, remove traj_label column after
            projected_data_list = [projected_data[projected_data['traj_label'] == file_index] for file_index in range(len(self.colvars_paths))]
            projected_data_list = [data.drop(columns=['traj_label']) for data in projected_data_list]
            
            # Scatter plots for each colvars
            if self.cv_dimension == 2:
                
                for index in range(len(self.colvars_paths)):
                    projected_data_i = projected_data_list[index]
                    projected_data_i['frame'] = np.arange(len(projected_data_i))
                    trajectory_name = self.trajectory_names[index]
                    
                    traj_output_folder = os.path.join(cv_output_folder, trajectory_name)
                    os.makedirs(traj_output_folder, exist_ok=True)
                    
                    # Plot scatter of projected data colored by frame number
                    figures.gradient_scatter_plot(
                        data = projected_data_i,
                        column_labels = self.cv_labels,
                        color_label = 'frame',
                        settings = self.figures_configuration['traj_projection'],
                        file_path = os.path.join(traj_output_folder, f'trajectory.png')
                    )
                    
                    # Erase the frame column
                    projected_data_i = projected_data_i.drop(columns=['frame'])
                    
                    # Save projected data
                    projected_data_i.to_csv(os.path.join(traj_output_folder, f'projected_trajectory.csv'), index=False, float_format='%.4f')

            projected_data_list = [data.to_numpy() for data in projected_data_list] 
            
            
            if self.model_traj_paths is not None:
                # Load training data for FES
                main_data = create_dataframe_from_files(self.model_traj_paths[model_index])
                # Plot FES of training data + scatter of all projected data
                self.create_fes_plots(
                    main_data = main_data,
                    output_folder = os.path.join(cv_output_folder, 'fes'),
                    sup_data = projected_data_list,
                    sup_data_labels = self.trajectory_names
                )
            
        return output_cv_data
                