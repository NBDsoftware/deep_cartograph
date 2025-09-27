import os
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import block_diag
from sklearn.decomposition import PCA       
from typing import Dict, List, Tuple, Union, Literal, Optional

from deep_cartograph.modules.plumed.colvars import create_dataframe_from_files
import deep_cartograph.modules.md as md

# Set logger
logger = logging.getLogger(__name__)

# Base class for collective variables calculators
class CVCalculator:
    """
    Base class for collective variables calculators.
    """
    def __init__(self, 
        configuration: Dict,
        output_path: Optional[str] = None
        ):
        """
        Initializes the base CV calculator.
        
        Parameters
        ----------
    
        configuration
            Configuration dictionary with settings for the CV

        output_path (Optional)
            Output path where the CV results folder will be created
        """
        
        # Configuration
        self.configuration: Dict = configuration
        self.architecture_config: Dict = configuration['architecture']
        self.training_reading_settings: Dict = configuration['input_colvars']
        self.feats_norm_mode: Literal['mean_std', 'min_max_range1', 'min_max_range2', 'none'] = configuration['features_normalization']
        self.bias: Dict = configuration['bias']

        # Colvars paths
        self.train_colvars_paths: Optional[List[str]] = None

        # Topologies
        self.topologies: Optional[List[str]] = None
        self.ref_topology_path: Optional[str] = None
        
        # Training data
        self.training_data: Optional[pd.DataFrame] = None
        self.training_data_labels: Optional[np.array] = None
        self.projected_training_data: Optional[pd.DataFrame] = None # NOTE: Is this needed?

        # Number of samples
        self.num_samples: int = None
        self.num_training_samples: Union[int, None] = None
    
        # Features
        self.features_ref_labels: List[str] = []
        self.features_stats: Dict[str, np.array] = {}
        self.features_norm_mean: np.array = None
        self.features_norm_range: np.array = None
        self.num_features: int = 0
        
        # General CV attributes
        self.cv_dimension: int = configuration['dimension']
        self.cv_labels: List[str] = []
        self.cv_name: str = None
        self.cv_range: List[Tuple[float, float]] = [] 

        # Parent output path
        self.parent_output_path: str = output_path
        
        # Plumed files - used to construct plumed zip files
        self.plumed_files: List[str] = []
    
    def load_model(self, 
        input_path: str
        ):
        """
        Loads a previously saved CV model from a zip file.
        Here we load just the common files to all CV calculators.
        Mainly the cv name, features labels and reference topology.
        
        Parameters
        ----------
        
        model_path : str
            Path to the zip file containing the CV model
        """
        
        from deep_cartograph.modules.common import unzip_files # NOTE: todo
        
        # Unzip the model files to a temporary folder
        temp_folder = os.path.join(self.parent_output_path, 'temp_model')
        unzip_files(input_path, temp_folder)

        # Load the cv name
        with open(os.path.join(temp_folder, 'cv_name.txt'), 'r') as f:
            self.cv_name = f.read().strip()
            
        # Move the temporary folder to the model output folder
        self.model_output_folder = os.path.join(self.parent_output_path, self.cv_name, 'model')
        shutil.move(temp_folder, self.model_output_folder)
        
        # Load the list of features reference labels used to compute the CV
        with open(os.path.join(self.model_output_folder, 'features_labels.txt'), 'r') as f:
            self.features_ref_labels = f.read().strip().split('\n')
            self.num_features = len(self.features_ref_labels)
        
        # Load the reference topology used to compute the CV
        ref_topology_path = os.path.join(self.model_output_folder, 'ref_topology.pdb')
        if os.path.exists(ref_topology_path):
            self.ref_topology_path = ref_topology_path
        else:
            self.ref_topology_path = None
            logger.warning('''Reference topology file not found in the model. 
                           Make sure to use the same topologies as the one used to train the CV.''')

    def create_output_folders(self):
        """ 
        Creates the output folders for this CV.

        Used after the specific CV calculator constructor has been called.
        """
        
        # Create output folder for this CV
        parent_path = Path(self.parent_output_path)
        self.output_path = parent_path / self.cv_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create output folders to sensitivity_analysis, training and model
        self.sensitivity_output_folder = self.output_path / 'sensitivity_analysis'
        self.sensitivity_output_folder.mkdir(parents=True, exist_ok=True)
        self.training_output_folder = self.output_path / 'training'
        self.training_output_folder.mkdir(parents=True, exist_ok=True)
        self.model_output_folder = self.output_path / 'model'
        self.model_output_folder.mkdir(parents=True, exist_ok=True)

    def load_training_data(self,
        train_colvars_paths: List[str],
        train_topology_paths: Optional[List[str]] = None, 
        ref_topology_path: Optional[str] = None, 
        features_list: Optional[List[str]] = None, 
        ):
        """
        Loads the training data from the colvars files
    
        Parameters
        ----------

        train_colvars_paths 
            List of paths to colvars files with the main data used for training
                
        train_topology_paths (Optional)
            List of paths to topology files corresponding to the training colvars files (same order)
        
        ref_topology_path (Optional)
            Path to the reference topology file. If None, the first topology file is used as reference topology
            
        features_list (Optional)
            List with the features to use for the training (names from reference topology) or None to use all features in the colvars files
        """

        # Colvars paths
        self.train_colvars_paths = train_colvars_paths
        
        # Topologies
        self.topologies = train_topology_paths
        self.ref_topology_path = ref_topology_path
        if self.topologies is not None:
            if self.ref_topology_path is None:
                self.ref_topology_path = self.topologies[0]
        
        # Filtered training data used to train / compute the CVs
        logger.info('Reading training data from colvars files...')    
        self.training_data = create_dataframe_from_files( 
            colvars_paths = self.train_colvars_paths,
            topology_paths = self.topologies,
            reference_topology = self.ref_topology_path,   
            features_list = features_list,  
            file_label = 'traj_label',
            **self.training_reading_settings
        )
        
        self.training_data_labels = self.training_data.pop('traj_label').to_numpy()
        
        # List of features used for training (features in the colvars files after filtering) 
        self.features_ref_labels: List[str] = self.training_data.columns.tolist()
        self.num_features: int = len(self.features_ref_labels)
        logger.info(f'Number of features: {self.num_features}')
        
        # Compute training data statistics
        stats = ['mean', 'std', 'min', 'max']
        stats_df = self.training_data.agg(stats).T
        self.features_stats = {stat: stats_df[stat].to_numpy() for stat in stats}
        self.features_norm_mean, self.features_norm_range = self.prepare_normalization()
        
        # Get the total number of samples
        self.num_samples = len(self.training_data)
        logger.info(f'Number of samples: {self.num_samples}')

    def cv_ready(self) -> bool:
        """
        Checks if the CV is ready to be used.
        """
        
        return self.cv is not None
        
    def prepare_normalization(self) -> Tuple[np.array, np.array]:

        """ 
        Prepare the normalization parameters for the features. Computes the normalization
        means and ranges based on the feature statistics and the chosen normalization mode.
        
        The normalization will be:

                              feature - normalization_mean
        normalized feature = --------------------------------
                                    normalization_range
        
        Returns
        -------
        
        means : np.array
            Means for normalization
        ranges : np.array
            Ranges for normalization
        """
        
        def sanitize_ranges(range_array: np.ndarray):
            """
            Check the ranges are not close to zero and set them to 1 if they are.
            """
            for i in range(len(range_array)):
                if abs(range_array[i]) < 1e-8:
                    range_array[i] = 1.0
                    logger.warning(f'Range for feature {i} is close to zero. Setting it to 1.0.')
            return range_array
            
        # Set the mean and range for the normalization
        # No normalization of input features (not recommended)
        if self.feats_norm_mode == 'none':
            means = np.zeros(len(self.features_stats["mean"]))
            ranges = np.ones(len(self.features_stats["mean"]))
        # Normalized data will have mean 0 and std 1, non-defined range.
        elif self.feats_norm_mode == 'mean_std':
            means = self.features_stats["mean"]
            ranges = self.features_stats["std"]
        # Normalized data will have a range of [0, 1]
        elif self.feats_norm_mode == 'min_max_range1':
            means = self.features_stats["min"]
            ranges = self.features_stats["max"] - self.features_stats["min"]
        # Normalized data will have a range of [-1, 1]
        elif self.feats_norm_mode == 'min_max_range2':
            means = (self.features_stats["min"] + self.features_stats["max"]) / 2
            ranges = (self.features_stats["max"] - self.features_stats["min"]) / 2
        else:
            logger.error(f'Normalization mode {self.feats_norm_mode} not recognized. Exiting...')
            raise ValueError(f'Normalization mode {self.feats_norm_mode} not recognized.')
        
        # Check the ranges are not close to zero
        ranges = sanitize_ranges(ranges)
        
        return means, ranges
    
    # Main CV-related methods
    def run(self, cv_dimension: Union[int, None] = None) -> Union[pd.DataFrame, None]:
        """
        Runs the CV calculator.
        
        Parameters
        ----------
        
        cv_dimension : int
            Dimension of the CV. If None, the dimension from the configuration is used.
        
        Returns
        -------
        
        projected_training_data : pd.DataFrame
            Projected training data or None if the CV computation failed
        """
        
        # Overwrite the dimension from the configuration if provided
        if cv_dimension:
            self.cv_dimension = cv_dimension
        
        # Compute the CV weights using the training data
        self.compute_cv()
        
        self.set_labels()
       
        # If the CV was computed successfully
        if self.cv is not None:
            
            self.normalize_cv()

            self.projected_training_data = self.project_data(self.training_data)

            # Set the cv labels to the projected training data
            self.projected_training_data.columns = self.cv_labels
            
            # Return file labels to the projected training data
            self.projected_training_data['traj_label'] = self.training_data_labels
            
            self.save_model()
            
            self.sensitivity_analysis()

        return self.projected_training_data
        
    def compute_cv(self):
        """
        Computes the collective variables. Implement in subclasses.
        """
        
        raise NotImplementedError

    def save_weights(self, weights_path: str):
        """
        Saves the collective variable to a text file. Implement in subclasses.
        
        Parameters
        ----------
        
        weights_path : str
            Path to the output file where the weights will be saved
        """
        
        raise NotImplementedError   
    
    def save_model(self):
        """
        Saves the collective variable to a zip file. Here we save the files common to all CV calculators.
        """
        # Save the CV name into a text file
        with open(os.path.join(self.model_output_folder, 'cv_name.txt'), 'w') as f:
            f.write(self.cv_name)
        
        # Save the list of features reference labels used to compute the CV
        np.savetxt(os.path.join(self.model_output_folder, 'features_labels.txt'), self.features_ref_labels, fmt='%s')
            
        # Save the reference topology used to compute the CV
        if self.ref_topology_path is not None:
            md.create_pdb(self.ref_topology_path, os.path.join(self.model_output_folder, 'ref_topology.pdb'))

    def get_cv_parameters(self) -> Dict:
        """
        Returns the parameters for the CV. Implement in subclasses.
        """
        
        raise NotImplementedError
    
    def get_cv_type(self) -> str:
        """
        Returns the type of the CV. Implement in subclasses.
        """
        
        raise NotImplementedError
    
    def project_data(self, data: Optional[pd.DataFrame]) -> Union[pd.DataFrame, None]:
        """
        Projects the data onto the CV space. Implement in subclasses.
        """
        
        raise NotImplementedError
          
    def set_labels(self):
        """
        Sets the labels of the CV.
        """
        
        self.cv_labels = [f'{cv_components_map[self.cv_name]} {i+1}' for i in range(self.cv_dimension)]
    
    def normalize_cv(self):
        """
        Min max normalization of the CV.
        Normalizes the collective variable space to the range [-1, 1]
        Using the min and max values from the evaluation of the training data.
        Implemented in subclasses.
        """
        
        raise NotImplementedError

    def write_plumed_files(self, topology: str, output_folder: str) -> None:
        """
        Creates all files needed to compute the collective variable from the features
        for the given topology using plumed.
        
        Parameters
        ----------
        
        topology : str
            Path to the topology file of the system (used to translate the features)
            
        output_folder : str
            Path to the output folder where the files will be written
        """
        from deep_cartograph.modules.plumed.input.builder import ComputeCVBuilder, ComputeEnhancedSamplingBuilder
        from deep_cartograph.modules.plumed.features import FeatureTranslator
        from deep_cartograph.modules.md import create_plumed_rmsd_template
        
        from deep_cartograph.modules.common import zip_files, remove_files
        
        # Reset plumed files list
        self.plumed_files = []

        # Save new PLUMED-compliant topology for this trajectory - we try to avoid the 
        # limitations of the PLUMED PDB parser going through MDAnalysis :)
        plumed_topology_path = os.path.join(output_folder, 'plumed_topology.pdb')
        md.create_pdb(topology, plumed_topology_path)
        self.plumed_files.append(plumed_topology_path)
        
        # Translate the features from the reference topology to this system's topology
        ref_plumed_topology_path = os.path.join(output_folder, 'ref_plumed_topology.pdb')
        md.create_pdb(self.ref_topology_path, ref_plumed_topology_path)
        features_list = FeatureTranslator(ref_plumed_topology_path, plumed_topology_path, self.features_ref_labels).run()
        
        # If features contain coordinates, we need to fit the structure to the reference topology
        need_fit_template = any(feat.startswith("coord") for feat in features_list)
        if need_fit_template:
            fit_template_path = os.path.join(output_folder, "fit_template.pdb")
            create_plumed_rmsd_template(self.ref_topology_path, fit_template_path)
            self.plumed_files.append(fit_template_path)
        else:
            fit_template_path = None

        # If the CV is non-linear, add the pytorch model file
        if self.get_cv_type() == 'non-linear':
            # Save the model weights and add to the plumed files
            self.weights_path = os.path.join(output_folder, f'{self.cv_name}_weights.pt')
            self.save_weights(self.weights_path)
            self.plumed_files.append(self.weights_path)

        # Build the plumed input file to track the CV
        plumed_input_path = os.path.join(output_folder, f'plumed_input_{self.cv_name}.dat')
        self.plumed_files.append(plumed_input_path)
        builder_args = {
            'plumed_input_path': plumed_input_path,
            'topology_path': plumed_topology_path,
            'features_list': features_list,
            'traj_stride': 1,
            'cv_type': self.get_cv_type(),
            'cv_params': self.get_cv_parameters(),
            'fit_template_path': fit_template_path
        }
        plumed_builder = ComputeCVBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_out.dat')
        
        # Zip all the accumulated plumed files into a single file
        unbiased_plumed_path = os.path.join(output_folder, f'plumed_{self.cv_name}_unbiased.zip')
        zip_files(unbiased_plumed_path, *self.plumed_files)

        # Remove previous plumed input file
        os.remove(plumed_input_path)
        self.plumed_files.remove(plumed_input_path)
        
        # Build the plumed input file to perform enhanced sampling
        plumed_input_path = os.path.join(output_folder, f'plumed_input_{self.cv_name}_{self.bias["method"]}.dat')
        self.plumed_files.append(plumed_input_path)
        builder_args.update({
            'sampling_method': self.bias["method"], 
            'sampling_params': self.bias["args"],
            'plumed_input_path': plumed_input_path
        })
        plumed_builder = ComputeEnhancedSamplingBuilder(**builder_args)
        plumed_builder.build(f'{self.cv_name}_{self.bias["method"]}_out.dat')
        
        # Zip all the accumulated plumed files into a single file
        biased_plumed_path = os.path.join(output_folder, f'plumed_{self.cv_name}_biased.zip')
        zip_files(biased_plumed_path, *self.plumed_files)

        # Remove previous plumed files
        remove_files(*self.plumed_files)
        
        # Erase the temporary reference topology
        os.remove(ref_plumed_topology_path)
        
    def sensitivity_analysis(self):
        """
        Perform a sensitivity analysis of the CV on the training data.
        Implemented in subclasses.
        """
        raise NotImplementedError("Sensitivity analysis not implemented for this CV calculator.")
    
    # Getters  
    def get_labels(self) -> List[str]:
        """
        Returns the labels of the collective variable.
        """
        
        return self.cv_labels
    
    def get_cv_dimension(self) -> int:
        """
        Returns the dimension of the collective variables.
        """
        
        return self.cv_dimension

    def get_range(self) -> List[Tuple[float, float]]:
        """
        Returns the limits of the collective variable.
        """
        
        return self.cv_range


# Types of Collective Variable calculators
class LinearCalculator(CVCalculator):
    """
    Base class for linear collective variable calculators
    """
    
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """ 
        Initializes a linear CV calculator.
        """
        super().__init__(
            configuration, 
            output_path)
                
        # Main attributes
        self.cv: Union[np.array, None] = None 
        self.weights_path: Union[str, None] = None 
        
        self.cv_stats: Dict[str, np.array] = {}
        self.cv_norm_mean: np.array = None
        self.cv_norm_range: np.array = None

    def load_model(self, 
        input_path: str
        ):
        """ 
        Loads a previously saved CV model from a zip file.
        Here we load the files specific to linear CV calculators.
        Mainly the cv weights, the cv normalization parameters and the features normalization parameters.
        """
        super().load_model(input_path)
        
        # Load the cv weights
        weights_path = os.path.join(self.model_output_folder, 'cv_weights.npy')
        if not os.path.exists(weights_path):
            logger.error(f'CV weights file not found in the model: {weights_path}')
            raise FileNotFoundError(f'CV weights file not found in the model: {weights_path}')
        self.cv = np.load(weights_path)
        
        # Load the cv normalization parameters
        cv_norm_mean_path = os.path.join(self.model_output_folder, 'cv_norm_mean.npy')
        cv_norm_range_path = os.path.join(self.model_output_folder, 'cv_norm_range.npy')
        if not os.path.exists(cv_norm_mean_path) or not os.path.exists(cv_norm_range_path):
            logger.error('CV normalization parameters not found in the model.')
            raise FileNotFoundError('CV normalization parameters not found in the model.')
        self.cv_norm_mean = np.load(cv_norm_mean_path)
        self.cv_norm_range = np.load(cv_norm_range_path)

        # Load the feature normalization parameters
        features_norm_mean_path = os.path.join(self.model_output_folder, 'features_norm_mean.npy')
        features_norm_range_path = os.path.join(self.model_output_folder, 'features_norm_range.npy')
        if not os.path.exists(features_norm_mean_path) or not os.path.exists(features_norm_range_path):
            logger.error('Features normalization parameters not found in the model.')
            raise FileNotFoundError('Features normalization parameters not found in the model.')
        self.features_norm_mean = np.load(features_norm_mean_path)
        self.features_norm_range = np.load(features_norm_range_path)

    def load_training_data(self, train_colvars_paths, train_topology_paths = None, ref_topology_path = None, features_list = None):
        super().load_training_data(train_colvars_paths, train_topology_paths, ref_topology_path, features_list)
        
        # Normalize the training data
        self.training_data: pd.DataFrame = self.normalize_data(self.training_data, self.features_norm_mean, self.features_norm_range)
        
    def normalize_data(self, 
                       data: pd.DataFrame,
                       normalizing_mean: np.array,
                       normalizing_range: np.array,
        ) -> pd.DataFrame:
        """
        Use the normalization mean and range to normalize the data.
        
        Parameters
        ----------
        
        data : pd.DataFrame
            Data to normalize
        
        normalizing_mean : np.array
            Mean values for normalization
        
        normalizing_range : np.array
            Range values for normalization
            
        Returns
        -------
        
        normalized_data : pd.DataFrame
            Normalized data
        """

        return (data - normalizing_mean) / normalizing_range
    
    def save_weights(self, weights_path: str):
        """
        Saves the collective variable linear weights to a text file.
        
        Parameters
        ----------
        
        weights_path : str
            Path to the output file where the weights will be saved
        """
        
        if self.cv is None:
            logger.error('CV has not been computed. Cannot save weights.')
            raise ValueError('CV has not been computed. Cannot save weights.')
            
        np.save(weights_path, self.cv)
        logger.debug(f'CV weights saved to {weights_path}')
        
    def save_model(self):
        """
        Saves the collective variable linear weights and normalization parameters to a zip file.
        """
        super().save_model()
        
        from deep_cartograph.modules.common import zip_files
        
        # Save the cv weights
        weights_path = os.path.join(self.model_output_folder, 'cv_weights.npy')
        self.save_weights(weights_path)

        # Check the cv normalization parameters have been computed
        if self.cv_norm_mean is None or self.cv_norm_range is None:
            logger.error('CV normalization parameters have not been computed. Cannot save model.')
            raise ValueError('CV normalization parameters have not been computed. Cannot save model.')
        np.save(os.path.join(self.model_output_folder, 'cv_norm_mean.npy'), self.cv_norm_mean)
        np.save(os.path.join(self.model_output_folder, 'cv_norm_range.npy'), self.cv_norm_range)

        # Check the features normalization parameters have been computed
        if self.features_norm_mean is None or self.features_norm_range is None:
            logger.error('Features normalization parameters have not been computed. Cannot save model.')
            raise ValueError('Features normalization parameters have not been computed. Cannot save model.')
        np.save(os.path.join(self.model_output_folder, 'features_norm_mean.npy'), self.features_norm_mean)
        np.save(os.path.join(self.model_output_folder, 'features_norm_range.npy'), self.features_norm_range)

        # Zip the model output folder
        model_path = os.path.join(self.output_path, 'model.zip')
        zip_files(model_path, self.model_output_folder)

        # Remove the unzipped model output folder
        shutil.rmtree(self.model_output_folder)

        logger.info(f'Model saved to {model_path}')

    def get_cv_parameters(self):
        """ 
        Get the collective variable parameters.
        """

        # Save CV data to parameters dictionary
        cv_parameters = {
            'cv_name': self.cv_name,
            'cv_dimension': self.cv_dimension,
            'features_norm_mode': self.feats_norm_mode,
            'features_norm_mean': self.features_norm_mean,
            'features_norm_range': self.features_norm_range,
            'cv_stats': self.cv_stats,
            'weights': self.cv
        }
        return cv_parameters
    
    def get_cv_type(self) -> str:
        """
        Returns the type of the collective variable.
        """
        
        return 'linear'
    
    def project_data(self, data: Optional[pd.DataFrame]) -> Union[pd.DataFrame, None]:
        """
        Projects the data onto the normalized CV space.
        """
        
        if data is None:
            return None

        logger.debug(f"Projecting data onto {cv_names_map[self.cv_name]} ...")
        
        # Check the cv has been computed
        if self.cv is None:
            logger.error('CV has not been computed. Cannot project data.')
            raise ValueError('CV has not been computed. Cannot project data.')
            
        projected_data = data @ self.cv

        # Check the CV normalization parameters have been computed
        if self.cv_norm_mean is None or self.cv_norm_range is None:
            logger.error('CV normalization parameters have not been computed. Cannot normalize projected data.')
            raise ValueError('CV normalization parameters have not been computed. Cannot normalize projected data.')

        # Normalize the projected data
        projected_data = self.normalize_data(projected_data, self.cv_norm_mean, self.cv_norm_range)

        return projected_data
            
    def normalize_cv(self):
        
        # Project the normalized training data onto the CV space
        projected_training_data = pd.DataFrame(self.training_data @ self.cv) # NOTE: is this needed or already a dataframe?
        
        # Compute statistics of the projected training data
        stats = ['min', 'max']
        stats_df = projected_training_data.agg(stats).T 
        self.cv_stats = {stat: stats_df[stat].to_numpy() for stat in stats}
        
        # Max min normalization between -1 and 1
        self.cv_norm_mean = (self.cv_stats['max'] + self.cv_stats['min']) / 2
        self.cv_norm_range = (self.cv_stats['max'] - self.cv_stats['min']) / 2

    def sensitivity_analysis(self):
        """  
        Perform a sensitivity analysis of the CV on the training data.
        """
        
        from deep_cartograph.modules.figures import plot_sensitivity_results
        
        # For a linear CV, the sensitivity of each feature is given by the absolute value of its coefficient in the CV weights
        cv_sensitivities = np.abs(self.cv)
        
        # For each dimension of the CV
        for cv_index in range(cv_sensitivities.shape[1]):
            
            # Create directory for sensitivity analysis results
            sensitivity_output_path = os.path.join(self.sensitivity_output_folder, f'sensitivity_analysis_{cv_index+1}')
            os.makedirs(sensitivity_output_path, exist_ok=True)
            
            sensitivities = cv_sensitivities[:, cv_index]
            logger.debug(f'Shape of sensitivities for CV dimension {cv_index}: {sensitivities.shape}')

            # Order the sensitivities from lowest to highest, order the feature labels accordingly
            indices = np.argsort(sensitivities)
            sensitivities = sensitivities[indices]
            feature_labels = np.array(self.features_ref_labels)[indices]
        
            # Create a Dictionary with the results
            results = {
                'feature_names': feature_labels,
                'sensitivity': {
                    'Dataset' : sensitivities,
                },
                'gradients': {
                    # Not used but expected by plot_sensitivity_results
                    'Dataset': np.array([[x] for x in sensitivities]) 
                }
            }
               
            # Save the sensitivities to a file
            sensitivity_df = pd.DataFrame({'sensitivity': sensitivities}, index = feature_labels)
            sensitivity_path = os.path.join(sensitivity_output_path, 'sensitivity_analysis.csv')
            sensitivity_df.to_csv(sensitivity_path)

            # Plot the top sensitivities
            plot_sensitivity_results(results, modes=['barh'], output_folder=sensitivity_output_path)

        return

class NonLinear(CVCalculator):
    """
    Non-linear collective variables calculator (e.g. Autoencoder)
    """
    
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """ 
        Initializes a non-linear CV calculator.
        """
        from lightning import LightningModule
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        from mlcolvar.utils.trainer import MetricsCallback
        from mlcolvar.cvs import AutoEncoderCV, DeepTICA, VariationalAutoEncoderCV
                
        super().__init__(
            configuration,
            output_path)

        self.nonlinear_cv_map: Dict = {
            'ae': AutoEncoderCV,
            'deep_tica': DeepTICA,
            'vae': VariationalAutoEncoderCV
        }
        
        # Main attributes
        self.cv: Optional[lightning.LightningModule] = None
        self.checkpoint: Optional[ModelCheckpoint] = None
        self.metrics: Optional[MetricsCallback] = None
        self.weights_path: Optional[str] = None

        # Training configuration
        self.training_config: Dict = configuration['training'] 
        self.general_config: Dict = self.training_config['general']
        self.early_stopping_config: Dict  = self.training_config['early_stopping']
        self.optimizer_config: Dict = self.training_config['optimizer']
        self.lr_scheduler_config: Optional[Dict] = self.training_config['lr_scheduler_config']
        self.lr_scheduler: Optional[Dict] = self.training_config['lr_scheduler']
        self.model_to_save: Literal['best', 'last'] = self.training_config['model_to_save']
        
        # Training attributes
        self.max_tries: int = self.general_config['max_tries']
        self.seed: int = self.general_config['seed']
        self.training_validation_lengths: List = self.general_config['lengths']
        self.batch_size: int = self.general_config['batch_size']
        self.shuffle: bool = self.general_config['shuffle']
        self.random_split: bool = self.general_config['random_split']
        self.max_epochs: int = self.general_config['max_epochs']
        self.check_val_every_n_epoch: int = self.general_config['check_val_every_n_epoch']
        self.save_check_every_n_epoch: int = self.general_config['save_check_every_n_epoch']
        self.training_metrics_paths: List[str] = []

        self.cv_score: Union[float, None] = None
        self.tries: int = 0
        
        self.early_stop_patience: int = self.early_stopping_config['patience']
        self.early_stop_delta: float = self.early_stopping_config['min_delta']
                
        # Neural network settings
        self.encoder_config: Dict = self.architecture_config['encoder']
        self.decoder_config: Optional[Dict] = self.architecture_config['decoder']
        self.encoder_hidden_layers: List[int] = self.encoder_config.get('layers', []) 
        self.decoder_hidden_layers: Optional[List[int]] = self.decoder_config.get('layers', []) if self.decoder_config is not None else None
        
        # CV options for the mlcolvar CV class
        self.cv_options: Dict = {}

        # Set options from NN settings
        self.encoder_options: Dict = self.encoder_config
        self.encoder_options.pop('layers', {})
        if self.decoder_config is not None:
            self.decoder_options = self.decoder_config
            self.decoder_options.pop('layers', {})
        else:
            self.decoder_options = self.encoder_options
            
        # Optimizer
        self.opt_name: str = None
        self.optimizer_options: Dict = {}
        
        # Set up the last layer of the decoder
        self.set_up_last_layer()

    def load_model(self, 
        input_path: str
        ):
        """ 
        Loads a previously saved CV model from a zip file.
        Here we load the files specific to non-linear CV calculators.
        Mainly the cv model in PyTorch format.
        """
        super().load_model(input_path)
        
        import torch
        
        # Load the cv model weights
        weights_path = os.path.join(self.model_output_folder, 'cv_weights.pt')

        if not os.path.exists(weights_path):
            logger.error(f"CV model weights not found at {weights_path}")
            raise FileNotFoundError(f"CV model weights not found at {weights_path}")
        self.cv = torch.jit.load(weights_path)
        self.cv.eval()

    def set_up_last_layer(self):
        """
        Sets up the last layer of the decoder based on the normalization of the features.
        """
        # Add activation function to the last layer of the decoder based on the normalization of the features
        if isinstance(self.decoder_options['activation'], list):
            # Normalized using min max with range [0, 1] -> use sigmoid activation
            if self.feats_norm_mode == 'min_max_range1':
                self.decoder_options['activation'].append('custom_sigmoid')
            # Normalized using min max with range [-1, 1] -> use tanh activation
            elif self.feats_norm_mode == 'min_max_range2':
                self.decoder_options['activation'].append('tanh')
            # In other cases (mean_std or none), avoid adding an activation function
            else:
                self.decoder_options['activation'].append(None)
        else:
            logger.warning('''Decoder activation function is a single value, the same activation will be used for all layers. 
                           Make sure the chosen activation function matches the normalization of the features when doing regression
                           or choose last_layer_activation = False.''')
        
        # We add a layer to the decoder
        
        # No dropout for the last layer of the decoder
        if isinstance(self.decoder_options['dropout'], list):
            self.decoder_options['dropout'].append(None)
        # No batch norm for the last layer of the decoder 
        if isinstance(self.decoder_options['batchnorm'], list):
            self.decoder_options['batchnorm'].append(False)
        elif self.decoder_options['batchnorm'] is True:
            logger.warning('''Batch normalization is set to True for all layers of the decoder, 
                           including the last layer. Make sure this is intended or provide a list for batchnorm.''')
                       
    def _adjust_lr_scheduler(self, datamodule):
        """
        Adjusts LR scheduler parameters based on the training configuration.
        This is called right after the datamodule is created.
        """
        
        # Proceed only if a scheduler is defined
        if self.lr_scheduler is None:
            return

        # Split the data to get the number of samples in the training set
        datamodule.setup(stage='fit')

        logger.debug("Adjusting LR Scheduler parameters...") 

        scheduler_name = self.lr_scheduler.get('name', '')
        
        if scheduler_name == 'OneCycleLR':
            
            # Give reasonable default values if not provided in the configuration
            self.cv_options["lr_scheduler"]['max_lr'] = self.cv_options["lr_scheduler"].get('max_lr', 1e-3)
            self.cv_options["lr_scheduler"]['epochs'] = self.cv_options["lr_scheduler"].get('epochs', self.max_epochs)
            steps_per_epoch = len(datamodule.train_dataloader())
            self.cv_options["lr_scheduler"]['steps_per_epoch'] = self.cv_options["lr_scheduler"].get('steps_per_epoch', steps_per_epoch)
            
            # Adjust the interval from the configuration to 'step'
            self.cv_options["lr_scheduler_config"]["interval"] = 'step'
            
        elif scheduler_name == 'ReduceLROnPlateau':
            
            # Give reasonable default values if not provided in the configuration
            self.cv_options["lr_scheduler"]['patience'] = self.cv_options["lr_scheduler"].get('patience', self.early_stop_patience // 4)
            self.cv_options["lr_scheduler"]['cooldown'] = self.cv_options["lr_scheduler"].get('cooldown', self.early_stop_patience // 8)

            # Adjust the interval from the configuration to 'epoch'
            self.cv_options["lr_scheduler_config"]["interval"] = 'epoch'
            
    def check_batch_size(self):
        """  
        Check the batch size is not larger than the number of samples in the training set.
        If it is, set the batch size to the closest power of two smaller than the number
        of samples in the training set.
        """
        from deep_cartograph.modules.common import closest_power_of_two
        
        # Get the number of samples in the training set
        self.num_training_samples = int(self.num_samples*self.training_validation_lengths[0])
        
        # Check the batch size is not larger than the number of samples in the training set
        if self.batch_size >= self.num_training_samples:
            self.batch_size = closest_power_of_two(self.num_samples*self.training_validation_lengths[0])
            logger.warning(f"""The batch size is larger than the number of samples in the training set. 
                           Setting the batch size to the closest power of two: {self.batch_size}""")
    
    def set_encoder_layers() -> List:
        """ 
        Set the layers for the encoder of the non-linear model.
        Implement in subclasses.
        
        self.encoder: [input_dim, hidden_layer_1, hidden_layer_2, ..., output_dim]
            input_dim: number of features
            hidden_layer_i: number of neurons in the i-th hidden layer
            output_dim: dimension of the collective variable (latent space)
            
        Each non-linear model has different assumptions about the encoder layers input: ae, vae, deep_tica
            
        Returns
        -------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def set_decoder_layers(self) -> Optional[List]:
        """ 
        Set the layers for the decoder of the non-linear model.
        Implement in subclasses.
        
        self.decoder: [input_dim, hidden_layer_1, hidden_layer_2, ..., output_dim]
            input_dim: dimension of the collective variable (latent space)
            hidden_layer_i: number of neurons in the i-th hidden layer
            output_dim: number of features
            
        Each non-linear model has different assumptions about the decoder layers input: ae, vae
        
        Returns
        -------
        
        nn_layers : Optional[List]
            List with the layers for the decoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
            If None, no decoder is used (e.g. DeepTICA).
        """
        
        return None
        
    def set_up_cv_options(self):
        """
        Sets up the options for the CV based on the training configuration and the normalization of the features
        """
        import torch 
        
        # Normalization of features in the Non-linear models
        # No normalization
        if self.feats_norm_mode == 'none':
            self.cv_options = {'norm_in' : None}
        # Corresponding normalization, see prepare_normalization() in base class
        else:
            self.cv_options = {'norm_in' : {'mode' : 'mean_std',
                                                  'mean': torch.tensor(self.features_norm_mean),
                                                  'range': torch.tensor(self.features_norm_range)}}

        # Optimizer
        self.opt_name = self.optimizer_config['name']
        self.optimizer_options = self.optimizer_config['kwargs']
        
        self.cv_options["optimizer"] = self.optimizer_options
        
        # Construct the lr_scheduler option - scheduler class and its kwargs
        if self.lr_scheduler is not None:           
            # Obtain the class from the name
            lr_scheduler_class = getattr(torch.optim.lr_scheduler, self.lr_scheduler['name'], None)
            if lr_scheduler_class is None:
                logger.error(f'Learning rate scheduler {self.lr_scheduler["name"]} not recognized. Exiting...')
                raise ValueError(f'Learning rate scheduler {self.lr_scheduler["name"]} not recognized.')
                
            self.cv_options["lr_scheduler"] = {
                "scheduler" : lr_scheduler_class    
            }   
            
            self.cv_options["lr_scheduler"].update(self.lr_scheduler.get('kwargs', {}))
            
            # Construct the lr_scheduler_config option
            if self.lr_scheduler_config is not None:
                self.cv_options["lr_scheduler_config"] = self.lr_scheduler_config
                
    def create_model(self):
        """
        Implement this method in subclasses to create the non-linear model.
        
        Creates the non-linear model based on the configuration. 
        This is needed because there is no common API for all non-linear models in the mlcolvars library,
        thus the creation of the model has to be done in each specific CV calculator.
        
        Returns
        -------
        
        model : AutoEncoderCV, DeepTICA, VariationalAutoEncoderCV
            Non-linear model object
        """

        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_callbacks(self) -> List:
        """ 
        Get the callbacks for the training of the Nonlinear model.
        """
        
        from mlcolvar.utils.trainer import MetricsCallback
    
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        
        general_callbacks = []
        
        # Define MetricsCallback to store the loss
        self.metrics = MetricsCallback()
        general_callbacks.append(self.metrics)

        # Define EarlyStopping callback to stop training
        self.early_stopping = EarlyStopping(
            monitor="valid_loss", 
            min_delta=self.early_stop_delta, 
            patience=self.early_stop_patience, 
            mode = "min")
        general_callbacks.append(self.early_stopping)

        # Define ModelCheckpoint callback to save the best/last model
        checkpoints_path = os.path.join(self.training_output_folder, 'checkpoints')
        self.checkpoint = ModelCheckpoint(
            dirpath=checkpoints_path,                  # Directory to save the checkpoints  
            monitor="valid_loss",                      # Quantity to monitor
            save_last=True,                            # Save the last checkpoint - useful for VAE
            save_top_k=1,                              # Number of best models to save according to the quantity monitored
            save_weights_only=False,                   # Save only the weights
            filename=None,                             # Default checkpoint file name '{epoch}-{step}'
            mode="min",                                # Best model is the one with the minimum monitored quantity
            every_n_epochs=self.save_check_every_n_epoch)   # Number of epochs between checkpoints
        general_callbacks.append(self.checkpoint)

        return general_callbacks

    def train(self) -> bool:
        """
        Trains the non-linear collective variable using the training data.
        """
        import torch
        import lightning
        from mlcolvar.data import DictModule

        logger.info(f'Training {cv_names_map[self.cv_name]} ...')
        
        self.set_up_cv_options()
        
        self.check_batch_size()

        training_succeeded = False
        while not training_succeeded and (self.tries < self.max_tries):
            self.tries += 1
            try:
                # Setup datamodule and model for the current try
                datamodule = DictModule(
                    random_split=self.random_split,
                    dataset=self.training_input_dtset,
                    lengths=self.training_validation_lengths,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    generator=torch.manual_seed(self.seed + self.tries)
                )
                self._adjust_lr_scheduler(datamodule)
                model = self.create_model()
                model.optimizer_name = self.opt_name
                logger.info(f"Model architecture: {model}")

                # Setup and run trainer
                trainer = lightning.Trainer(
                    callbacks=self.get_callbacks(),
                    max_epochs=self.max_epochs,
                    logger=False,
                    enable_checkpointing=True,
                    enable_progress_bar=False,
                    check_val_every_n_epoch=self.check_val_every_n_epoch
                )
                logger.debug(f'Starting training try {self.tries}/{self.max_tries}...')
                trainer.fit(model, datamodule)
                logger.debug(f'Training try {self.tries} completed.')

                # Check if loss has decreased to consider the training a success
                if self.loss_decreased(self.metrics.metrics['valid_loss']):
                    training_succeeded = self._finalize_training()
                else:
                    logger.warning(f'Try {self.tries} failed: validation loss did not decrease. Retrying...')

            except Exception as e:
                logger.error(f'Training try {self.tries} failed with an exception: {e}')
                if self.tries < self.max_tries:
                    logger.info('Retrying...')

        if not training_succeeded:
            logger.error(f'{cv_names_map[self.cv_name]} did not converge after {self.max_tries} tries.')

        return training_succeeded

    def _finalize_training(self) -> bool:
        """
        Handles post-training tasks: model selection, loading, and logging scores.
        """
        # 1. Gather all available model info first
        
        # Last model info
        last_score = self.metrics.metrics['valid_loss'][-1] if self.metrics.metrics.get('valid_loss') else None
        last_model_path = self.checkpoint.last_model_path

        # Best overall model info
        best_score = self.checkpoint.best_model_score
        best_model_path = self.checkpoint.best_model_path

        # Best post-annealing model info (only for VAEs)
        best_post_anneal_score = None
        best_post_anneal_path = None
        if self.cv_name == 'vae' and hasattr(self, 'post_annealing_checkpoint'):
            best_post_anneal_score = self.post_annealing_checkpoint.best_score
            best_post_anneal_path = self.post_annealing_checkpoint.best_model_path

        # 2. Select the model to load
        model_path_to_load = None
        model_description = "N/A"
        
        # If the user requested the best model
        if self.model_to_save == 'best':
            if self.cv_name == 'vae':
                if best_post_anneal_path and os.path.exists(best_post_anneal_path):
                    model_path_to_load = best_post_anneal_path
                    model_description = "best post-annealing"
                else:
                    logger.warning("Best post-annealing model not found, falling back to last model.")
            elif best_model_path and os.path.exists(best_model_path):
                model_path_to_load = best_model_path
                model_description = "best overall"
            else:
                logger.warning("Best overall model not found, falling back to last model.")
        
        # If nothing selected yet, load the last model
        if model_path_to_load is None and last_model_path and os.path.exists(last_model_path):
            model_path_to_load = last_model_path
            model_description = "last"
            
        # 3. Load the selected model and log results
        if model_path_to_load:
            # The score of the model we are actually loading
            # (This is an example, you might want to store self.cv_score differently)
            if model_description == "best post-annealing":
                self.cv_score = best_post_anneal_score
            elif model_description == "best overall":
                self.cv_score = best_score
            else: # "last"
                self.cv_score = last_score
                
            self.cv = self.nonlinear_cv_map[self.cv_name].load_from_checkpoint(model_path_to_load)
            
            logger.info(f"Successfully loaded the '{model_description}' model from: {model_path_to_load}")
            if best_score is not None:
                logger.info(f"  -> Best Overall Score:      {best_score:.5f}")
            if best_post_anneal_score is not None:
                logger.info(f"  -> Best Post-Annealing Score: {best_post_anneal_score:.5f}")
            if last_score is not None:
                logger.info(f"  -> Last Model Score:          {last_score:.5f}")
                
            return True
        else:
            logger.error("Training finished, but no valid model checkpoint was found.")
            return False
        
    def loss_decreased(self, loss: List):
        """
        Check if the loss has decreased by the end of the training.

        Inputs
        ------

            loss:         loss for each epoch.
        """

        # Soft convergence condition: Check if the minimum of the validation loss is lower than the initial value
        if min(loss) > loss[0]:
            logger.warning('Validation loss has not decreased by the end of the training.')
            return False

        return True
    
    def plot_training_metrics(self):
        """
        Plots and saves training metrics common to all Non-linear CVs.
        """
        from mlcolvar.utils.plot import plot_metrics
        import torch
        
        # Create a new dictionary with all tensor metrics moved to the CPU
        cpu_metrics = {}
        for key, metric_list in self.metrics.metrics.items():
            # Check if the list is not empty and its first element is a tensor
            if metric_list and isinstance(metric_list[0], torch.Tensor):
                cpu_metrics[key] = [v.cpu().numpy() for v in metric_list] # Convert to numpy array directly
            else:
                cpu_metrics[key] = metric_list # Assume it's already CPU-compatible
        
        try:        
            # Move the best_model_score tensor to the CPU
            # Check if it's a tensor before calling .cpu()
            best_model_score_cpu = self.cv_score
            if isinstance(self.cv_score, torch.Tensor):
                best_model_score_cpu = best_model_score_cpu.cpu().detach()
            
            # Save the loss if requested
            if self.training_config['save_loss']:
                metrics_to_save = ['train_loss', 'valid_loss', 'epoch']
                for key in metrics_to_save:
                    if key not in cpu_metrics:
                        logger.warning(f'Metric {key} not found in metrics. It will not be saved.')
                        continue
                    filepath = os.path.join(self.training_output_folder, f'{key}.npy')
                    self.training_metrics_paths.append(filepath)
                    np.save(filepath, np.array(cpu_metrics[key]))
                np.savetxt(os.path.join(self.training_output_folder, 'model_score.txt'), np.array([best_model_score_cpu]), fmt='%.7g')
                    
            # Plot general metrics to all Non-linear CVs
            
            # Training and Validation loss
            Loss_present = ('train_loss' in cpu_metrics) and ('valid_loss' in cpu_metrics)
            if Loss_present:
                loss_plot_config = {
                    'keys': ['train_loss', 'valid_loss'],
                    'labels': ['Training', 'Validation'],
                    'colors': ['fessa1', 'fessa5'],
                    'linestyles': ['-', '-'],
                    'yscale': 'log'
                }
                ax = plot_metrics(cpu_metrics, **loss_plot_config)
                ax.figure.savefig(os.path.join(self.training_output_folder, f'loss.png'), dpi=300, bbox_inches='tight')
                ax.figure.clf()
            else:
                logger.warning('Training and/or validation loss not found in metrics. Loss plot will not be generated.')
                
            # Learning rate
            if 'lr' in cpu_metrics:
                lr_plot_config = {
                    'keys': ['lr'],
                    'labels': ['Learning Rate'],
                    'colors': ['fessa2'],
                    'linestyles': ['-'],
                    'yscale': 'log'
                }
                ax = plot_metrics(cpu_metrics, **lr_plot_config)
                ax.figure.savefig(os.path.join(self.training_output_folder, f'learning_rate.png'), dpi=300, bbox_inches='tight')
                ax.figure.clf()
            else:
                logger.warning('Learning rate not found in metrics. Learning rate plot will not be generated.')

        except Exception as e:
            import traceback
            logger.error(f'Failed to save/plot the loss. Error message: {e}\n{traceback.format_exc()}')

    def normalize_cv(self):
        
        import torch
        
        from mlcolvar.core.transform import Normalization
        from mlcolvar.core.transform.utils import Statistics
        
        # Data projected onto original latent space of the best model
        # The feature normalization is included in the model
        # We move the data to the device of the model (GPU or CPU) manually
        with torch.no_grad():
            self.cv.postprocessing = None
            projected_training_data = self.cv(self.training_input_dtset[:]['data'].to(self.cv.device))
        
        # Compute statistics of the projected training data
        stats = Statistics(projected_training_data)
        
        # Normalize the latent space
        norm =  Normalization(self.cv_dimension, mode='min_max', stats = stats )
        self.cv.postprocessing = norm
        
    def compute_cv(self):
        """
        Compute Non-linear CV.
        """

        # Train the non-linear model
        successfully_trained = self.train()  
        
        # If training was successful
        if successfully_trained:
        
            # Save the loss 
            self.plot_training_metrics()

            # After training, put model in evaluation mode - needed for cv normalization and data projection
            self.cv.eval()

    def save_weights(self, weights_path: str):
        """ 
        Saves the collective variable weights to a pytorch file.
        """

        successfully_saved = False
        try:
            # The model is set to evaluation mode before tracing
            self.cv.eval() 
            self.cv.to_torchscript(file_path=weights_path, method='trace')
            logger.debug(f'Collective variable model saved to {weights_path}')
            successfully_saved = True
        except Exception as e:
            logger.error(f'Failed to save TorchScript model using trace mode. Error: {e}')
            
        if not successfully_saved:
            logger.debug('Attempting to save the model using script mode instead of trace...')
            try:
                # Attempt to save the model using script mode
                self.cv.to_torchscript(file_path=weights_path, method='script')
                logger.debug(f'Collective variable model saved to {weights_path} using script mode')
            except Exception as e:
                logger.error(f'Failed to save TorchScript model using script mode. Error: {e}')
        
    def save_model(self):
        """
        Saves the collective variable model to a PyTorch TorchScript file.
        """
        super().save_model()   

        from deep_cartograph.modules.common import zip_files
        
        # Path to output model
        weights_path = os.path.join(self.model_output_folder, 'cv_weights.pt')

        if self.cv is None:
            logger.error('No collective variable model to save.')
            return

        # Save the model weights
        self.save_weights(weights_path)

        # Zip the model output folder
        model_path = os.path.join(self.output_path, 'model.zip')
        zip_files(model_path, self.model_output_folder)

        # Remove the unzipped model output folder
        shutil.rmtree(self.model_output_folder)

        logger.info(f'Model saved to {model_path}')
   
    def get_cv_parameters(self):
        """
        Get the collective variable parameters.
        """
        cv_parameters = {
            'cv_name': self.cv_name,
            'cv_dimension': self.cv_dimension,
            'weights_path': self.weights_path
        }
        return cv_parameters
    
    def get_cv_type(self) -> str:
        """
        Returns the type of the collective variable.
        """
        
        return "non-linear"

    def project_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Projects the given data onto the CV space.
        """
        import torch 
        
        if self.cv is None:
            logger.error('No collective variable model to project data.')
            raise ValueError('No collective variable model to project data.')
        
        logger.info(f'Projecting data onto {cv_names_map[self.cv_name]} ...')
        
        # Project the data onto the CV space
        with torch.no_grad():
            # Move data to the device of the model (GPU or CPU)
            data_on_model_device = torch.tensor(data.values).to(self.cv.device)
            # Project the data onto the CV space
            projected_tensor = self.cv(data_on_model_device)
            
        # Move to CPU and convert to numpy array
        projected_array = projected_tensor.cpu().numpy()
            
        projected_data = pd.DataFrame(projected_array, columns=self.cv_labels)   
        
        return projected_data  

    def sensitivity_analysis(self):
        """  
        Perform a sensitivity analysis of the CV on the training data.
        """
        from mlcolvar.explain import sensitivity_analysis
        
        from deep_cartograph.modules.figures import plot_sensitivity_results
        
        # Compute the sensitivity analysis
        results = sensitivity_analysis(self.cv, self.training_input_dtset, metric="mean_abs_val", 
                                       feature_names=None, per_class=False, plot_mode=None)
        
        # Save the sensitivities to a file
        sensitivity_df = pd.DataFrame({'sensitivity': results['sensitivity']['Dataset']}, index = results['feature_names'])
        sensitivity_path = os.path.join(self.sensitivity_output_folder, 'sensitivity_analysis.csv')
        sensitivity_df.to_csv(sensitivity_path)
        
        # Plot the sensitivity results
        modes = ['barh', 'violin']
        plot_sensitivity_results(results, modes=modes, output_folder=self.sensitivity_output_folder)


# Specific Collective Variable calculators
class PCACalculator(LinearCalculator):
    """
    Principal component analysis calculator.
    """
    
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """
        Initializes the PCA calculator.
        """
        super().__init__(
            configuration, 
            output_path)
        
        self.cv_name = 'pca'

        self.create_output_folders()
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...')

    def compute_cv(self):
        """
        Compute Principal Component Analysis (PCA) on the input features. 
        """
        
        # Create PCA object
        pca = PCA(n_components=self.cv_dimension)
        
        # Fit the PCA model
        pca.fit(self.training_data.to_numpy())
        
        # Save the eigenvectors as CVs
        self.cv = pca.components_.T
        
        # Follow a criteria for the sign of the eigenvectors - first weight of each eigenvector should be positive
        for i in range(self.cv_dimension):
            if self.cv[0,i] < 0:
                self.cv[:,i] = -self.cv[:,i]
                
class TICACalculator(LinearCalculator):
    """ 
    Time-lagged independent component analysis calculator.
    """
    
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """
        Initializes the TICA calculator.
        """
        super().__init__(
            configuration, 
            output_path)
        
        self.cv_name = 'tica'
        
        self.create_output_folders()
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...')
    
    def load_training_data(self, train_colvars_paths, train_topology_paths = None, ref_topology_path = None, features_list = None):
        super().load_training_data(train_colvars_paths, train_topology_paths, ref_topology_path, features_list)

        from mlcolvar.utils.timelagged import create_timelagged_dataset
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag) NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data.to_numpy(), lag_time=self.configuration['lag_time'])
        
    def compute_cv(self):
        """
        Compute Time-lagged Independent Component Analysis (TICA) on the input features. 
        """
        from mlcolvar.core.stats import TICA
        
        # Use TICA to compute slow linear combinations of the input features
        # Here out_features is the number of eigenvectors to keep
        tica_algorithm = TICA(in_features = self.num_features, out_features=self.cv_dimension)

        try:
            # Compute TICA
            _, tica_eigvecs = tica_algorithm.compute(data=[self.training_input_dtset['data'], self.training_input_dtset['data_lag']], save_params = True, remove_average = True)
        except Exception as e:
            logger.error(f'TICA could not be computed. Error message: {e}')
            return

        # Save the first cv_dimension eigenvectors as CVs
        self.cv = tica_eigvecs.numpy()
  
class HTICACalculator(LinearCalculator):
    """ 
    Hierarchical Time-lagged independent component analysis calculator.
    
    See: 
    
    Pérez-Hernández, Guillermo, and Frank Noé. “Hierarchical Time-Lagged Independent Component Analysis: 
    Computing Slow Modes and Reaction Coordinates for Large Molecular Systems.” Journal of Chemical Theory 
    and Computation 12, no. 12 (December 13, 2016): 6118–29. https://doi.org/10.1021/acs.jctc.6b00738.
    """
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """
        Initializes the HTICA calculator.
        """
        
        super().__init__(
            configuration, 
            output_path)
        
        self.cv_name = 'htica'
        
        self.num_subspaces = configuration['num_subspaces']
        self.subspaces_dimension = configuration['subspaces_dimension']
        
        self.create_output_folders()
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...')
    
    def load_training_data(self, train_colvars_paths, train_topology_paths = None, ref_topology_path = None, features_list = None):
        super().load_training_data(train_colvars_paths, train_topology_paths, ref_topology_path, features_list)
        
        from mlcolvar.utils.timelagged import create_timelagged_dataset
        
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag)
        # NOTE: Are we duplicating the data here? :(
        # NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data.to_numpy(), lag_time=self.configuration['lag_time'])
    
    def compute_cv(self):
        """
        Compute Hierarchical Time-lagged Independent Component Analysis (TICA) on the input features. 
        
        Initial space of features (num_features) -> TICA LEVEL 1 (subspaces_dimension x num_subspaces) -> TICA LEVEL 2 (CV_dimension)
        
            1. Divide the original dataset into num_subspaces
            2. Compute TICA on each sub-space (TICA LEVEL 1)
            3. Project each sub-space onto the TICA eigenvectors of LEVEL 1
            3. Construct the sparse - block diagonal - matrix transforming the original features into TICA LEVEL 1
            4. Compute TICA on the concatenated projected data (TICA LEVEL 2)
            5. Obtain the transformation matrix from features to TICA LEVEL 2 (final CV)
        """
        import torch
        from mlcolvar.core.stats import TICA
        
        data_tensor = self.training_input_dtset['data']
        data_lag_tensor = self.training_input_dtset['data_lag'] 
        
        # Split the data tensor into 10 tensors using torch_split
        data_tensors = torch.split(data_tensor, self.num_features//self.num_subspaces, dim=1)
        
        # Split the data lag tensor into 10 tensors using torch_split
        data_lag_tensors = torch.split(data_lag_tensor, self.num_features//self.num_subspaces, dim=1)
        
        # Initialize the eigenvectors and eigenvalues
        level_1_eigvecs = []     
        
        # Projected data
        projected_data = []
        projected_data_lag = []
        
        # Compute TICA on each of these pairs of tensors
        for data, data_lag in zip(data_tensors, data_lag_tensors):
            
            # Initialize the TICA object
            tica_algorithm = TICA(in_features = data.shape[1], out_features=self.subspaces_dimension)
            
            try:
                # Compute TICA
                _, eigvecs = tica_algorithm.compute(data=[data, data_lag], save_params = True, remove_average = True)
            except Exception as e:
                logger.error(f'TICA could not be computed. Error message: {e}')
                return
            
            # Save the eigenvectors and eigenvalues
            level_1_eigvecs.append(eigvecs.numpy())
            
            # Project each of the tensors onto the eigenvectors
            projected_data.append(torch.matmul(data, eigvecs))
            projected_data_lag.append(torch.matmul(data_lag, eigvecs))
        
        # Create the matrix that converts from the space of features to TICA LEVEL 1
        Transform_level_1_TICA = block_diag(level_1_eigvecs, format='csr') 
        
        # Concatenate the projected tensors
        projected_data = torch.concatenate(projected_data, axis=1)
        projected_data_lag = torch.concatenate(projected_data_lag, axis=1)
        
        # Apply TICA to the concatenated dataset
        tica_algorithm = TICA(in_features = projected_data.shape[1], out_features=self.cv_dimension)
        
        try:
            # Compute TICA
            _, level_2_eigvecs = tica_algorithm.compute(data=[projected_data, projected_data_lag], save_params = True, remove_average = True)
        except Exception as e:
            logger.error(f'TICA could not be computed. Error message: {e}')
            return
        
        # Obtain the transformation matrix from features to TICA LEVEL 2
        self.cv = Transform_level_1_TICA @ level_2_eigvecs
           
class AECalculator(NonLinear):
    """
    Autoencoder calculator.
    """
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """
        Initializes the Autoencoder calculator.
        """
        
        super().__init__(
            configuration, 
            output_path)
        
        self.cv_name = 'ae'
        
        self.create_output_folders()
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...')
        
        # Add encoder activation and dropout option for last layer if these are lists
        if isinstance(self.encoder_options['dropout'], list):
            self.encoder_options['dropout'].append(None)
        if isinstance(self.encoder_options['activation'], list):
            self.encoder_options['activation'].append(None)

    def set_up_cv_options(self):
        """ 
        Update cv options for the Autoencoder.
        """
        super().set_up_cv_options()
        
        # Update options
        cv_options = {
            "encoder": self.encoder_options,
            "decoder": self.decoder_options,
            "optimizer": self.optimizer_options
        }
        self.cv_options.update(cv_options)
        
    def load_training_data(self, train_colvars_paths, train_topology_paths = None, ref_topology_path = None, features_list = None):
        super().load_training_data(train_colvars_paths, train_topology_paths, ref_topology_path, features_list)
        
        import torch 
        from mlcolvar.data import DictDataset
        # Create DictDataset NOTE: we have to find another solution as this will duplicate the data
        dictionary = {"data": torch.Tensor(self.training_data.values)}
        self.training_input_dtset = DictDataset(dictionary, feature_names=self.features_ref_labels)
        
    def set_encoder_layers(self) -> List:
        """ 
        Set the layers for the encoder of the Autoencoder
        
        Return
        ------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        return [self.num_features] + self.encoder_hidden_layers + [self.cv_dimension]
    
    def set_decoder_layers(self) -> Optional[List]:
        """ 
        Set the layers for the decoder of the Autoencoder
        
        Return
        ------
        
        nn_layers : Optional[List]
            List with the layers for the decoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
            If None, no decoder is used (e.g. DeepTICA).
        """
        
        if self.architecture_config['decoder'] is None:
            return None
        else:
            return [self.cv_dimension] + self.decoder_hidden_layers + [self.num_features]

    def create_model(self):
        """ 
        Create the Autoencoder model.
        
        Returns
        -------
        
        model : AutoEncoderCV
        """
        
        from mlcolvar.cvs import AutoEncoderCV

        # Set layers from NN settings 
        self.encoder_layers: List = self.set_encoder_layers()
        self.decoder_layers: Optional[List] = self.set_decoder_layers()
        
        model = AutoEncoderCV(
            encoder_layers=self.encoder_layers, 
            decoder_layers=self.decoder_layers,
            options=self.cv_options)
         
        return model

    def plot_training_metrics(self): 
        """ 
        Plots and saves training metrics specific to Deep TICA.
        """      
        super().plot_training_metrics()
        
        from deep_cartograph.modules.common import zip_files, remove_files
    
        metrics_zip_file = os.path.join(self.training_output_folder, 'training_metrics.zip')
        zip_files(metrics_zip_file, *self.training_metrics_paths)
        remove_files(*self.training_metrics_paths)
        logger.info(f'Training metrics saved to {metrics_zip_file}')
        
class DeepTICACalculator(NonLinear):
    """
    DeepTICA calculator.
    """
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """
        Initializes the DeepTICA calculator.
        """      
        super().__init__(
            configuration, 
            output_path)
        
        self.cv_name = 'deep_tica'

        self.create_output_folders()
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...')
            
        # NOTE: In the future we might want to allow for a custom last layer before the latent space
        # Currently it will be the same as the one for the hidden layers - if not a list
        # or no activation/dropout is applied to the last layer
        
        # Add encoder activation and dropout option for last layer if these are lists
        if isinstance(self.encoder_options['dropout'], list):
            self.encoder_options['dropout'].append(None)
        if isinstance(self.encoder_options['activation'], list):
            self.encoder_options['activation'].append(None)

    def set_up_cv_options(self):
        """ 
        Update cv options for DeepTICA.
        """
        super().set_up_cv_options()
        
        # Update options
        cv_options = {
            "nn": self.encoder_options
        }
        self.cv_options.update(cv_options)
        
    def load_training_data(self, train_colvars_paths, train_topology_paths = None, ref_topology_path = None, features_list = None):
        super().load_training_data(train_colvars_paths, train_topology_paths, ref_topology_path, features_list)
        
        from mlcolvar.utils.timelagged import create_timelagged_dataset
        # Create time-lagged dataset (composed by pairs of samples at time t, t+lag) NOTE: this function returns less samples than expected: N-lag_time-2
        self.training_input_dtset = create_timelagged_dataset(self.training_data, lag_time=self.configuration['lag_time'])

    def set_encoder_layers(self) -> List:
        """ 
        Set the layers for the encoder of DeepTICA
        
        Return
        ------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        return [self.num_features] + self.encoder_hidden_layers + [self.cv_dimension]

    def create_model(self):
        """
        Create the DeepTICA model.
        
        Returns
        -------
        
        model : DeepTICA
            DeepTICA model object.
        """
        
        from mlcolvar.cvs import DeepTICA

        # Set layers from NN settings 
        self.encoder_layers: List = self.set_encoder_layers()
        self.decoder_layers: Optional[List] = self.set_decoder_layers()
        
        model = DeepTICA(
            layers=self.encoder_layers,
            options=self.cv_options)
        
        return model
    
    def plot_training_metrics(self): 
        """ 
        Plots and saves training metrics specific to Deep TICA.
        """      
        super().plot_training_metrics()
    
        from deep_cartograph.modules.common import zip_files, remove_files
        from mlcolvar.utils.plot import plot_metrics   

        # Find the epoch where the best model was found
        best_index = self.metrics.metrics['valid_loss'].index(self.cv_score)
        best_epoch = self.metrics.metrics['epoch'][best_index]
        logger.info(f'Took {best_epoch} epochs')

        # Find eigenvalues of the best model
        best_eigvals = [self.metrics.metrics[f'valid_eigval_{i+1}'][best_index] for i in range(self.cv_dimension)]
        for i in range(self.cv_dimension):
            logger.info(f'Eigenvalue {i+1}: {best_eigvals[i]}')
            
        np.savetxt(os.path.join(self.training_output_folder, 'eigenvalues.txt'), np.array(best_eigvals), fmt='%.7g')
        
        # Plot eigenvalues
        ax = plot_metrics(self.metrics.metrics,
                            labels=[f'Eigenvalue {i+1}' for i in range(self.cv_dimension)], 
                            keys=[f'valid_eigval_{i+1}' for i in range(self.cv_dimension)],
                            ylabel='Eigenvalue',
                            yscale=None)

        # Save figure
        ax.figure.savefig(os.path.join(self.training_output_folder, f'eigenvalues.png'), dpi=300, bbox_inches='tight')
        ax.figure.clf()
        
        metrics_zip_file = os.path.join(self.training_output_folder, 'training_metrics.zip')
        zip_files(metrics_zip_file, *self.training_metrics_paths)
        remove_files(*self.training_metrics_paths)
        logger.info(f'Training metrics saved to {metrics_zip_file}')

class VAECalculator(NonLinear):
    """
    Variational Autoencoder calculator.
    """
    def __init__(self, 
        configuration: Dict, 
        output_path: Union[str, None] = None
        ):
        """
        Initializes the Variational Autoencoder calculator.
        """      
        super().__init__(
            configuration, 
            output_path)
        
        # VAE-specific settings
        self.kl_annealing_config = self.training_config['kl_annealing']
        self.type = self.kl_annealing_config['type']
        self.start_beta = self.kl_annealing_config['start_beta']
        self.max_beta = self.kl_annealing_config['max_beta']
        self.start_epoch = self.kl_annealing_config['start_epoch']
        self.n_cycles = self.kl_annealing_config['n_cycles']
        self.n_epochs_anneal = self.kl_annealing_config['n_epochs_anneal']
        
        self.cv_name = 'vae'

        self.create_output_folders()
        
        logger.info(f'Calculating {cv_names_map[self.cv_name]} ...')
        
        # If the activation functions / dropout are given as a list, add one for the last layer 
        # Needed due to the addition of a n_cvs layer before passing it to Feed Forward in VAE model

    def set_up_cv_options(self):
        """ 
        Update cv options for the VAE.
        """
        super().set_up_cv_options()
        
        # Update options
        nn_options = {
            "encoder": self.encoder_options,
            "decoder": self.decoder_options
        }
        self.cv_options.update(nn_options)
        
    def load_training_data(self, train_colvars_paths, train_topology_paths = None, ref_topology_path = None, features_list = None):
        super().load_training_data(train_colvars_paths, train_topology_paths, ref_topology_path, features_list)

        import torch 
        from mlcolvar.data import DictDataset
        
        # Create DictDatase
        dictionary = {"data": torch.Tensor(self.training_data.values)}
        self.training_input_dtset = DictDataset(dictionary, feature_names=self.features_ref_labels)
        
    def set_encoder_layers(self) -> List:
        """ 
        Set the layers for the VAE
        
        Here the model already includes a mean and variance layer with
        cv_dimension outputs, so we do not need to add the last layer explicitly.
        
        Return
        ------
        
        nn_layers : List
            List with the layers for the encoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
        """
        
        return [self.num_features] + self.encoder_hidden_layers
    
    def set_decoder_layers(self):
        """ 
        Set the layers for the decoder of the VAE
        
        Here the model already includes a layer with the latent space dimension
        as input, so we do not need to add it explicitly.
        
        Return
        ------
        
        nn_layers : Optional[List]
            List with the layers for the decoder of the non-linear model.
            Contains the input dimension, hidden layers and output dimension.
            If None, no decoder is used (e.g. DeepTICA).
        """
        
        if self.decoder_config is None:
            return None
        else:
            return self.decoder_hidden_layers + [self.num_features]
   
    def create_model(self):
        """
        Create the Variational Autoencoder model.
        
        Returns
        -------
        
        model : VariationalAutoEncoderCV
            Variational Autoencoder model object.
        """
        
        from mlcolvar.cvs import VariationalAutoEncoderCV
        
        # Set layers from NN settings 
        self.encoder_layers: List = self.set_encoder_layers()
        self.decoder_layers: Optional[List] = self.set_decoder_layers()
        
        model = VariationalAutoEncoderCV(
            n_cvs=self.cv_dimension,
            encoder_layers=self.encoder_layers, 
            decoder_layers=self.decoder_layers,
            options=self.cv_options)
         
        return model
        
    def get_callbacks(self) -> List:
        """ 
        Get the callbacks for the VAE training.
        
        Uses the parent class method to get the callbacks and adds the VAE-specific callbacks.
        
        Returns
        -------
        
        callbacks: List
            List of callbacks for the VAE training.
        """
        import deep_cartograph.modules.ml as ml
        
        general_callbacks = super().get_callbacks()
        
        kl_annealing_args = {
            'type': self.type,
            'start_beta': self.start_beta,
            'max_beta': self.max_beta,
            'start_epoch': self.start_epoch,
            'n_cycles': self.n_cycles,
            'n_epochs_anneal': self.n_epochs_anneal
        }
        kl_anneal_callback = ml.KLAAnnealing(**kl_annealing_args)
        general_callbacks.append(kl_anneal_callback)
        
        # If ReduceLROnPlateau is the scheduler, add our custom manager
        if self.lr_scheduler and self.lr_scheduler.get('name') == 'ReduceLROnPlateau':
            # The start epoch should be when the kl annealing finishes + 1/4 of the remaining epochs
            start_monitoring_epoch = self.start_epoch + self.n_epochs_anneal + (self.max_epochs - self.start_epoch - self.n_epochs_anneal)//4
        
            lr_plateau_manager = ml.LROnPlateauManager(start_epoch=start_monitoring_epoch)
            general_callbacks.append(lr_plateau_manager)
        
        # If there is a KL annealing stage
        if self.n_epochs_anneal > 0:
            # Add a callback to start saving the best model after the KL annealing 
            checkpoints_path = os.path.join(self.training_output_folder, "checkpoints")
            self.post_annealing_checkpoint = ml.PostAnnealingCheckpoint(
                monitor="valid_loss",
                dirpath=checkpoints_path,
                annealing_end_epoch=self.start_epoch + self.n_epochs_anneal,
            )
            general_callbacks.append(self.post_annealing_checkpoint)
            
        return general_callbacks
    
    def _plot_metric(self, 
                     metric_data: Dict, 
                     metric_keys: List, 
                     labels: List, 
                     colors: List, 
                     yscale: str = 'log',
                     filepath: Optional[str] = None
                     ):
        """ 
        Plot a specific metric
        
        Parameters
        ----------
        
        metric_data : Dict
            Dictionary with the training metrics data.
        metric_keys : List
            List of keys in the metric_data to plot.
        labels : List
            List of labels for the metric keys.
        colors : List
            List of colors for the metric keys in fessa color scheme.
        yscale : str
            Y-axis scale. Default is 'log'.
        filepath : Optional[str]
            File path to save the plot.
        
        """
        from mlcolvar.utils.plot import plot_metrics  
        
        # Check the metric key is present in the data
        for key in metric_keys:
            if key not in metric_data:
                logger.warning(f'Metric {key} not found in metrics. It will not be plotted.')
                return None 
            
        if filepath is None:
            filepath = os.path.join(self.training_output_folder, f"{'_'.join(metric_keys)}.png")
        
        args = {
            'metrics': metric_data,
            'keys': metric_keys,
            'labels': labels,
            'linestyles': ['-']*len(metric_keys),
            'colors': colors,
            'yscale': yscale
        }
        ax = plot_metrics(**args)
        ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
        ax.figure.clf()

        return
        
            
    def plot_training_metrics(self): 
        """ 
        Plots and saves training metrics specific to VAE.
        """      
        super().plot_training_metrics()
        
        from deep_cartograph.modules.common import zip_files, remove_files
        import torch
        
        # Create a new dictionary with all tensor metrics moved to the CPU
        cpu_metrics = {}
        for key, metric_list in self.metrics.metrics.items():
            # Check if the list is not empty and its first element is a tensor
            if metric_list and isinstance(metric_list[0], torch.Tensor):
                cpu_metrics[key] = [v.cpu().numpy() for v in metric_list] # Convert to numpy array directly
            else:
                cpu_metrics[key] = metric_list # Assume it's already CPU-compatible

        try:
            # Save the training metrics if requested
            if self.training_config['save_loss']:
                metrics_to_save = ['train_kl_loss', 'valid_kl_loss', 'train_reconstruction_loss', 'valid_reconstruction_loss', 'beta', 'lr']
                for key in metrics_to_save:
                    if key not in cpu_metrics:
                        logger.warning(f'Metric {key} not found in metrics. It will not be saved.')
                        continue
                    filepath = os.path.join(self.training_output_folder, f'{key}.npy')
                    self.training_metrics_paths.append(filepath)
                    np.save(filepath, np.array(cpu_metrics[key]))

            # Plot metrics specific to VAE
            
            # KL divergence loss
            self._plot_metric(
                metric_data=cpu_metrics,
                metric_keys=['train_kl_loss', 'valid_kl_loss'],
                labels=['Training KL', 'Validation KL'],
                colors=['fessa1', 'fessa5'],
                yscale='log',
                filepath=os.path.join(self.training_output_folder, 'vae_kl_loss.png')
            )
            
            # Reconstruction loss
            self._plot_metric(
                metric_data=cpu_metrics,
                metric_keys=['train_reconstruction_loss', 'valid_reconstruction_loss'],
                labels=['Training Reconstruction', 'Validation Reconstruction'],
                colors=['fessa2', 'fessa6'],
                yscale='log',
                filepath=os.path.join(self.training_output_folder, 'vae_reconstruction_loss.png')
            )

            # KL + Reconstruction loss
            self._plot_metric(
                metric_data=cpu_metrics,
                metric_keys=['train_kl_loss', 'valid_kl_loss', 'train_reconstruction_loss', 'valid_reconstruction_loss'],
                labels=['Training KL', 'Validation KL', 'Training Reconstruction', 'Validation Reconstruction'],
                colors=['fessa1', 'fessa5', 'fessa2', 'fessa6'],
                yscale='log',
                filepath=os.path.join(self.training_output_folder, 'vae_kl_reconstruction_loss.png')
            )
            
            # Beta value
            self._plot_metric(
                metric_data=cpu_metrics,
                metric_keys=['beta'],
                labels=['Beta'],
                colors=['fessa3'],
                yscale='linear',
                filepath=os.path.join(self.training_output_folder, 'vae_beta.png')
            )
               
        except Exception as e:
            import traceback
            logger.error(f'Failed to save/plot the loss. Error message: {e}\n{traceback.format_exc()}')

        metrics_zip_file = os.path.join(self.training_output_folder, 'training_metrics.zip')
        zip_files(metrics_zip_file, *self.training_metrics_paths)
        remove_files(*self.training_metrics_paths)
        logger.info(f'Training metrics saved to {metrics_zip_file}')

# Mappings
cv_calculators_map = {
    'pca': PCACalculator,
    'ae': AECalculator,
    'tica': TICACalculator,
    'htica': HTICACalculator,
    'deep_tica': DeepTICACalculator,
    'vae': VAECalculator
}

cv_names_map = {
    'pca': 'PCA',
    'ae': 'AE',
    'tica': 'TICA',
    'htica': 'HTICA',
    'deep_tica': 'DeepTICA',
    'vae': 'VAE'
}

cv_components_map = {
    'pca': 'PC',
    'ae': 'AE',
    'tica': 'TIC',
    'htica': 'HTIC',
    'deep_tica': 'DeepTIC',
    'vae': 'VAE'
}