# Import modules
import os
import sys
import logging
from typing import Dict, List, Literal

# Import local modules
import deep_cartograph.modules.plumed as plumed
from deep_cartograph.modules.md import md

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# Assemblers 
# They assemble the contents of the PLUMED input file into a string
# They inherit from each other to add more sections 
class Assembler:
    """
    Base class to assemble the contents of a PLUMED input file.
    """
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], traj_stride: int):
        """ 
        Minimal attributes to construct a PLUMED input file.
        
        Parameters
        ----------
        
            input_path (str):
                Path to the PLUMED input file. The file that will be written.
                
            topology_path (str):
                Path to the topology file. The one used by the MOLINFO command to define atom shortcuts.
                
            feature_list (list):
                List of features to be tracked.
            
            traj_stride (int):
                Stride to use when computing the features from a trajectory or MD simulation.
        """
        # Path to the contents of the input file
        self.input_content: str = ""
        
        # Path to the input file
        self.input_path: str = input_path
        
        # Path to the topology file used by PLUMED (MOLINFO command)
        self.topology_path: str = topology_path
        
        # List of features to be tracked
        self.feature_list: List[str] = feature_list
        
        # List of variables to be printed in a COLVAR file
        self.print_args: List[str] = []
        
        # Trajectory stride
        self.traj_stride: int = traj_stride
            
    def build(self):
        """
        Build the base content of the PLUMED input file. This method should be overridden by subclasses.
        """
        
        # Write Header title
        self.input_content += "# PLUMED input file generated with Deep Cartograph\n"
        
        # Write MOLINFO command - to use shortcuts for atom selections
        self.input_content += plumed.command.molinfo(os.path.abspath(self.topology_path))
        
        # Get the indices of the molecules that should be made whole - all by default
        whole_mol_indices = md.get_indices(self.topology_path)
        
        # Write WHOLEMOLECULES command - to correct for periodic boundary conditions
        self.input_content += plumed.command.wholemolecules(whole_mol_indices)
        
        # Leave blank line
        self.input_content += "\n"
        
        # Write Features section title  
        self.input_content += "# Features\n"
        
        # Write center commands first - Some features might need to use the geometric center of a group of atoms
        self.add_center_commands()
        
        # Write feature commands
        for feature in self.feature_list:
            self.input_content += self.get_feature_command(feature)
     
    def get_feature_command(self, feature: str) -> str:
        """
        Get the PLUMED command to compute a feature from its definition.
        
        Each feature is defined by a string with different 'entities' joined by '-'.
        The first entity is always the feature name, and defines which command/s should be used.
        The rest of the entities define the atoms that should be used to compute the feature and 
        the number of them will depend on the specific feature.

            entity1  - entity2 - entity3

            feat_name -  atom1  -  atom2   
            
            Ex: dist-@CA_584-@CA_549

        Parameters
        ----------
        
            feature (str):
                Name (i.e. definition) of the feature to compute.
        
        Returns
        -------
        
            command (str):
                PLUMED command to compute the feature.
        """   
        
        # Divide the feature definition into entities
        entities = feature.split("-")
        
        # Get the feature name
        feat_name = entities[0]
        
        # Construct the corresponding command from the entities
        if feat_name == "dist":
            
            # Distance
            if len(entities) != 3:
                logger.error(f"Malformed distance feature label: {feature}")
                sys.exit(1)
                
            for i in range(1,3):
                if entities[i].startswith("center_"):
                    pass
                else:
                    entities[i] = plumed.utils.to_atomgroup(entities[i])
            
            return plumed.command.distance(feature, entities[1:])
            
        elif feat_name == "sin":
            
            # Sinus of a dihedral angle
            if len(entities) != 5 and len(entities) != 2:
                logger.error(f"Malformed sin feature label: {feature}")
                sys.exit(1)
            
            return plumed.command.sin(feature, [plumed.utils.to_atomgroup(entity) for entity in entities[1:]])
        
        elif feat_name == "cos":
            
            # Cosinus of a dihedral angle
            if len(entities) != 5 and len(entities) != 2:
                logger.error(f"Malformed cos feature label: {feature}")
                sys.exit(1)
            
            return plumed.command.cos(feature, [plumed.utils.to_atomgroup(entity) for entity in entities[1:]])
        
        elif feat_name == "tor":
            
            # Dihedral angle
            if len(entities) != 5 and len(entities) != 2:
                logger.error(f"Malformed tor feature label: {feature}")
                sys.exit(1)
            
            return plumed.command.torsion(feature, [plumed.utils.to_atomgroup(entity) for entity in entities[1:]])
    
        else:
            
            logger.error(f"Feature {feature} not recognized.")
            sys.exit(1)

    def add_center_commands(self):
        """ 
        Write any center command needed to compute the features.
        """
        
        written_centers = []
        
        # Iterate over features
        for feature in self.feature_list:
            
            # Split into entities
            entities = feature.split("-")
            
            # Check if any entity is a center
            for entity in entities:
                
                if entity.startswith("center_"):
                    
                    # Check if the center has been written already
                    if entity not in written_centers:
                        
                        mda_selection = md.to_mda_selection(entity.replace('center_',''))
                        
                        # Write the center command
                        self.input_content += plumed.command.center(entity, md.get_indices(self.topology_path, mda_selection))
                        
                        # Save the center
                        written_centers.append(entity)
             
    def add_print_command(self, colvars_path: str, stride: int):
        """ 
        Add the print command to the PLUMED input file.
        """
        # Leave a blank line
        self.input_content += "\n"
        
        self.input_content += plumed.command.print(self.print_args, colvars_path, stride)

    def write(self):
        """
        Write the PLUMED input file. This method is not used by the Assembler classes but the Builder classes.
        """
        with open(self.input_path, "w") as f:
            f.write(self.input_content)
            
class CollectiveVariableAssembler(Assembler):
    """
    Assembler class to add the calculation of a collective variable to a PLUMED input file.
    """
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], traj_stride: int, 
                 cv_type: str, cv_params: Dict):
        """ 
        Assembler to add the calculation of a collective variable to a PLUMED input file.
        
        Parameters
        ----------
        
            input_path (str):
                Path to the PLUMED input file. The file that will be written.
                
            topology_path (str):
                Path to the topology file. The one used by the MOLINFO command to define atom shortcuts.
                
            feature_list (list):
                List of features to be tracked. Make sure the features are defined for this topoogy.
            
            traj_stride (int):
                Stride to use when computing the features from a trajectory or MD simulation.
                
            cv_type (str):
                Type of collective variable to compute. Can be 'linear' or 'non-linear'.
                
            cv_params (dict):
                Parameters for the collective variable. The parameters depend on the CV type.
        """
        super().__init__(input_path, topology_path, feature_list, traj_stride)
        self.cv_type: Literal["linear", "non-linear"] = cv_type
        self.cv_params: Dict = cv_params
        self.cv_labels: List[str] = []
    
    def build(self):
        """Override the base build method to include the CV section."""
        super().build()
        
        # Add the CV section 
        self.add_cv_section()
        
    def add_cv_section(self):
        """
        Add the collective variable section to the contents of the PLUMED input file.
        """
        
        # Add the corresponding CV commands
        if self.cv_type == "linear":
            self.add_linear_cv()
        elif self.cv_type == "non-linear":
            self.add_non_linear_cv()
        else:
            raise ValueError(f"CV type {self.cv_type} not recognized.")
        
    def add_linear_cv(self):
        """ 
        Add a linear collective variable to the PLUMED input file.
        """
        
        # Validate cv params
        self.validate_linear_cv()
        
        # Set up feature normalization
        features_norm_mode = self.cv_params['features_norm_mode']
        features_mean = self.cv_params['features_norm_mean']
        features_range = self.cv_params['features_norm_range']
        
        # Normalize the input features
        if features_norm_mode != 'none': 
            self.input_content += "\n# Normalized features\n"
            normalized_feature_labels = []
            for index, feature in enumerate(self.feature_list):
                normalized_feature = f"feat_{index}"
                self.input_content += plumed.command.combine(normalized_feature, [feature], [1/features_range[index]], [features_mean[index]])
                normalized_feature_labels.append(normalized_feature)
        else:
            normalized_feature_labels = self.feature_list
        
        # Compute the CV
        self.input_content += "\n# Collective variable\n"
        cv_labels = []
        for i in range(self.cv_params['weights'].shape[1]):
            component_name = f"{self.cv_params['cv_name']}_{i}"
            self.input_content += plumed.command.combine(component_name, normalized_feature_labels, self.cv_params['weights'][:,i])
            cv_labels.append(component_name)
        
        # Set up CV normalization
        cv_stats = self.cv_params['cv_stats']
        cv_offset = (cv_stats['min'] + cv_stats['max'])/2
        cv_scale = 2/(cv_stats['max'] - cv_stats['min'])
        
        # Normalize the CV
        self.input_content += "\n# Normalized Collective variable\n"
        normalized_cv_labels = []
        for i in range(self.cv_params['weights'].shape[1]):
            component_name = f"norm_{self.cv_params['cv_name']}_{i}"
            self.input_content += plumed.command.combine(component_name, [cv_labels[i]], [cv_scale[i]], [cv_offset[i]])
            normalized_cv_labels.append(component_name)
            
        # Set the final CV labels
        self.cv_labels = normalized_cv_labels
              
    def validate_linear_cv(self):
        """
        Validate the parameters of a linear collective variable.
        
        NOTE: migrate this to a pydantic model
        """
        
        if 'features_norm_mode' not in self.cv_params:
            raise ValueError("Linear CV requires features normalization mode.")
        
        if 'features_norm_mean' not in self.cv_params:
            raise ValueError("Linear CV requires features normalization mean.")
        
        if 'features_norm_range' not in self.cv_params:
            raise ValueError("Linear CV requires features normalization range.")
        
        if 'weights' not in self.cv_params:
            raise ValueError("Linear CV requires weights.")
        
        if 'cv_dimension' not in self.cv_params:    
            raise ValueError("Linear CV requires CV dimension.")
        
        if 'cv_stats' not in self.cv_params:
            raise ValueError("Linear CV requires CV statistics.")
        
        if 'cv_name' not in self.cv_params:
            self.cv_params['cv_name'] = 'cv'
        
        # Check if the weights have the right shape
        if self.cv_params['weights'].shape[0] != len(self.feature_list):
            raise ValueError(f"CV weights shape {self.cv_params['weights'].shape} does not match the number of features {len(self.feature_list)}")

        # Check that the CV dimension matches the number of components in the weights
        if self.cv_params['cv_dimension'] != self.cv_params['weights'].shape[1]:
            raise ValueError(f"CV dimension {self.cv_params['cv_dimension']} does not match the number of components in the weights {self.cv_params['weights'].shape[1]}")
        
    def add_non_linear_cv(self):
        """ 
        Add a non-linear collective variable to the PLUMED input file.
        
        Note that the feature and CV normalization are included inside the model
        """
        
        self.validate_non_linear_cv()
    
        # Compute the CV
        self.input_content += "\n# Collective variable\n"
        self.input_content += plumed.command.pytorch_model(self.cv_params['cv_name'], self.feature_list, os.path.abspath(self.cv_params['weights_path']))
            
        # Set the final CV labels
        self.cv_labels = [f"{self.cv_params['cv_name']}.node-{i}" for i in range(self.cv_params['cv_dimension'])]
        
    def validate_non_linear_cv(self):
        """
        Validate the parameters of a non-linear collective variable.
        
        NOTE: migrate this to a pydantic model
        """
        
        if 'weights_path' not in self.cv_params:
            raise ValueError("Non-linear CV requires weights path.")
        
        if 'cv_dimension' not in self.cv_params:
            raise ValueError("Non-linear CV requires CV dimension.")
        
        if 'cv_name' not in self.cv_params:
            self.cv_params['cv_name'] = 'cv'
             
class EnhancedSamplingAssembler(CollectiveVariableAssembler):
    """
    Assembler class to add enhanced sampling to a PLUMED input file.
    """
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], traj_stride: int, cv_type: str, cv_params: Dict, sampling_method: str, sampling_params: Dict):
        super().__init__(input_path, topology_path, feature_list, traj_stride, cv_type, cv_params)
        self.sampling_method = sampling_method  # Type of enhanced sampling (e.g., metadynamics, umbrella sampling)
        self.sampling_params = sampling_params  # Parameters for the enhanced sampling method
        self.bias_labels = []  # Labels of the bias potentials
        
    def build(self):
        """Override the base build method to include the enhanced sampling section."""
        super().build()
        
        # Add the enhanced sampling section
        self.add_enhanced_sampling_section()
        
    def add_enhanced_sampling_section(self):
        """ 
        Add the enhanced sampling section to the contents of the PLUMED input file.
        """
        
        if self.sampling_method == "wt_metadynamics":
            self.add_wt_metadynamics()
        elif self.sampling_method == "opes_metad":
            self.add_opes_metadynamics()
        elif self.sampling_method == "opes_metad_explore":
            self.add_opes_metadynamics_explore()
        elif self.sampling_method == "opes_expanded":
            self.add_opes_expanded()
        else:
            raise ValueError(f"Enhanced sampling method {self.sampling_method} not recognized.")
        
    def add_wt_metadynamics(self):
        """
        Add well-tempered metadynamics action to the PLUMED input file.
        """
        
        bias_name = 'wt_metad'
        
        # Ensure a CV is defined before applying enhanced sampling
        if not self.cv_type:
            raise ValueError("Enhanced sampling requires a collective variable.")
        
        # Set up the bias parameters
        metad_params = {
            'command_label' : bias_name,
            'arguments' : self.cv_labels,
            'sigmas' : [self.sampling_params['sigma'] for _ in range(self.cv_params['cv_dimension'])],
            'height' : self.sampling_params['height'],
            'bias_factor' : self.sampling_params['bias_factor'],
            'temperature' : self.sampling_params['temperature'],
            'pace' : self.sampling_params['pace'],
            'grid_mins' : [self.sampling_params['grid_min'] for _ in range(self.cv_params['cv_dimension'])],
            'grid_maxs' : [self.sampling_params['grid_max'] for _ in range(self.cv_params['cv_dimension'])],
            'grid_bins' : [self.sampling_params['grid_bin'] for _ in range(self.cv_params['cv_dimension'])],
        }
        
        # Add enhanced sampling section title
        self.input_content += "\n# Enhanced Sampling\n"
        
        # Generate the enhanced sampling command
        self.input_content += plumed.command.metad(**metad_params)
            
        self.bias_labels.append(f"{bias_name}.rbias")
        
    def add_opes_metadynamics(self):
        """
        Add OPES metadynamics action to the PLUMED input file.
        """
        
        bias_name = 'opes_metad'
        
        # Ensure a CV is defined before applying enhanced sampling
        if not self.cv_type:
            raise ValueError("Enhanced sampling requires a collective variable.")
        
        # Set up the bias parameters
        opes_params = {
            'command_label' : bias_name,
            'arguments' : self.cv_labels,
            'temperature' : self.sampling_params['temperature'],
            'pace' : self.sampling_params['pace'],
            'sigmas': [self.sampling_params['sigma'] for _ in range(self.cv_params['cv_dimension'])],
            'barrier': self.sampling_params['barrier'],
            'compression_threshold': self.sampling_params['compression_threshold']
        }

        # Add enhanced sampling section title
        self.input_content += "\n# Enhanced Sampling\n"
        
        # Generate the enhanced sampling command
        self.input_content += plumed.command.opes_metad(**opes_params)
        
        self.bias_labels.append(f"{bias_name}.bias")
        
    def add_opes_metadynamics_explore(self):
        """
        Add OPES metadynamics explore action to the PLUMED input file.
        """
        bias_name = 'opes_metad_explore'
        
        # Ensure a CV is defined before applying enhanced sampling
        if not self.cv_type:
            raise ValueError("Enhanced sampling requires a collective variable.")
        
        # Set up the bias parameters
        opes_params = {
            'command_label' : bias_name,
            'arguments' : self.cv_labels,
            'temperature' : self.sampling_params['temperature'],
            'pace' : self.sampling_params['pace'],
            'sigmas': [self.sampling_params['sigma'] for _ in range(self.cv_params['cv_dimension'])],
            'barrier': self.sampling_params['barrier'],
            'compression_threshold': self.sampling_params['compression_threshold']
        }

        # Add enhanced sampling section title
        self.input_content += "\n# Enhanced Sampling\n"

        # Generate the enhanced sampling command
        self.input_content += plumed.command.opes_metad_explore(**opes_params)
        
        self.bias_labels.append(f"{bias_name}.bias")
        
    def add_opes_expanded(self):
        """
        Add OPES expanded action to the PLUMED input file.
        """
        bias_name = 'opes_expanded'
        
        raise NotImplementedError("OPES expanded sampling is not yet implemented.")
        