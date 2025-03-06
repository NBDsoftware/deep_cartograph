# Import modules
import sys
import logging
import numpy as np
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
        
            plumed_input_path (str):
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
            if len(entities) != 5:
                logger.error(f"Malformed sin feature label: {feature}")
                sys.exit(1)
            
            return plumed.command.sin(feature, [plumed.utils.to_atomgroup(entity) for entity in entities[1:]])
        
        elif feat_name == "cos":
            
            # Cosinus of a dihedral angle
            if len(entities) != 5:
                logger.error(f"Malformed cos feature label: {feature}")
                sys.exit(1)
            
            return plumed.command.cos(feature, [plumed.utils.to_atomgroup(entity) for entity in entities[1:]])
        
        elif feat_name == "tor":
            
            # Dihedral angle
            if len(entities) != 5:
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
        super().__init__(input_path, topology_path, feature_list, traj_stride)
        self.cv_type: Literal["linear", "non-linear"] = cv_type
        self.cv_params: Dict = cv_params
    
    def build(self):
        """Override the base build method to include the CV section."""
        super().build()
        
        # Add the CV section 
        self.add_cv_section()
        
    def add_cv_section(self):
        """
        Add the collective variable section to the PLUMED input file.
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
        
        features_stats = self.cv_params['features_stats']
        features_norm_mode = self.cv_params['features_norm_mode']
        
        if features_norm_mode == 'mean_std':
            features_offset = features_stats['mean'].numpy()
            features_scale = features_stats['std'].numpy()
        elif features_norm_mode == 'min_max':
            features_offset = features_stats['min'].numpy()
            features_scale = features_stats['max'].numpy() - features_stats['min'].numpy()
        elif features_norm_mode == 'none':
            pass
        else:
            raise ValueError(f"Features normalization mode {features_norm_mode} not recognized.")
        
        # Normalize the input features
        if features_norm_mode != 'none': 
            self.input_content += "\n# Normalized features\n"
            normalized_feature_list = []
            for index, feature in enumerate(self.feature_list):
                normalized_feature = f"feat_{index}"
                self.input_content += plumed.command.combine(normalized_feature, [feature], [features_scale[index]], [features_offset[index]])
                normalized_feature_list.append(normalized_feature)
        else:
            normalized_feature_list = self.feature_list
        
        # Add a combine command for each component of the CV
        self.input_content += "\n# Collective variable\n"
        for i in range(self.cv_params['weights'].shape[1]):
            self.input_content += plumed.command.combine(self.cv_params['cv_labels'][i], normalized_feature_list, self.cv_params['weights'][:,i])
            
    
    def validate_linear_cv(self):
        """
        Validate the parameters of a linear collective variable.
        """
        
        # Check if all required parameters are present
        if 'features_stats' not in self.cv_params:
            raise ValueError("Linear CV requires features statistics.")
        
        if 'features_norm_mode' not in self.cv_params:
            raise ValueError("Linear CV requires features normalization mode.")
        
        if 'weights' not in self.cv_params:
            raise ValueError("Linear CV requires weights.")
        
        if 'cv_labels' not in self.cv_params:
            raise ValueError("Linear CV requires CV labels.")
    
        # Clean labels
        self.cv_params['cv_labels'] = [label.replace(' ', '_') for label in self.cv_params['cv_labels']]
        
        # Check if the weights have the right shape
        if self.cv_params['weights'].shape[0] != len(self.feature_list):
            raise ValueError(f"CV weights shape {self.cv_params['weights'].shape} does not match the number of features {len(self.feature_list)}")
        
class EnhancedSamplingAssembler(CollectiveVariableAssembler):
    """
    Assembler class to add enhanced sampling to a PLUMED input file.
    """
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], traj_stride: int, cv_type: str, cv_params: Dict, sampling_method: str, sampling_params: Dict):
        super().__init__(input_path, topology_path, feature_list, traj_stride, cv_type, cv_params)
        self.sampling_method = sampling_method  # Type of enhanced sampling (e.g., metadynamics, umbrella sampling)
        self.sampling_params = sampling_params  # Parameters for the enhanced sampling method
    
    def build(self):
        """Override the base build method to include the enhanced sampling section."""
        super().build()
        
        # Ensure a CV is defined before applying enhanced sampling
        if not self.cv_type:
            raise ValueError("Enhanced sampling requires a collective variable.")
        
        # Add enhanced sampling section title
        self.input_content += "\n# Enhanced Sampling\n"
        
        # Generate the enhanced sampling command (to be implemented based on sampling_method)
        self.input_content += self.get_sampling_command()
    
    def get_sampling_command(self) -> str:
        """
        Generate the PLUMED command to apply enhanced sampling.
        """
        # Placeholder - implement logic based on self.sampling_method and self.sampling_params
        return f"# Apply {self.sampling_method} enhanced sampling here\n"