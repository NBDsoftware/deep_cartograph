# Import modules
import os
import sys
import logging
from typing import Dict, Tuple, List, Union

# Import local modules
import deep_cartograph.modules.plumed as plumed
from deep_cartograph.modules.md import md

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# Base Builders - they write the contents of the PLUMED input file
class Builder:
    """
    Base class to build PLUMED input files.
    """
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], traj_stride: int):
        """ 
        Minimal attributes to build a PLUMED input file.
        
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
    
    def write(self):
        """
        Write the PLUMED input file.
        """
        with open(self.input_path, "w") as f:
            f.write(self.input_content)
            
    def build(self):
        """
        Build the base content of the PLUMED input file. This method should be overridden by subclasses.
        """
        
        # Write Header title
        self.input_content += "# PLUMED input file generated with Deep Cartograph\n"
        
        # Write MOLINFO command - to use shortcuts for atom selections
        self.input_content += plumed.command.molinfo(self.topology_path)
        
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
        
class TrackFeaturesBuilder(Builder):
    """
    Builder class to track a collection of features during an MD simulation or trajectory.
    """           
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], configuration: Dict):
        return super().__init__(input_path, topology_path, feature_list, configuration)
    
    def build(self, colvars_path: str):
        """ 
        Override the base build method to include the print command.
        """
        super().build()
        
        # Add features to print arguments
        self.print_args = self.feature_list
        
        # Add the print command
        self.add_print_command(colvars_path, self.configuration.get('traj_stride', 1))
        
        # Write the file
        self.write()