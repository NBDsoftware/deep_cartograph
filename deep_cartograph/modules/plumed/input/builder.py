# Import modules
import sys
import logging
from typing import Dict, List, Optional

# Import local modules
from deep_cartograph.modules.plumed.input.assembler import Assembler, CollectiveVariableAssembler, EnhancedSamplingAssembler

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# Builders
# They determine the print arguments 
# They write the PLUMED input file
class ComputeFeaturesBuilder(Assembler):
    """
    Builder to create an input file that computes a collection of features during an MD simulation or trajectory.
    """           
    def __init__(self, input_path: str, 
                 topology_path: str, 
                 feature_list: List[str], 
                 traj_stride: int,
                 ref_topology_path: Optional[str] = None):
        return super().__init__(input_path, topology_path, feature_list, traj_stride, ref_topology_path)
    
    def build(self, colvars_path: str):
        """ 
        Override the base build method to include the print command.
        """
        super().build()
        
        # Add features to print arguments
        self.print_args = self.feature_list
        
        # Add the print command
        self.add_print_command(colvars_path, self.traj_stride)
        
        # Write the file
        self.write()
        
class ComputeCVBuilder(CollectiveVariableAssembler):
    """
    Builder to create an input file that computes a collective variable during an MD simulation or trajectory.
    """
    def __init__(self, input_path: str, 
                 topology_path: str, 
                 feature_list: List[str], 
                 traj_stride: int, 
                 cv_type: str, 
                 cv_params: Dict,
                 ref_topology_path: Optional[str] = None):
        return super().__init__(input_path, topology_path, feature_list, 
                                traj_stride, cv_type, cv_params, ref_topology_path)
    
    def build(self, colvars_path: str):
        """ 
        Override the base build method to include the print command.
        """
        super().build()
        
        # Check the cv_labels are defined
        if len(self.cv_labels) == 0:
            logger.error('No CV labels defined.')
            sys.exit(1)
        
        # Add CV to print arguments
        self.print_args.extend(self.cv_labels)
        
        # Add the print command
        self.add_print_command(colvars_path, self.traj_stride)
        
        # Write the file
        self.write()
        
class ComputeEnhancedSamplingBuilder(EnhancedSamplingAssembler):
    """ 
     Builder to create an input file to enhance sampling during an MD simulation or trajectory.
    """
    
    def __init__(self, input_path: str, topology_path: str, feature_list: List[str], 
                 traj_stride: int, cv_type: str, cv_params: Dict, sampling_method: str, 
                 sampling_params: Dict, ref_topology_path: Optional[str] = None):
        return super().__init__(input_path, topology_path, feature_list, traj_stride, 
                                cv_type, cv_params, sampling_method, sampling_params,
                                ref_topology_path)
    
    def build(self, colvars_path: str):
        """ 
        Override the base build method to include the print command.
        """
        super().build()
        
        # Check the cv_labels are defined
        if len(self.cv_labels) == 0:
            logger.error('No CV labels defined.')
            sys.exit(1)
        
        # Add CV to print arguments
        self.print_args.extend(self.cv_labels)
        
        # Add enhanced sampling variables to print arguments
        self.print_args.extend(self.bias_labels)
        
        # Add the print command
        self.add_print_command(colvars_path, self.traj_stride)
        
        # Write the file
        self.write()