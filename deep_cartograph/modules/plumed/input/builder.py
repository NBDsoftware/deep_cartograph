"""
Builders that write complete PLUMED input files.

Each builder uses an assembler to create the content, adds a PRINT command and writes the file.
"""
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
    Builder for an input file that computes and prints a list of features.

    Duplicated feature labels are dropped with a warning. See ``Assembler`` for the parameters.
    """
    def __init__(self, plumed_input_path: str, 
                 topology_path: str, 
                 features_list: List[str], 
                 traj_stride: int,
                 fit_template_path: Optional[str] = None):

        # Guard against duplicated feature labels
        # PLUMED fails to parse an input file containing the same label twice, and feature
        # labels (@atomName_atomResid) are not guaranteed to be unique in a topology.
        # Callers are expected to hand over a clean list, so this is only a safety net.
        unique_features_list = list(dict.fromkeys(features_list))
        if len(unique_features_list) < len(features_list):
            logger.warning(f"{len(features_list) - len(unique_features_list)} duplicated feature labels were dropped before building the PLUMED input.")
            logger.debug(f"Dropped duplicated feature labels: "
                         f"{[feature for feature in unique_features_list if features_list.count(feature) > 1]}")

        return super().__init__(plumed_input_path, topology_path, unique_features_list, traj_stride, fit_template_path)

    def build(self, colvars_path: str):
        """
        Build the input file, add a PRINT command for all features and write it.

        Parameters
        ----------
        colvars_path : str
            Path to the colvars file where PLUMED will print the values.
        """
        super().build()
        
        # Add features to print arguments
        self.print_args = self.features_list

        # Add the print command
        self.add_print_command(colvars_path, self.traj_stride)
        
        # Write the file
        self.write()
        
class ComputeCVBuilder(CollectiveVariableAssembler):
    """
    Builder for an input file that computes and prints a collective variable.

    See ``CollectiveVariableAssembler`` for the parameters.
    """
    def __init__(self, plumed_input_path: str, 
                 topology_path: str, 
                 features_list: List[str], 
                 traj_stride: int, 
                 cv_type: str, 
                 cv_params: Dict,
                 fit_template_path: Optional[str] = None):
        return super().__init__(plumed_input_path, topology_path, features_list, 
                                traj_stride, cv_type, cv_params, fit_template_path)
    
    def build(self, colvars_path: str):
        """
        Build the input file, add a PRINT command for the CV components and write it.

        Parameters
        ----------
        colvars_path : str
            Path to the colvars file where PLUMED will print the values.
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
    Builder for an input file that biases a collective variable with an enhanced sampling method.

    See ``EnhancedSamplingAssembler`` for the parameters.
    """
    
    def __init__(self, plumed_input_path: str, topology_path: str, features_list: List[str], 
                 traj_stride: int, cv_type: str, cv_params: Dict, sampling_method: str, 
                 sampling_params: Dict, fit_template_path: Optional[str] = None, 
                 rmsd_restraint_reference_path: Optional[str] = None, 
                 rmsd_restraint_k: Optional[float] = None,
                 rmsd_restraint_eq: Optional[float] = None):
        return super().__init__(plumed_input_path, topology_path, features_list, traj_stride, 
                                cv_type, cv_params, sampling_method, sampling_params,
                                fit_template_path, rmsd_restraint_reference_path, 
                                rmsd_restraint_k, rmsd_restraint_eq)
    
    def build(self, colvars_path: str):
        """
        Build the input file, add a PRINT command for the CV components and the bias, and write it.

        Parameters
        ----------
        colvars_path : str
            Path to the colvars file where PLUMED will print the values.
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