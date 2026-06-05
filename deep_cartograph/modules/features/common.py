import os
import logging
import numpy as np
from pathlib import Path
from typing import List

import deep_cartograph.modules.md as md
import deep_cartograph.modules.features as features

# Set logger
logger = logging.getLogger(__name__)


def find_common_features(features_configuration: dict, 
                         topologies: List[str],  
                         reference_topology: str,
                         output_folder: str
                         ) -> List[str]:
    """ 
    Find a common list of features to compute across all topologies. 
    This is done by translating the features from a reference topology to each of the 
    other topologies using a topology mapper, and then keeping only the features that 
    are present in all topologies.
    
    If there is no common set of features, the function will exit with an error message. 
    If some features are not present in all topologies, they will be discarded and a warning 
    message will be logged.
    
    Parameters
    ----------
    
        features_configuration : dict
            Configuration dictionary, see "features" in compute features schema for details
        
        topologies : List[str]
            List of paths to topology files.
        
        reference_topology : str
            Path to reference topology file. The features will be translated from this topology to the others.
        
        output_folder : str
            Path to output folder where the reference topology will be saved and used for feature translation.
    
    Returns
    -------
    
        common_features_list :
            List of strings with the features that are present in all topologies. These are the features
    """
        
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a reference topology file - pass through MDAnalysis to ensure the same topology 
    # as seen by PLUMED later when computing features
    ref_plumed_topology = os.path.join(output_folder, 'ref_topology.pdb')
    md.create_pdb(reference_topology, ref_plumed_topology)
    
    # Find reference features list
    ref_features_list = md.get_features_list(features_configuration, ref_plumed_topology)
    
    # Find features lists for each topology 
    features_lists = []
    for topology in topologies:
        
        top_name = Path(topology).stem

        # Create new temporary topology file
        plumed_topology = os.path.join(output_folder, 'plumed_topology.pdb')
        md.create_pdb(topology, plumed_topology)
        
        # Translate features to new topology
        logger.debug(f"Translating features from reference topology {Path(reference_topology).name} to topology {Path(topology).name}")
        features_list = features.Translator(ref_plumed_topology, plumed_topology, ref_features_list).run()
        
        if logger.isEnabledFor(logging.DEBUG):
            # Find indices of None values in feature list
            absent_features_idxs = [idx for idx, feature in enumerate(features_list) if feature is None]
            absent_features = [ref_features_list[idx] for idx in absent_features_idxs]
            if absent_features:
                logger.debug(f"There are {len(absent_features)} absent features in {top_name}: {absent_features}")
            else:
                logger.debug(f"No absent features in {top_name}. All reference features were translated successfully.")
        
        # If no features were translated, exit
        num_translated = sum(feature is not None for feature in features_list)
        if num_translated == 0:
            logger.error(f"No features could be translated to topology {top_name}. Please check your feature selection and the topology files. Exiting...")
            raise ValueError(f"No features could be translated to topology {top_name}. Please check your feature selection and the topology files. Exiting...")
        
        # Append to list of features lists
        features_lists.append(features_list)
        
        # Remove temporary topology file
        os.remove(plumed_topology)

    # Find a mask for None values in the features lists       
    masks = np.array([[x is not None for x in lst] for lst in features_lists])
    
    # Flatten the mask to keep only common features present in all topologies
    mask =  masks.all(axis=0)
    
    # Use the mask to keep only common features in each features list
    common_features_lists = [[lst[i] for i in range(len(lst)) if mask[i]] for lst in features_lists]
    ref_common_features_list = [ref_features_list[i] for i in range(len(ref_features_list)) if mask[i]]
    common_features_lists.extend([ref_common_features_list])
    
    # Check if all common feature lists have the same length
    if not all(len(lst) == len(common_features_lists[0]) for lst in common_features_lists):
        logger.error("Feature lists are not the same length, something went wrong when finding common features. Exiting...")
        raise ValueError("Feature lists are not the same length, something went wrong when finding common features. Exiting...")
    
    # Find and log discarded features
    if logger.isEnabledFor(logging.DEBUG):
        discarded_features = [ref_features_list[i] for i in range(len(ref_features_list)) if not mask[i]]
        if len(discarded_features) > 0:
            logger.debug(f"{len(discarded_features)} features were discarded because they are not present in all topologies:")
            logger.debug(discarded_features)
            logger.debug(f"{len(common_features_lists[0])} features were kept")
        else: 
            logger.debug("No features were discarded. All reference features are present in all topologies.")
    
    # Check that there are common features to compute
    if len(common_features_lists[0]) == 0:
        logger.error("There are no common features to compute. Please check your feature selection and the topology files. Exiting...")
        raise ValueError("There are no common features to compute. Please check your feature selection and the topology files. Exiting...")
    
    # Return reference common features
    return ref_common_features_list