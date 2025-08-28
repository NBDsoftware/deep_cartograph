import logging
from typing import List

from deep_cartograph.modules.bio import PDBTopologyMapper
# Set logger
logger = logging.getLogger(__name__)

# * NOTE: It would be better if the translator had a method with:
# *       - origin topology
# *       - target topology
# *       - list of features

class FeatureTranslator:
    """
    Class that uses a topology mapper to translate a list of features from a reference topology to another topology.
    """
    
    def __init__(self, reference_topology: str, target_topology: str, reference_features: List[str]):
        """
        Initialize the feature translator.
        """
        self.reference_topology = reference_topology
        self.target_topology = target_topology
        self.reference_features = reference_features
        
    def run(self) -> List[str]:
        """ 
        Create a topology mapper and translate the list of features from the reference topology to the target topology
        
        Returns
        -------
        
            translated_features :
                List of strings with translated feature or None if the feature is not present in the target topology
        """
        
        # Create a topology mapper between the reference topology and the target topology
        self.top_mapper = PDBTopologyMapper(self.reference_topology, self.target_topology)
        
        return self.translate_features()
    
    def translate_features(self) -> List[str]:
        """ 
        Translate each feature from the reference topology to the target topology. 
        """
        
        translated_features = []
        # For each feature given
        for feature in self.reference_features:
            
            # Separate into its parts. First item is name, the rest are atoms
            entities = feature.split("-")
            
            if len(entities) == 1:
                # If the feature doesn't have atoms, store it and continue (e.g. time, walker... columns)
                translated_features.append(feature)
                continue
            
            feature_name = entities[0]
            ref_atoms = entities[1:]
            
            if feature_name == "coord":
                # Remove the axis suffix from the last entity
                atom, axis = ref_atoms[-1].split(".")
                ref_atoms[-1] = atom

            # Translate each atom from the reference topology to the target topology
            atoms = [self.translate_atom(atom) for atom in ref_atoms]

            # If the target topology file has all the necessary atoms
            if None not in atoms:
                # Recompose the feature in the target topology
                translated_features.append(feature_name + "-" + "-".join(atoms))
                
                if feature_name == "coord":
                    # Add the axis suffix to the last entity
                    translated_features[-1] += "." + axis
            else:
                # Store None otherwise
                translated_features.append(None)
        
        return translated_features
            
    def translate_atom(self, atom: str) -> str:
        """ 
        Translate atom from the reference topology to the new topology
        
        NOTE: We assume the following format for atoms: @CA_579 or @phi_579 (from distance or torsion features)
        
        NOTE: We assume the atom name is not changing in the target topology
        """
        
        ref_atom_name, ref_resid = atom.split('_')

        target_resid = self.top_mapper.map_residue(int(ref_resid))

        if target_resid:
            target_atom = ref_atom_name + "_" + str(target_resid)
        else:
            target_atom = None

        return target_atom
        
        
        
        
        
        
            
            
        
        
        
