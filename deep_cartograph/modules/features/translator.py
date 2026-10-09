"""
Translate feature names from a reference topology to another topology,
matching residues through a sequence alignment.
"""

import logging
from typing import List, Optional

from deep_cartograph.modules.bio import PDBTopologyMapper
# Set logger
logger = logging.getLogger(__name__)

# * NOTE: It would be better if the translator had a method with:
# *       - origin topology
# *       - target topology
# *       - list of features

class Translator:
    """
    Class that uses a topology mapper to translate a list of features from a reference topology to another topology.

    Only the residue numbers in the feature names are changed. For example, 'dist-@CA_579-@CA_600'
    could become 'dist-@CA_575-@CA_596' in a topology with a different numbering.
    """

    def __init__(self, reference_topology: str, target_topology: str, reference_features: List[str]):
        """
        Initialize the feature translator.

        Parameters
        ----------
        reference_topology : str
            Path to the reference topology (PDB) file
        target_topology : str
            Path to the target topology (PDB) file
        reference_features : List[str]
            Feature names defined in the reference topology
        """
        self.reference_topology = reference_topology
        self.target_topology = target_topology
        self.reference_features = reference_features
        
    def run(self) -> List[str]:
        """ 
        Translate the list of features from the reference topology to the target topology respecting 
        the original order of the features. This is done using a topology mapper that maps
        residues from the reference topology to the target topology using a sequence alignment.
        
        If the feature is not present in the target topology, it is replaced by None.
        
        Returns
        -------
        translated_features : List[Optional[str]]
            Translated feature names, or None for features that are not present in the target topology
        """
        
        # Create a topology mapper between the reference topology and the target topology
        self.top_mapper = PDBTopologyMapper(self.reference_topology, self.target_topology)
        
        return self.translate_features()
    
    def translate_features(self) -> List[str]:
        """ 
        Translate each feature from the reference topology to the target topology respecting the original
        order of the features.

        Requires the topology mapper created in run(). Names without atoms (e.g. 'time')
        are kept as they are.

        Returns
        -------
        translated_features : List[Optional[str]]
            Translated feature names, or None for features that are not present in the target topology
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
            
    def translate_atom(self, atom: str) -> Optional[str]:
        """ 
        Translate an atom from the reference topology to the target topology.

        NOTE: We assume the following format for atoms: @CA_579 or @phi_579 (from distance or torsion features)

        NOTE: We assume the atom name is not changing in the target topology

        Parameters
        ----------
        atom : str
            Atom in the reference topology, as '<name>_<resid>'

        Returns
        -------
        Optional[str]
            Atom in the target topology, or None if the residue has no match
        """
        
        ref_atom_name, ref_resid = atom.split('_')

        target_resid = self.top_mapper.map_residue(int(ref_resid))

        if target_resid:
            target_atom = ref_atom_name + "_" + str(target_resid)
        else:
            target_atom = None

        return target_atom
        
        
        
        
        
        
            
            
        
        
        
