import logging
from typing import Tuple, List, Dict, Union

from Bio import PDB, Align
from Bio.SeqUtils import seq1

# Set logger
logger = logging.getLogger(__name__)

class PDBTopologyMapper:
    def __init__(self, reference_topology: str, topology: str):
        """
        Use biopython to read two PDB files, align their sequences, and store a mapping.
        
        Input
        -----
        
            reference_topology :
                Path to the reference PDB file.
            topology :
                Path to the other PDB file.
        """

        # Type hints
        self.ref_sequence: str
        self.ref_resids: List[int]
        self.ref_residues: List[Tuple[int, str]]
        self.sequence: str
        self.resids: List[int]
        self.residues: List[Tuple[int, str]]
        self.alignment: Align.Alignment
        self.mapping: Dict[int, Tuple[str, str, int]]
        
        # Find information per residue of each topology
        self.ref_sequence, self.ref_resids = self.find_residues(reference_topology)
        self.ref_residues = [(resid, resname) for resid, resname in zip(self.ref_resids, self.ref_sequence)]

        self.sequence, self.resids = self.find_residues(topology)
        self.residues = [(resid, resname) for resid, resname in zip(self.resids, self.sequence)]

        # Align sequences and create mapping between residues in reference topology and the other topology
        self.alignment = self.align_sequences(self.ref_sequence, self.sequence)
        self.mapping = self.get_mapping()

    @staticmethod
    def find_residues(pdb_file: str, chain_id: str = None) -> Tuple[str, List[int]]:
        """
        Finds the one-letter amino acid sequence and its residue indices from a PDB file.
        
        Input
        -----
        
            pdb_file :
                Path to the PDB file.
            chain_id :
                Chain identifier in the PDB file.
        
        Returns
        -------
        
            sequence :
                One-letter amino acid sequence.
            indices :
                Residue indices.
        """
        parser = PDB.PDBParser(QUIET=True)
    
        structure = parser.get_structure("protein", pdb_file)
        
        sequence = []
        indices = []
        model = structure[0]  # Assume first model is relevant
        
        try:
            chains = [model[chain_id]] if chain_id else model  # Direct access if chain_id is provided
        except KeyError:
            raise ValueError(f"Chain '{chain_id}' not found in PDB file.")
        
        for chain in chains:
            for residue in chain:
                res_name = residue.get_resname()
                sequence.append(seq1(res_name))
                indices.append(residue.id[1])  # Store residue sequence number
        
        return "".join(sequence), indices
    
    @staticmethod
    def align_sequences(seq1: str, seq2: str) -> Align.Alignment:
        """Aligns two sequences using Bio.Align.PairwiseAligner and returns the best local alignment."""
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
        
        alignments = aligner.align(seq1, seq2)
        return alignments[0]  # Take the best alignment
    
    def get_mapping(self) -> Dict[int, int]:
        """
        Creates a mapping from reference residues to other topology residues using the sequence alignment.
        
        NOTE: this mapping currently only takes into account amino acid residues.
        
        The format of the mapping is the following:

            self.mapping = {
                6: ('A', 'A', 509),
                7: ('A', 'A', 510),
                8: ('M', 'M', 511),
                ...
            }

        Where the key is the resid of the reference topology (for quick translation lookup) and
        the Tuple value is formed by the reference resname, the other topology resname and the other topology resid.
        """

        # Find indices of matching sequence segments (there can be more than one segments if there are mismatches or gaps)
        # These indices refer to the original sequences (self.ref_sequence and self.residues)
        reference_segment_indices = self.alignment.aligned[0]
        segment_indices = self.alignment.aligned[1]

        mapping = {}
        # For each matching segment
        for reference_indices, indices in zip(reference_segment_indices, segment_indices):

            reference_residues_segment = self.ref_residues[reference_indices[0]:reference_indices[1]]
            residues_segment = self.residues[indices[0]:indices[1]]
            
            # Save each residue in the matching segment
            for reference_residue, residue in zip(reference_residues_segment, residues_segment):

                # Save relation between residues: ref_resid : (ref_resname, resname, resid)
                mapping.update({reference_residue[0]: (reference_residue[1], residue[1], residue[0])})

        return mapping 
    
    def map_residue(self, ref_residue_index: int) -> Union[int, None]:
        """
        Given a resid in the reference topology, return the corresponding resid in the other topology.
        
        Input
        -----
        
            ref_residue_index :
                Residue index in the reference topology.
                
        Returns
        -------
        
            resid :
                Residue index in the other topology or None if not found.
        """
        
        # NOTE: Should we check here the residue is the same? Should we ask for the atom name as well?
        
        map_entry = self.mapping.get(ref_residue_index)

        if map_entry:
            resid = map_entry[2]
        else:
            resid = None

        return resid

