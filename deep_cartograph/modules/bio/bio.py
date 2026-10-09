"""
Map residue numbers between two PDB topologies by aligning their sequences.
"""

import logging
from typing import Tuple, List, Dict, Union, Optional

from Bio import PDB, Align
from Bio.SeqUtils import seq1

# Set logger
logger = logging.getLogger(__name__)

class PDBTopologyMapper:
    """
    Map residue numbers from a reference PDB topology to a target PDB topology.

    Useful when the same protein is numbered differently in two files
    (e.g. different constructs or mutants).
    """
    def __init__(self, reference_topology: str, target_topology: str):
        """
        Use biopython to read two PDB files, align their sequences, and store a mapping.

        Parameters
        ----------
        reference_topology : str
            Path to the reference PDB file.
        target_topology : str
            Path to the other PDB file.
        """

        self.ref_sequence: str
        self.ref_resids: List[int]
        self.ref_resnames: List[Tuple[int, str]]

        self.sequence: str
        self.resids: List[int]
        self.resnames: List[Tuple[int, str]]

        self.alignment: Align.Alignment
        
        # The format of the mapping is the following:

        #    self.mapping = {
        #        6: ('A', 'A', 509),
        #        7: ('A', 'A', 510),
        #        8: ('M', 'M', 511),
        #        ...
        #    }

        #Where the key is the resid of the reference topology (for quick translation lookup) and
        #the Tuple value is formed by the reference resname, the target topology resname and 
        #the target topology resid.
        self.mapping: Dict[int, Tuple[str, str, int]]
        
        # Find information per residue of each topology
        self.ref_sequence, self.ref_resids = self.find_residues(reference_topology)
        self.ref_resnames = [(resid, resname) for resid, resname in zip(self.ref_resids, self.ref_sequence)]

        self.sequence, self.resids = self.find_residues(target_topology)
        self.resnames = [(resid, resname) for resid, resname in zip(self.resids, self.sequence)]

        # Align sequences and create mapping between residues in reference topology and the target topology
        self.alignment = self.align_sequences(self.ref_sequence, self.sequence)
        self.mapping = self.get_mapping()

    @staticmethod
    def find_residues(pdb_file: str, chain_id: Optional[str] = None) -> Tuple[str, List[int]]:
        """
        Find the one-letter sequence and the residue numbers from a PDB file.

        Only the first model is read. Residues that are not standard amino acids
        (e.g. water or ligands) are written as 'X'.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file.
        chain_id : str, optional
            Chain identifier in the PDB file. If None, all chains are read.

        Returns
        -------
        sequence : str
            One-letter sequence.
        indices : List[int]
            Residue numbers (resids), in the same order as the sequence.

        Raises
        ------
        ValueError
            If the file can't be parsed or the chain is not found.
        """
        parser = PDB.PDBParser(QUIET=True)
    
        structure = parser.get_structure("protein", pdb_file)
        
        if structure is None:
            raise ValueError(f"Could not parse PDB file: {pdb_file}")

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
        """
        Align two sequences using Bio.Align.PairwiseAligner and return the best local alignment.

        Parameters
        ----------
        seq1 : str
            First (reference) sequence
        seq2 : str
            Second (target) sequence

        Returns
        -------
        Align.Alignment
            Best-scoring local alignment
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
        
        alignments = aligner.align(seq1, seq2)
        return alignments[0]  # Take the best alignment
    
    def get_mapping(self) -> Dict[int, Tuple[str, str, int]]:
        """
        Create a mapping from reference residues to target residues using the sequence alignment.

        Only residues inside aligned segments are included. Residues in gaps are left out.

        NOTE: this mapping currently only takes into account amino acid residues.

        The format of the mapping is the following:

            mapping = {
                6: ('A', 'A', 509),
                7: ('A', 'A', 510),
                8: ('M', 'M', 511),
                ...
            }

        Where the key is the resid of the reference topology (for quick translation lookup) and
        the Tuple value is formed by the reference resname, the target topology resname and 
        the target topology resid.

        Returns
        -------
        Dict[int, Tuple[str, str, int]]
            Mapping from reference resid to (reference resname, target resname, target resid)
        """

        # Find indices of matching sequence segments (there can be more than one segments if there are mismatches or gaps)
        # These indices refer to the original sequences (self.ref_sequence and self.resnames)
        reference_segment_indices = self.alignment.aligned[0]
        segment_indices = self.alignment.aligned[1]

        mapping = {}
        # For each matching segment
        for reference_indices, indices in zip(reference_segment_indices, segment_indices):

            reference_residues_segment = self.ref_resnames[reference_indices[0]:reference_indices[1]]
            residues_segment = self.resnames[indices[0]:indices[1]]
            
            # Save each residue in the matching segment
            for reference_residue, residue in zip(reference_residues_segment, residues_segment):

                # Save relation between residues: ref_resid : (ref_resname, resname, resid)
                mapping.update({reference_residue[0]: (reference_residue[1], residue[1], residue[0])})

        return mapping 
    
    def map_residue(self, ref_residue_index: int) -> Union[int, None]:
        """
        Given a resid in the reference topology, return the corresponding resid in the other topology.

        Parameters
        ----------
        ref_residue_index : int
            Residue number (resid) in the reference topology.

        Returns
        -------
        resid : Union[int, None]
            Residue number in the other topology, or None if not found.
        """
        
        # NOTE: Should we check here the residue is the same? Should we ask for the atom name as well?
        
        map_entry = self.mapping.get(ref_residue_index)

        if map_entry:
            resid = map_entry[2]
        else:
            resid = None

        return resid

