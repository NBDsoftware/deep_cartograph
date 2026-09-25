"""
Helpers built on MDAnalysis to work with topologies and trajectories.

Includes functions to list feature labels, extract frames, interpolate trajectories and compute RMSD-like metrics.
"""
# Import modules
import os
import sys
import glob
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Literal, Tuple, Optional

import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.analysis.align
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis import transformations as trans

from deep_cartograph.modules.bio import PDBTopologyMapper

# Set logger
logger = logging.getLogger(__name__)

# Constants for the module
covalent_bond_threshold = 2.0


# To find labels from topologies and MDAnalysis selections
def find_distances(topology_path: str, selection1: str, selection2: str, stride1: int, stride2: int, skip_neighbors: bool, skip_bonded_atoms: bool) -> List[str]:
    """
    Find all pairwise distances between heavy atoms of two selections.

    Each distance is returned as a label of the form ``@atom1Name_atom1Resid-@atom2Name_atom2Resid``.
    Repeated pairs (A-B and B-A) are only listed once.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    selection1 : str
        MDAnalysis selection string for the first group of atoms (e.g. ``"name CA"``).
    selection2 : str
        MDAnalysis selection string for the second group of atoms.
    stride1 : int
        Keep one of every ``stride1`` heavy atoms from the first selection.
    stride2 : int
        Keep one of every ``stride2`` heavy atoms from the second selection.
    skip_neighbors : bool
        If True, skip pairs of atoms in the same residue or in consecutive residues.
    skip_bonded_atoms : bool
        If True, skip pairs of bonded atoms. Bonds are taken from the topology or,
        if missing, guessed with a distance cutoff.

    Returns
    -------
    all_labels : list of str
        Distance labels.

    Raises
    ------
    ValueError
        If any of the two selections has no heavy atoms.
    """

    # Load topology
    u = mda.Universe(topology_path)

    # Select first group of atoms
    first_atoms = u.select_atoms(selection1)

    # Keep only heavy atoms
    first_atoms = first_atoms.select_atoms("not name H*")

    # Keep only one every first_stride atoms
    first_atoms = first_atoms[::stride1]

    # Select second group of atoms
    second_atoms = u.select_atoms(selection2)

    # Keep only heavy atoms
    second_atoms = second_atoms.select_atoms("not name H*")

    # Keep only one every second_stride atoms
    second_atoms = second_atoms[::stride2]

    # Check the selections are not empty
    if len(first_atoms) == 0:
        raise ValueError(f"First selection: '{selection1}' is empty, please review the selection string.")
    if len(second_atoms) == 0:
        raise ValueError(f"Second selection: '{selection2}' is empty, please review the selection string.")

    # Check if bonds are present in the topology
    if hasattr(u, 'bonds') and skip_bonded_atoms:
        logger.info("Provided topology contains bonds, distances between bonded atoms will be excluded.")
    elif not hasattr(u, 'bonds') and skip_bonded_atoms:
        logger.warning(f"Provided topology does not contain bonds. Bonds will be guessed using a distance criterion (bond_length < {covalent_bond_threshold}). Distances between bonded atoms will be excluded.")
    elif not skip_bonded_atoms:
        logger.warning("Distances between bonded atoms will be included. Note that these distances will not contain relevant information. To exclude them use 'skip_bonded_atoms: True'.")

    if not skip_neighbors:
        logger.warning("Distances between atoms in the same residue or neighboring residues will be included. These are unlikely to contain relevant information. To exclude them use 'skip_neigh_residues: True'.")

    all_labels = []

    # Create all possible pairs of atoms without repetition
    for first_atom in first_atoms:

        for second_atom in second_atoms:
            
            # Check both atoms are different
            if first_atom.index != second_atom.index:
                
                # Create entities
                entity1 = f"@{first_atom.name}_{first_atom.resid}"
                entity2 = f"@{second_atom.name}_{second_atom.resid}"

                # Create atomic label for this distance
                label = f"{entity1}-{entity2}"
                equivalent_label = f"{entity2}-{entity1}"

                # Check distance is not repeated
                if label not in all_labels and equivalent_label not in all_labels:
                    
                    # Check atoms are not bonded 
                    if skip_bonded_atoms:
                        # Using the topology
                        if hasattr(u, 'bonds'):
                            if second_atom in first_atom.bonded_atoms:
                                continue
                        else:
                            # Using a distance criterion
                            if calc_bonds(first_atom.position, second_atom.position) < covalent_bond_threshold:
                                continue
                    
                    # Check if atoms pertain to the same residue or neighboring residues
                    if skip_neighbors:
                        if abs(first_atom.resid - second_atom.resid) <= 1:
                            continue

                    # If previous checks are passed, add the distance to the list
                    all_labels.append(label)
                    
    return all_labels

def find_dihedrals(topology_path: str, selection: str, search_mode: str) -> List[str]:
    """
    Find dihedral angles in a selection of the topology.

    Each dihedral is returned as a label. ``search_mode`` controls which dihedrals are found:

    - ``virtual``: dihedrals between consecutive heavy atoms, assuming they are connected in order.
      Meant for coarse-grained models (e.g. only C-alpha atoms).
    - ``protein_backbone``: backbone phi and psi angles of a protein with standard residues.
    - ``real``: all dihedrals formed by 4 bonded heavy atoms. Meant for all-atom models.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string.
    search_mode : str
        One of ``'virtual'``, ``'protein_backbone'`` or ``'real'``.

    Returns
    -------
    dihedral_labels : list of str
        Dihedral labels. See the function used for each mode for the label format.

    Raises
    ------
    ValueError
        If ``search_mode`` is not supported.
    """
    
    if search_mode == "virtual":
    
        # Find virtual dihedrals
        dihedral_labels = find_virtual_dihedral(topology_path, selection)

    elif search_mode == "protein_backbone":

        # Find protein backbone dihedrals
        dihedral_labels = find_protein_back_dihedrals(topology_path, selection)
    
    elif search_mode == "real":

        # Find real dihedrals
        dihedral_labels = find_all_real_dihedrals(topology_path, selection)

    else:

        raise ValueError(f"search_mode {search_mode} not supported. Options: (virtual, protein_backbone, real)")

    return dihedral_labels

def find_coordinates(topology_path: str, selection: str, stride: int) -> List[str]:
    """
    List the atoms of a selection whose coordinates will be used as features.

    Each atom is returned as a label of the form ``@atomName_atomResid``.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string.
    stride : int
        Keep one of every ``stride`` atoms of the selection.

    Returns
    -------
    all_labels : list of str
        Atom labels.

    Raises
    ------
    ValueError
        If the selection is empty.
    """

    # Load topology
    u = mda.Universe(topology_path)

    # Select atoms
    atoms = u.select_atoms(selection)

    # Keep only one every stride atoms
    atoms = atoms[::stride]

    # Check the selection is not empty
    if len(atoms) == 0:
        raise ValueError(f"Selection: '{selection}' for topology {topology_path} is empty, please review the selection string.")

    all_labels = []

    # Iterate over all selected atoms
    for atom in atoms:

        # Create entity
        entity = f"@{atom.name}_{atom.resid}"

        # Add atomic definition of the coordinate
        all_labels.append(entity)

    return all_labels

def find_virtual_dihedral(topology_path: str, selection: str) -> List[str]:
    """
    Find all virtual dihedrals between consecutive heavy atoms in a selection.

    The atoms are assumed to be connected in the order they appear in the topology,
    so every 4 consecutive atoms define one dihedral. Each dihedral is returned as a label of the form
    ``@atom1Name_atom1Resid-@atom2Name_atom2Resid-@atom3Name_atom3Resid-@atom4Name_atom4Resid``.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string.

    Returns
    -------
    dihedral_labels : list of str
        Dihedral labels.

    Raises
    ------
    ValueError
        If the selection has no heavy atoms.
    """

    # Load topology
    u = mda.Universe(topology_path)

    # Select atoms
    atoms = u.select_atoms(selection)

    # Keep only heavy atoms
    heavy_atoms = atoms.select_atoms("not name H*")

    # Check the selection is not empty
    if len(heavy_atoms) == 0:
        raise ValueError(f"Selection: '{selection}' is empty, please review the selection string.")
    
    dihedral_labels = []

    # Iterate over all CA atoms skipping the first 3
    for i in range(3, len(heavy_atoms)):

        # Create atom label
        label = f"@{atoms[i-3].name}_{atoms[i-3].resid}-@{atoms[i-2].name}_{atoms[i-2].resid}-@{atoms[i-1].name}_{atoms[i-1].resid}-@{atoms[i].name}_{atoms[i].resid}"

        # Add atomic definition of the dihedral
        dihedral_labels.append(label)

    return dihedral_labels

def find_protein_back_dihedrals(topology_path: str, selection: str) -> List[str]:
    """
    Find the protein backbone dihedrals (phi and psi) of the residues in a selection.

    Each dihedral is returned as a label of the form ``@phi_resid`` or ``@psi_resid``.
    The phi angle is skipped if the previous residue is not in the selection, and the psi
    angle is skipped if the next residue is not in the selection.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string. It should select part of a protein.

    Returns
    -------
    dihedral_labels : list of str
        Dihedral labels.
    """

    # Load topology
    u = mda.Universe(topology_path)

    # Select atoms
    atoms = u.select_atoms(selection)

    dihedrals = ['phi', 'psi']

    dihedral_labels = []

    # Find all residues in the list of atoms
    residues = np.unique([atom.resid for atom in atoms])
    
    # Iterate over all residues
    for residue in residues:

        # Iterate over all backbone dihedrals
        for dihedral in dihedrals:

            # Create dihedral label
            label = f"@{dihedral}_{residue}"
            
            if dihedral == 'phi':
                # Check the existence of the previous residue
                if residue -1 not in residues:
                    logger.warning(f"Residue {residue} does not have a previous residue, skipping phi dihedral.")
                    continue
            elif dihedral == 'psi':
                # Check the existence of the next residue
                if residue + 1 not in residues:
                    logger.warning(f"Residue {residue} does not have a next residue, skipping psi dihedral.")
                    continue
                
            # Add dihedral definition
            dihedral_labels.append(label)
    
    return dihedral_labels

def find_all_real_dihedrals(topology_path: str, selection: str) -> List[str]:
    """
    Find all dihedrals formed by 4 bonded heavy atoms in a selection.

    Bonds are taken from the topology if present. Otherwise they are guessed with a distance cutoff.
    Each dihedral is returned as a label of the form
    ``@atom1Name_atom1Resid-@atom2Name_atom2Resid-@atom3Name_atom3Resid-@atom4Name_atom4Resid``.
    Reversed duplicates are only listed once.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string.

    Returns
    -------
    dihedral_labels : list of str
        Dihedral labels.

    Raises
    ------
    ValueError
        If the selection has no heavy atoms.
    """
    # Load topology
    u = mda.Universe(topology_path)

    # Select atoms
    atoms = u.select_atoms(selection)

    # Keep only heavy atoms
    heavy_atom_selection = atoms.select_atoms("not name H*")

    # Check the selection is not empty
    if len(heavy_atom_selection) == 0:
        raise ValueError(f"Selection: '{selection}' is empty, please review the selection string.")
    
    # Check if bonds are present in the topology
    if not hasattr(u, 'bonds'):
        logger.info(f"Topology does not contain bonds. Bonds will be guessed using a distance criterion (bond_length < {covalent_bond_threshold}).")

    # Find set with all indices for fast membership-to-heavy-atoms checking)
    heavy_atom_indices = set(heavy_atom_selection.indices)

    # Dictionary to keep track of bonded atoms
    neighbors_dict = {atom.index: set() for atom in heavy_atom_selection}

    # Dictionary to keep heavy track of heavy atoms
    heavy_atom_dict = {atom.index: atom for atom in heavy_atom_selection}

    # Find bonds in the structure
    if hasattr(heavy_atom_selection, 'bonds'):

        # Extract bonds from the topology
        bonds = heavy_atom_selection.bonds

        # Use the topology to find the bonds
        logger.info(f"Topology contains bonds. Using bonds to find dihedrals.")

        # Fill neighbors_dict using the bonds in the topology
        for bond in bonds:

            i, j = bond.indices

            if i in heavy_atom_indices and j in heavy_atom_indices:
                neighbors_dict[i].add(j)
                neighbors_dict[j].add(i)

    else:

        # Use a distance criterion to guess bonds
        logger.warning(f"Topology does not contain bonds. Using a distance criterion (bond_length < {covalent_bond_threshold}) to guess bonds.")

        # Initialize set to save all bonds
        bonds_indices = set()

        # Fill neighbors_dict using a distance criterion
        for i, atom_i in enumerate(heavy_atom_selection):
            for j, atom_j in enumerate(heavy_atom_selection):

                # Check atoms are different
                if i != j:

                    # Check distance between atoms
                    if calc_bonds(atom_i.position, atom_j.position) < covalent_bond_threshold:
                        
                        if (atom_j.index, atom_i.index) not in bonds_indices:

                            # Add bond to set
                            bonds_indices.add((atom_i.index, atom_j.index))

                            # Add bond to neighbors_dict
                            neighbors_dict[atom_i.index].add(atom_j.index)
                            neighbors_dict[atom_j.index].add(atom_i.index)

        # Add bonds to the Universe
        u.add_TopologyAttr('bonds', bonds_indices)

    # Extract bonds from the topology
    all_bonds = u.bonds

    dihedral_labels = []

    # Iterate over all bonds 
    for bond in all_bonds:
        
        # Get indices of the bonded atoms
        i_index, j_index = bond.indices

        # Check if both atoms are heavy atoms
        if i_index in heavy_atom_indices and j_index in heavy_atom_indices:

            # Get neighbors of each atom
            i_neighbors = neighbors_dict[i_index]
            j_neighbors = neighbors_dict[j_index]

            # For each possible set of 4 bonded atoms around the bond (i, j)
            for i_neighbor in i_neighbors:
                if i_neighbor != j_index:
                    for j_neighbor in j_neighbors:
                        if j_neighbor != i_index and j_neighbor != i_neighbor:

                            # Create dihedral label for atoms with indices i_neighbor, i_index, j_index, j_neighbor
                            atom_1 = f"@{heavy_atom_dict[i_neighbor].name}_{heavy_atom_dict[i_neighbor].resid}"
                            atom_2 = f"@{heavy_atom_dict[i_index].name}_{heavy_atom_dict[i_index].resid}"
                            atom_3 = f"@{heavy_atom_dict[j_index].name}_{heavy_atom_dict[j_index].resid}"
                            atom_4 = f"@{heavy_atom_dict[j_neighbor].name}_{heavy_atom_dict[j_neighbor].resid}"
                            dihedral_label = f"{atom_1}-{atom_2}-{atom_3}-{atom_4}"
                            equivalent_dihedral_label = f"{atom_4}-{atom_3}-{atom_2}-{atom_1}"
                            
                            # Check dihedral is not repeated
                            if dihedral_label not in dihedral_labels and equivalent_dihedral_label not in dihedral_labels:
                                dihedral_labels.append(dihedral_label)

    return dihedral_labels    


# Wrapper to get labels from a features definition dictionary
def get_dihedral_labels(topology_path: str, dihedrals_definition: Dict) -> List[str]:
    """
    Get the feature labels for a group of dihedrals.

    With periodic encoding (the default) each dihedral gives two features, ``sin-<label>`` and
    ``cos-<label>``. Without it, each dihedral gives one ``tor-<label>`` feature.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    dihedrals_definition : dict
        Definition of the group. Keys used: ``selection`` (default ``'all'``),
        ``search_mode`` (default ``'real'``) and ``periodic_encoding`` (default True).

    Returns
    -------
    dihedral_names : list of str
        Dihedral feature labels.
    """

    # Read dihedral group definition
    selection = dihedrals_definition.get('selection', 'all')
    search_mode = dihedrals_definition.get('search_mode', 'real')

    atom_labels = find_dihedrals(topology_path, selection, search_mode)
    
    # Define command labels
    dihedral_names = []
    for label in atom_labels:
        
        if dihedrals_definition.get('periodic_encoding', True):
            dihedral_names.append(f"sin-{label}")
            dihedral_names.append(f"cos-{label}")
        else:
            dihedral_names.append(f"tor-{label}")

    return dihedral_names

def get_distance_labels(topology_path: str, distances_definition: Dict) -> List[str]:
    """
    Get the feature labels for a group of distances.

    Each label has the form ``dist-<atom1>-<atom2>``.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    distances_definition : dict
        Definition of the group. Keys used: ``first_selection``, ``second_selection``
        (default ``'all'``), ``first_stride``, ``second_stride`` (default 1),
        ``skip_neigh_residues`` and ``skip_bonded_atoms`` (default False).

    Returns
    -------
    distance_labels : list of str
        Distance feature labels.
    """

    # Read distance group definition
    selection1 = distances_definition.get('first_selection', 'all')
    selection2 = distances_definition.get('second_selection', 'all')
    stride1 = distances_definition.get('first_stride', 1)
    stride2 = distances_definition.get('second_stride', 1)
    skip_neighbors = distances_definition.get('skip_neigh_residues', False)
    skip_bonded_atoms = distances_definition.get('skip_bonded_atoms', False)

    atom_labels = find_distances(topology_path, selection1, selection2, stride1, stride2, skip_neighbors, skip_bonded_atoms)
    
    # Define command labels
    distance_labels = []
    for label in atom_labels:
        distance_labels.append(f"dist-{label}")

    return distance_labels

def get_coordinate_labels(topology_path: str, coordinate_definition: Dict) -> List[str]:
    """
    Get the feature labels for a group of atomic coordinates.

    Each atom gives three features, one per axis: ``coord-<atom>.x``, ``coord-<atom>.y`` and ``coord-<atom>.z``.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    coordinate_definition : dict
        Definition of the group. Keys used: ``selection`` (default ``'all'``) and ``stride`` (default 1).

    Returns
    -------
    coordinate_labels : list of str
        Coordinate feature labels.
    """
    axis = ['x', 'y', 'z']
    
    # Read coordinate group definition
    selection = coordinate_definition.get('selection', 'all')
    stride = coordinate_definition.get('stride', 1)

    atom_labels = find_coordinates(topology_path, selection, stride)
    
    # Define command labels
    coordinate_labels = []
    for label in atom_labels:
        for ax in axis:
            coordinate_labels.append(f"coord-{label}.{ax}")

    return coordinate_labels


# Wrapper to get the full list of features from a features configuration dictionary
def get_features_list(features_configuration: Dict, topology_path: str) -> List:
    """
    Get the list of feature labels defined in a features configuration for a given topology.

    Each label both names and defines a feature, so it can be turned into the PLUMED command
    that computes it. The list may contain duplicates, which are removed later.

    Parameters
    ----------
    features_configuration : dict
        Features to compute, organized in groups. Supported keys: ``coordinate_groups``,
        ``distance_groups``, ``dihedral_groups`` and ``distance_to_center_groups``.
    topology_path : str
        Path to the topology file.

    Returns
    -------
    features_labels : list of str
        Feature labels.

    Raises
    ------
    ValueError
        If no features are found.
    """
    
    # NOTE: this list may contain duplicated labels, both within a group and between groups,
    # because a label (@atomName_atomResid) is not guaranteed to identify a single atom.
    # Duplicates are dropped downstream before writing a PLUMED input file.

    features_labels = []
    
    logger.debug(f"Searching for features in {Path(topology_path).name}...")
    
    #############
    # POSITIONS #
    #############
    
    # Check for coordinate groups
    if features_configuration.get('coordinate_groups'):
        
        # Find list of groups
        coordinate_group_names = features_configuration['coordinate_groups'].keys()

        # Iterate over groups
        for group_name in coordinate_group_names:
            
            group_definition = features_configuration['coordinate_groups'][group_name]
            
            # Find labels for all coordinates in the group
            coordinate_labels = get_coordinate_labels(topology_path, group_definition)
            
            # Log number of coordinates found
            logger.debug(f"Found {len(coordinate_labels)} features for {group_name}")
            
            # Add labels to the list
            features_labels.extend(coordinate_labels)

    #############
    # DISTANCES #
    #############
    
    # Check for distance groups
    if features_configuration.get('distance_groups'):

        # Find list of groups
        distance_group_names = features_configuration['distance_groups'].keys()

        # Iterate over groups
        for group_name in distance_group_names:    

            # Find group definition
            group_definition = features_configuration['distance_groups'][group_name]

            # Find labels for all distances in the group
            distance_labels = get_distance_labels(topology_path, group_definition)

            # Log number of distances found
            logger.debug(f"Found {len(distance_labels)} features for {group_name}")
            
            # Add labels to the list
            features_labels.extend(distance_labels)

    #############
    # DIHEDRALS #
    #############

    # Check for dihedral groups
    if features_configuration.get('dihedral_groups'):

        # Find list of groups
        dihedral_group_names = features_configuration['dihedral_groups'].keys()

        # Iterate over groups
        for group_name in dihedral_group_names:

            # Find group definition
            group_definition = features_configuration['dihedral_groups'][group_name]
                
            dihedral_labels = get_dihedral_labels(topology_path, group_definition)
            
            # Log number of dihedrals found
            logger.debug(f"Found {len(dihedral_labels)} features for {group_name} (sin and cos of each dihedral for periodic encoding)")
            
            # Add labels to the list
            features_labels.extend(dihedral_labels)

    ######################
    # DISTANCE TO CENTER #
    ######################

    # Check for distance to com groups
    if features_configuration.get('distance_to_center_groups'):

        # Find the list of groups
        distance_to_center_group_names = features_configuration['distance_to_center_groups'].keys()

        # Iterate over groups
        for group_name in distance_to_center_group_names:

            # Find group definition
            group_definition = features_configuration['distance_to_center_groups'][group_name]

            # Get the CENTER command
            center_command_label = f"center_{to_entity_name(group_definition['center_selection'])}"

            # Find atoms in selection to compute the distance to the CENTER 
            atoms = get_indices(topology_path, group_definition['selection'])

            # Create labels
            center_labels = [f"dist-{atom}-{center_command_label}" for atom in atoms]

            logger.debug(f"Found {len(center_labels)} features for {group_name}")
            
            # Add labels to the list
            features_labels.extend(center_labels)
    
    # Check if any features were found
    if len(features_labels) == 0:
        raise ValueError("No features found, please check the features section of the configuration file and the topology.")
    
    # Log reference features list
    logger.debug(f"The feature list contains {len(features_labels)} features:")
    logger.debug(features_labels)
 
    return features_labels
    
# Working with trajectories
def extract_XTC(trajectory_path: str, topology_path: str, traj_frames: list, new_traj_path: str):
    """
    Save some frames of a trajectory to a new XTC file.

    Frames are written in increasing order. Nothing is written if ``traj_frames`` is empty.

    Parameters
    ----------
    trajectory_path : str
        Path to the original trajectory file.
    topology_path : str
        Path to the topology file.
    traj_frames : list of int
        Indices of the frames to extract (starting at 0).
    new_traj_path : str
        Path to the new XTC file.
    """
    
    # Check if any traj_frames were requested
    if len(traj_frames) == 0:
        logger.warning(f"No frames requested for {new_traj_path}.")
        return
    
    # Load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)
        
    # Make sure traj_frames is a list
    if not isinstance(traj_frames, list):
        traj_frames = list(traj_frames)
        
    # Order the list of traj_frames by default
    traj_frames.sort()
    
    # Save new trajectory to an XTC file
    with mda.Writer(new_traj_path, n_atoms=u.atoms.n_atoms, format='XTC') as writer:
        for frame in traj_frames:
            u.trajectory[frame]
            writer.write(u)

def extract_PDB(trajectory_path: str, topology_path: str, pdb_frame: int, pdb_path: str):
    """
    Save one frame of a trajectory to a PDB file, without CONECT records.

    Parameters
    ----------
    trajectory_path : str
        Path to the original trajectory file.
    topology_path : str
        Path to the topology file.
    pdb_frame : int
        Index of the frame to extract (starting at 0).
    pdb_path : str
        Path to the new PDB file.
    """
    
    # Load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)
    
    # Save new topology to a temporary PDB file including CONECT records
    tmp_topology_path = os.path.join(Path(pdb_path).parent, "tmp.pdb")
    
    # Write temporary PDB topology
    with mda.Writer(tmp_topology_path, n_atoms=u.atoms.n_atoms, format='PDB') as writer:
        u.trajectory[pdb_frame]
        writer.write(u)

    # Remove CONECT records from the temporary topology
    with open(tmp_topology_path, 'r') as f:
        lines = f.readlines()

    with open(pdb_path, 'w') as f:
        for line in lines:
            if not line.startswith("CONECT"):
                f.write(line)
                
    # Remove the temporary topology file
    if os.path.exists(tmp_topology_path):
        os.remove(tmp_topology_path)

def get_num_frames(trajectory_path: str, topology_path: str) -> int:
    """
    Get the number of frames in a trajectory.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file.
    topology_path : str
        Path to the topology file.

    Returns
    -------
    num_frames : int
        Number of frames in the trajectory.
    """
    
    # Load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)
    
    # Get number of frames
    num_frames = len(u.trajectory)
    
    return num_frames

def get_number_atoms(topology: str, selection: str = None) -> int:
    """
    Get the number of atoms in a selection of a topology.

    Parameters
    ----------
    topology : str
        Path to the topology file.
    selection : str, optional
        MDAnalysis selection string. If None, all atoms are counted.

    Returns
    -------
    num_atoms : int
        Number of atoms in the selection.
    """
    
    # Load topology
    u = mda.Universe(topology)

    # Select atoms
    if selection is None:
        atoms = u.select_atoms('all')
    else:
        atoms = u.select_atoms(selection)

    # Get number of atoms
    num_atoms = len(atoms)

    return num_atoms

def get_indices(topology: str, selection: Union[str, None] = None) -> list:
    """
    Get the indices of the atoms in a selection of a topology.

    Indices start at 1, as in PLUMED and most MD engines (MDAnalysis starts at 0).

    Parameters
    ----------
    topology : str
        Path to the topology file.
    selection : str, optional
        MDAnalysis selection string. If None, all atoms are used.

    Returns
    -------
    indices : list of int
        1-based indices of the selected atoms.
    """

    # Load topology
    u = mda.Universe(topology)

    # Select atoms
    if selection is None:
        atoms = u.select_atoms('all')
    else:
        atoms = u.select_atoms(selection)

    # Get indices
    indices = atoms.indices

    # Convert to list
    indices = list(indices)

    # Convert to ints
    indices = [int(index)+1 for index in indices]

    return indices

def load_coordinates(
    topology_file: str, 
    trajectory_file: str, 
    selection: str = "all", 
    prepare_trajectory: bool = False,
    start: Optional[int] =None,
    stop: Optional[int] = None,
    step: Optional[int] = None
):
    """
    Load the coordinates of a selection of atoms along a trajectory.

    Parameters
    ----------
    topology_file : str
        Path to the topology file (pdb, gro, tpr, etc.).
    trajectory_file : str
        Path to the trajectory file (xtc, dcd, trr, etc.).
    selection : str, optional
        MDAnalysis selection string. Default is ``"all"``.
        Use ``"name CA"`` to keep only C-alpha atoms.
    prepare_trajectory : bool, optional
        If True, unwrap and center the trajectory before reading it. Default is False.
    start, stop, step : int, optional
        Slice of frames to read. Default is None (read all frames).

    Returns
    -------
    frame_array : np.ndarray
        Shape (n_frames,). Position of each frame in the slice (0, 1, 2, ...), as floats.
        This is not the simulation time.
    coords_array : np.ndarray
        Shape (n_frames, n_atoms, 3). Coordinates of the selected atoms.
    """
    
    # Load Universe
    u = load_universe(topology_file, trajectory_file, selection, prepare_trajectory)

    # Pre-allocate Arrays
    # We define the slice of frames we want to read
    atom_group = u.select_atoms(selection)
    trajectory_slice = u.trajectory[start:stop:step]
    n_frames = len(trajectory_slice)
    n_atoms = len(atom_group)
        
    # Shape: (Time, Atoms, Coordinates) -> (N, M, 3)
    coords_array = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    frame_array = np.zeros(n_frames, dtype=np.float32)

    # Read coordinates
    for i, _ in enumerate(trajectory_slice):
        coords_array[i] = atom_group.positions
        frame_array[i] = float(i)

    return frame_array, coords_array

def load_universe(topology_file: str, 
                  trajectory_file: str, 
                  selection: str = "all",
                  prepare_trajectory: bool = False
    ) -> mda.Universe:
    """
    Load an MDAnalysis Universe from topology and trajectory files.

    Bonds are guessed on loading. If requested, unwrapping and centering are applied on the fly
    to the selected atoms, when the topology has bonds and the trajectory has box dimensions.

    Parameters
    ----------
    topology_file : str
        Path to the topology file (pdb, gro, tpr, etc.).
    trajectory_file : str
        Path to the trajectory file (xtc, dcd, trr, etc.).
    selection : str, optional
        MDAnalysis selection string used for the transformations. Default is ``"all"``.
        The program exits if it matches no atoms.
    prepare_trajectory : bool, optional
        If True, unwrap and center the trajectory. Default is False.

    Returns
    -------
    MDAnalysis.Universe
        The loaded Universe.
    """
    
    # Load the Universe
    try:
        u = mda.Universe(topology_file, trajectory_file, guess_bonds=True)
    except Exception as e:
        logger.error(f"Could not load universe with provided files. Error: {e}")
    
    # Select Atoms
    selected_atom_group = u.select_atoms(selection)
    if len(selected_atom_group) == 0:
        logger.error(f"Selection '{selection}' matched 0 atoms.")
        sys.exit(1)
        
    logger.info(f"Loading {len(selected_atom_group)} atoms from {u.trajectory.n_frames} frames.")

    # Trajectory preparation
    preparation_steps = []
    
    if prepare_trajectory:
        logger.info("Preparing trajectory: applying unwrapping and centering transformations.")
    
        # Unwrap trajectory if bonds are present
        if len(u.bonds) == 0:
            logger.warning("Topology does not contain bonds. Cannot unwrap trajectory.")
        else:
            logger.debug("Topology contains bonds. Unwrapping trajectory.")
            try:
                preparation_steps.append(trans.unwrap(selected_atom_group))
            except Exception as e:
                logger.warning(f"Could not unwrap trajectory. Error: {e}")
                logger.warning("Make sure your trajectory has been unwrapped properly.")
        
        # Center trajectory if box dimensions are present
        if u.dimensions is not None:
            logger.debug("Trajectory contains box dimensions. Centering trajectory.")
            try:
                preparation_steps.append(trans.center_in_box(selected_atom_group, wrap=True))
            except Exception as e:
                logger.warning(f"Could not center trajectory. Error: {e}")
                logger.warning("Make sure your trajectory has been centered properly.")

    if preparation_steps:
        u.trajectory.add_transformations(*preparation_steps)
    
    return u

def interpolate_trajectory(
    topology_file: str,
    trajectory_file: str,
    num_frames: int,
    keep_original_frames: bool = True,
    interpolation_method: Optional[Literal['akima', 'pchip']] = 'pchip',
    noise_std: Optional[float] = None,
    random_seed: int = 42,
    atom_selection: str = 'all',
    traj_format: Literal['xtc', 'dcd', 'nc', 'pdb'] = 'xtc',
    prepare_trajectory: bool = False,
    output_path: Optional[str] = None,
    suffix: str = "",
    ) -> Tuple[str, str]:
    """
    Create a new trajectory with more (or fewer) frames by interpolating between frames.

    Useful to get smooth, denser trajectories, for example from a few structures along a path.
    If the output files already exist, they are reused and nothing is computed.

    Parameters
    ----------
    topology_file : str
        Path to the topology file (pdb, gro, tpr, etc.).
    trajectory_file : str
        Path to the trajectory file (xtc, dcd, trr, etc.).
    num_frames : int
        Number of frames in the new trajectory.
    keep_original_frames : bool, optional
        If True, keep the original frames and add interpolated frames between them.
        If False, use only evenly spaced interpolated frames. Default is True.
    interpolation_method : str, optional
        ``'akima'`` or ``'pchip'``. Default is ``'pchip'``. Akima looks smoother, while pchip
        avoids overshooting when there are abrupt changes. If None, no interpolation is done
        and the original frames are used.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the coordinates. Default is None (no noise).
    random_seed : int, optional
        Seed for the noise, for reproducibility. Default is 42.
    atom_selection : str, optional
        MDAnalysis selection string. Only these atoms are kept. Default is ``"all"``.
    traj_format : str, optional
        Format of the new trajectory: ``'xtc'``, ``'dcd'``, ``'nc'`` or ``'pdb'``. Default is ``'xtc'``.
    prepare_trajectory : bool, optional
        If True, unwrap and center the trajectory before interpolating. Default is False.
    output_path : str, optional
        Folder where the new files are saved. If None, the current directory is used.
    suffix : str, optional
        Text added to the output file names before the extension (e.g. ``"_rep0"``). Default is ``""``.

    Returns
    -------
    new_trajectory_path : str
        Path to the new trajectory file.
    new_topology_path : str
        Path to a PDB topology with only the selected atoms, matching the new trajectory.
    """

    from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
    from MDAnalysis.coordinates.memory import MemoryReader

    # Get the trajectory name without extension
    traj_name = Path(trajectory_file).stem
    new_traj_path = os.path.join(output_path if output_path else ".", f"{traj_name}_augmented_{interpolation_method}{suffix}.{traj_format}")
    new_top_path = os.path.join(output_path if output_path else ".", f"{traj_name}_augmented_{interpolation_method}{suffix}.pdb")

    # Check if the output files already exist
    if os.path.exists(new_traj_path) and os.path.exists(new_top_path):
        logger.info(f"Interpolated trajectory and topology already exist at {new_traj_path} and {new_top_path}. Skipping interpolation.")
        return new_traj_path, new_top_path
        
    # Load the trajectory using MDAnalysis
    frames, coords = load_coordinates(topology_file, trajectory_file, atom_selection, prepare_trajectory)
    
    # Define new frames
    if keep_original_frames:
        # Generate additional frames 
        additional_frames = np.linspace(frames[0], frames[-1], num_frames - len(frames) + 2)[1:-1]
        # Merge with original frames and sort - unevenly spaced frames
        new_frames = np.sort(np.concatenate((frames, additional_frames)))
    else:
        # Generate new frames - evenly spaced
        new_frames = np.linspace(frames[0]+.5, frames[-1]+.5, num_frames)
    
    # Interpolate coordinates
    if interpolation_method == 'akima':
        interpolator = Akima1DInterpolator(x=frames, y=coords, axis=0, method='makima')
    elif interpolation_method == 'pchip':
        interpolator = PchipInterpolator(x=frames, y=coords, axis=0)
    elif interpolation_method is None:
        new_coords = coords
    else:
        logger.error(f"Interpolation method '{interpolation_method}' not supported. Use 'akima' or 'pchip'.")
    
    if interpolation_method is not None:
        new_coords = interpolator(new_frames)
    
    # Add noise if specified
    if noise_std is not None:
        # Seed for reproducibility
        np.random.seed(random_seed)
        noise = np.random.normal(0, noise_std, new_coords.shape)
        new_coords += noise
    
    # Create a matching topology for the selection
    # Load the original universe to grab the atom information
    u_orig = mda.Universe(topology_file)
    selected_atoms = u_orig.select_atoms(atom_selection)
    # Write just the selected atoms to the new topology PDB
    selected_atoms.write(new_top_path)
    
    # Create a new trajectory file
    u = mda.Universe(new_top_path)
    u.load_new(new_coords, format=MemoryReader)
    with mda.Writer(new_traj_path, n_atoms=u.atoms.n_atoms) as Writer:
        for ts in u.trajectory:
            Writer.write(u)
    
    return new_traj_path, new_top_path
    
# I/O functions
def find_supported_traj(parent_path, filename = None):
    """
    Find the trajectory files in a folder with formats supported by MDAnalysis.

    Parameters
    ----------
    parent_path : str
        Path to the folder with trajectories.
    filename : str, optional
        Glob pattern to select a subset of files (e.g. ``"*.xtc"``). Default is None (all files).

    Returns
    -------
    all_supported_trajs : list of str
        Sorted paths to the supported trajectories.
    """

    # List supported formats (MDAnalysis supported)
    traj_formats = ['.xtc', '.pdb', '.xpdb', '.pdbqt', '.parmed', '.ncdf', '.nc', '.dcd', '.xyz', '.arc', '.trr', '.tng', '.nc', '.netcdf', '.mdcrd', '.crd', '.h5', '.hdf5', '.lh5']

    if filename is None:
        filename = "*"  

    # Find all ensemble trajectories using generic name given in global_parameters
    all_trajectories = glob.glob(os.path.join(parent_path, filename))

    # Check if all_trajectories is None
    if all_trajectories is None or len(all_trajectories) == 0:
        logger.warning(f"No trajectories found in {parent_path}.")

    # Find all trajectories that comply with supported formats
    all_supported_trajs = [file for file in all_trajectories if Path(file).suffix in traj_formats]

    # Complain if no supported trajectories are found
    if len(all_supported_trajs) == 0:
        logger.warning(f"No supported trajectories found in {parent_path}.")

    all_supported_trajs.sort()

    return all_supported_trajs

def find_supported_top(parent_path, filename = None):
    """
    Find the topology files in a folder with formats supported by MDAnalysis.

    Parameters
    ----------
    parent_path : str
        Path to the folder with topologies.
    filename : str, optional
        Glob pattern to select a subset of files (e.g. ``"*.pdb"``). Default is None (all files).

    Returns
    -------
    all_supported_tops : list of str
        Sorted paths to the supported topologies.
    """

    # List supported formats
    top_formats = ['.pdb', '.gro', '.crd', '.data', '.pqr', '.pdbqt', '.mol2', '.psf', '.prmtop', '.parm7', '.top', '.itp']

    if filename is None:
        filename = "*"

    # Find all ensemble topologies using generic name given in global_parameters
    all_topologies = glob.glob(os.path.join(parent_path, filename))

    # Check if topology is None
    if all_topologies is None or len(all_topologies) == 0:
        logger.warning(f"No topology found in {parent_path}.")
    
    # Find all topologies that comply with supported formats
    all_supported_tops = [file for file in all_topologies if Path(file).suffix in top_formats]

    # Complain no supported topology is found
    if len(all_supported_tops) == 0:
        logger.warning(f"No supported topology found in {parent_path}.")
    
    # Sort them to make sure we pair the right trajectory and topology
    all_supported_tops.sort()

    return all_supported_tops

def create_pdb(structure_path, file_name):
    """
    Convert a structure file to PDB format using MDAnalysis.

    Parameters
    ----------
    structure_path : str
        Path to the structure file.
    file_name : str
        Path to the new PDB file.
    """

    # Load structure
    u = mda.Universe(structure_path)

    # Save structure in PDB format
    u.atoms.write(file_name)

    return

def create_plumed_rmsd_template(
    topology_path: str, 
    output_path: str, 
    align_selection: str = 'backbone', 
    rmsd_selection: str = 'backbone'
    ):
    """
    Create a PDB template for PLUMED RMSD-type commands (e.g. FIT_TO_TEMPLATE or RMSD).

    PLUMED reads the occupancy column to know which atoms to align and the B-factor column to know
    which atoms to use for the RMSD. Selected atoms get 1.00 and the rest get 0.00.

    Parameters
    ----------
    topology_path : str
        Path to the topology file.
    output_path : str
        Path to the new PDB template.
    align_selection : str, optional
        MDAnalysis selection string for the atoms used to align. Default is ``'backbone'``.
    rmsd_selection : str, optional
        MDAnalysis selection string for the atoms used to compute the RMSD. Default is ``'backbone'``.

    Raises
    ------
    ValueError
        If any of the selections is empty.
    """

    # Load topology
    u = mda.Universe(topology_path)

    # Select atoms for alignment
    align_atoms = u.select_atoms(align_selection)

    # Select atoms for RMSD calculation
    rmsd_atoms = u.select_atoms(rmsd_selection)

    # Check selections are not empty
    if len(align_atoms) == 0:
        raise ValueError(f"Selection: '{align_selection}' for topology {topology_path} is empty, please review the selection string.")

    if len(rmsd_atoms) == 0:
        raise ValueError(f"Selection: '{rmsd_selection}' for topology {topology_path} is empty, please review the selection string.")

    # Create a new PDB file in which 
    # Alignment atoms have a 1.00 occupancy
    # RMSD atoms have a 1.00 B-factor
    # Other atoms have 0.00 occupancy and B-factor
    with mda.Writer(output_path, n_atoms=u.atoms.n_atoms, format='PDB') as writer:
        for atom in u.atoms:
            if atom in align_atoms:
                atom.occupancy = 1.00
            else:
                atom.occupancy = 0.00

            if atom in rmsd_atoms:
                atom.bfactor = 1.00
            else:
                atom.bfactor = 0.00

        writer.write(u)

    return

def create_rmsd_waypoint_reference(waypoint_structures: List[str], 
                                   plumed_topology_path: str, 
                                   rmsd_restraint_reference_path: str,
                                   align_waypoint_structures: Optional[bool] = True,
                                   distance_threshold: Optional[float] = 2.0):
    """
    Create a PDB reference for a PLUMED RMSD restraint from a set of waypoint structures.

    Only C-alpha atoms of residues present in all waypoints are considered. A residue is "stable" if its
    C-alpha moves at most ``distance_threshold`` Angstroms between any two waypoints. Stable C-alpha atoms
    get 1.0 in the occupancy and B-factor columns; all other atoms get 0.0.

    Parameters
    ----------
    waypoint_structures : list of str
        Paths to the waypoint structures. The first one is the reference for alignment.
    plumed_topology_path : str
        Path to the topology used by PLUMED. The output uses its atoms and numbering.
    rmsd_restraint_reference_path : str
        Path to the new PDB reference.
    align_waypoint_structures : bool, optional
        If True, align the waypoints to the first one before comparing them. Default is True.
    distance_threshold : float, optional
        Maximum displacement (Angstroms) for a residue to be stable. Default is 2.0.
    """
    
    # 1. Map all waypoints to the PLUMED topology ("Universal Reference" for mapping)
    mappings = []
    for wp in waypoint_structures:
        mapper = PDBTopologyMapper(reference_topology=plumed_topology_path, topology=wp)
        mappings.append(mapper.mapping)

    # 2. Find common resids that exist in ALL waypoints AND the PLUMED topology
    # The keys in mapper.mapping are the resids of the plumed_topology
    common_plumed_resids = set(mappings[0].keys())
    for m in mappings[1:]:
        common_plumed_resids &= set(m.keys())
    
    sorted_common_resids = sorted(list(common_plumed_resids))
    
    # 3. Load Waypoints into MDAnalysis for alignment
    wp_universes = [mda.Universe(wp) for wp in waypoint_structures]
    
    # Extract coordinates for the common residues (CA atoms)
    # We need to map the PLUMED resid back to the specific WP resid
    coords_list = []
    for i, u in enumerate(wp_universes):
        wp_mapping = mappings[i]
        # Get the resids in the current waypoint PDB that correspond to our common set
        current_wp_resids = [wp_mapping[r][2] for r in sorted_common_resids]
        
        # Select CA atoms. Note: We use the index-based mapping to ensure correct order
        selection_string = "resid " + " ".join([str(r) for r in current_wp_resids]) + " and name CA"
        ca_atoms = u.select_atoms(selection_string)
        
        # Safety check: ensure we got the exact same number of atoms
        if len(ca_atoms) != len(sorted_common_resids):
            logger.warning(f"Waypoint {waypoint_structures[i]} missing some CA atoms for common residues.")
        
        coords_list.append(ca_atoms.positions)

    # 4. Perform Alignment (Conditional)
    ref_coords = coords_list[0]
    aligned_coords = [ref_coords]

    if align_waypoint_structures:
        logger.info("Aligning waypoints to the first structure for stability check...")
        for i in range(1, len(coords_list)):
            # Calculate rotation/translation to align coords_list[i] to ref_coords
            mobile_coords = coords_list[i]
            R, residue = MDAnalysis.analysis.align.rotation_matrix(mobile_coords, ref_coords)
            # Apply transformation: (coords - centroid) @ R + ref_centroid
            aligned = (mobile_coords - mobile_coords.mean(axis=0)) @ R.T + ref_coords.mean(axis=0)
            aligned_coords.append(aligned)
    else:
        logger.info("Skipping alignment (using raw coordinates)...")
        # Simply append the rest of the coordinates without modification
        for i in range(1, len(coords_list)):
            aligned_coords.append(coords_list[i])

    # 5. Compute all pairwise distances between the aligned waypoints for each residue
    aligned_coords = np.array(aligned_coords)  # Shape: (num_waypoints, num_residues, 3)
    num_residues = aligned_coords.shape[1]
    stable_residues = []
    
    for resid_idx in range(num_residues):
        # Extract coordinates for this residue across all waypoints
        residue_coords = aligned_coords[:, resid_idx, :]  # Shape: (num_waypoints, 3)
        
        # Compute pairwise distances
        max_distance = 0.0
        for i in range(len(residue_coords)):
            for j in range(i + 1, len(residue_coords)):
                dist = np.linalg.norm(residue_coords[i] - residue_coords[j])
                if dist > max_distance:
                    max_distance = dist
        
        # Check if max distance is within threshold
        if max_distance <= distance_threshold:
            stable_residues.append(sorted_common_resids[resid_idx])

    # 6. Create the PLUMED Reference PDB
    # Load the original PLUMED topology
    plumed_u = mda.Universe(plumed_topology_path)
    
    # Initialize all Occupancy and B-factors to 0
    plumed_u.atoms.occupancies = 0.0
    plumed_u.atoms.tempfactors = 0.0
    
    if stable_residues:
        # Set 1.0 for CA atoms of the stable common residues
        final_selection = "resid " + " ".join([str(r) for r in stable_residues]) + " and name CA"
        target_atoms = plumed_u.select_atoms(final_selection)
        target_atoms.occupancies = 1.0
        target_atoms.tempfactors = 1.0
        logger.info(f"Reference structure created with {len(target_atoms)} active atoms.")
    else:
        logger.warning("No stable residues found within the distance threshold!")
    
    # Save the file
    plumed_u.atoms.write(rmsd_restraint_reference_path)
    logger.info(f"Reference structure created with {len(target_atoms)} active atoms.")

def RMSD(trajectory_path: str, 
         topology_path: str, 
         selection: str = "backbone",
         fitting_selection: str = "backbone", 
         reference_path: Optional[str] = None
    ) -> np.array:
    """
    Compute the RMSD of each frame of a trajectory with respect to a reference structure.

    Each frame is first fitted using ``fitting_selection`` and then the RMSD of ``selection`` is computed.
    The reference and trajectory topologies may use different residue numbers; residues are matched
    automatically before selecting atoms.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file.
    topology_path : str
        Path to the topology file.
    selection : str, optional
        MDAnalysis selection string for the atoms used in the RMSD. Default is ``"backbone"``.
    fitting_selection : str, optional
        MDAnalysis selection string for the atoms used in the fit. Default is ``"backbone"``.
    reference_path : str, optional
        Path to the reference structure. If None, the topology file is used as reference.

    Returns
    -------
    rmsd : np.ndarray
        RMSD value for each frame. Empty if no common residues are found or the selections do not match.
    """
    
    u = mda.Universe(topology_path, trajectory_path)
    ref_structure = reference_path if reference_path else topology_path
    ref = mda.Universe(ref_structure)

    # 1. Map topologies to get the pairs
    mapper = PDBTopologyMapper(ref_structure, topology_path)
    mapping_pairs = [(ref_id, val[2]) for ref_id, val in mapper.mapping.items()]

    if not mapping_pairs:
        logger.error(f"No common residues found between {ref_structure} and {topology_path}")
        return np.array([])

    # 2. Build distinct strings for the two different numbering systems
    ref_res_str = "resid " + " ".join([str(p[0]) for p in mapping_pairs])
    sim_res_str = "resid " + " ".join([str(p[1]) for p in mapping_pairs])

    # 3. Create Tuples for the RMSD class
    # Format: (mobile_selection, reference_selection)
    fit_tuple = (
        f"({fitting_selection}) and ({sim_res_str})", 
        f"({fitting_selection}) and ({ref_res_str})"
    )
    
    analysis_tuple = (
        f"({selection}) and ({sim_res_str})", 
        f"({selection}) and ({ref_res_str})"
    )
    
    # Check the simulation and reference selections are equal and not empty
    ref_atoms = ref.select_atoms(analysis_tuple[1])
    sim_atoms = u.select_atoms(analysis_tuple[0])
    if len(ref_atoms) == 0 or len(sim_atoms) == 0:
        logger.error(f"Selections resulted in zero atoms. Please check the selection strings.")
        return np.array([])
    if len(ref_atoms) != len(sim_atoms):
        logger.error(f"Number of atoms in simulation ({len(sim_atoms)}) and reference ({len(ref_atoms)}) selections do not match.")
        return np.array([])

    # 4. Initialize the RMSD class using the Tuples
    # 'select' handles the fitting (superposition)
    # 'groupselections' handles the actual RMSD calculation after fitting
    R = MDAnalysis.analysis.rms.RMSD(
        u, 
        ref, 
        select=fit_tuple,         # This performs the fit
        groupselections=[analysis_tuple]  # This calculates the value
    ).run()
    
    # Column 0: Frame, Column 1: Time, Column 2: Fit RMSD, Column 3: Groupselection RMSD
    # We return Column 3 because it represents the actual requested selection
    return R.results.rmsd.T[3]
 
def RMSF(trajectory_path: str, topology_path: str, selection: str, fitting_selection: str) -> np.array:
    """
    Compute the RMSF of each residue in a selection.

    The trajectory is first aligned to its average structure using ``fitting_selection``.
    The per-atom RMSF is then averaged over the atoms of each residue.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file.
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string for the atoms used in the RMSF.
    fitting_selection : str
        MDAnalysis selection string for the atoms used in the alignment.

    Returns
    -------
    rmsf_per_residue : list of float
        Mean RMSF of each residue.
    residues : list of int
        Residue numbers, in the same order as ``rmsf_per_residue``.
    """

    # Load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)
        
    # Align trajectory to the average structure
    average = MDAnalysis.analysis.align.AverageStructure(u, u, select=fitting_selection, ref_frame=0).run()
    ref = average.results.universe
    MDAnalysis.analysis.align.AlignTraj(u, ref, select=fitting_selection, in_memory=True).run()
        
    # Select the atoms to compute the RMSF for
    rmsf_atoms = u.select_atoms(selection)
    
    # Calculate the RMSF
    R = MDAnalysis.analysis.rms.RMSF(rmsf_atoms).run()

    rmsf_per_atom = R.results.rmsf
    
    residues = list(set(rmsf_atoms.resnums))
    
    rmsf_per_residue = []
    for residue in residues:
        rmsf_per_residue.append(np.mean(rmsf_per_atom[rmsf_atoms.resnums == residue]))
    
    return rmsf_per_residue, residues

def dRMSD(trajectory_path: str, topology_path: str, selection: str, selection_stride: int, reference_path: str, output_path: str) -> np.array:
    """
    Compute the distance-RMSD of each frame of a trajectory with respect to a reference structure.

    The dRMSD compares all pairwise distances between the selected atoms in each frame with the same
    distances in the reference. It does not need any alignment. Distances are computed with PLUMED.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file.
    topology_path : str
        Path to the topology file.
    selection : str
        MDAnalysis selection string for the atoms used in the distances.
    selection_stride : int
        Keep one of every ``selection_stride`` atoms of the selection.
    reference_path : str
        Path to the reference structure file.
    output_path : str
        Folder to save the intermediate files.

    Returns
    -------
    drmsd : np.ndarray
        dRMSD value for each frame.
    """
    
    from deep_cartograph.tools import compute_features
    from deep_cartograph.modules.plumed.colvars import read_features, read_column_names
    
    # Define configuration for pairwise distances calculation
    distance_group = {
        'first_selection' : selection,
        'second_selection' : selection, 
        'first_stride' : selection_stride,
        'second_stride' : selection_stride,
        'skip_neigh_residues': True,
        'skip_bonded_atoms': True
    }
        
    config = {
        'plumed_settings': {
            'features': {
                'distance_groups' : {
                    'distances': distance_group
                }
            }
        }
    }
    
    # Compute distances between selected atoms along the trajectory and the reference
    traj_colvars_paths = compute_features(configuration=config,
                                          trajectory_data=[trajectory_path, reference_path],
                                          topology_data=[topology_path, reference_path],
                                          reference_topology=reference_path,
                                          output_folder=os.path.join(output_path, 'compute_features'))

    ref_feature_names = read_column_names(traj_colvars_paths[1])
    
    # Load distances during trajectory
    traj_distance_df = read_features(traj_colvars_paths[0], 
                                     ref_feature_names=ref_feature_names,
                                     topology_paths=[topology_path],
                                     reference_topology=reference_path)
    
    # Load reference distances
    ref_distance_df = read_features(traj_colvars_paths[1],
                                    ref_feature_names=ref_feature_names,
                                    topology_paths=[reference_path],
                                    reference_topology=reference_path)
    
    # Remove time column if present
    if 'time' in traj_distance_df.columns:
        traj_distance_df = traj_distance_df.drop(columns=['time'])
    if 'time' in ref_distance_df.columns:
        ref_distance_df = ref_distance_df.drop(columns=['time'])
    
    # Calculate dRMSD for each frame
    drmsd_values = []
    ref_distances = ref_distance_df.iloc[0].values  # Reference distances are constant
    for index, row in traj_distance_df.iterrows():
        traj_distances = row.values
        drmsd = np.sqrt(np.mean((traj_distances - ref_distances) ** 2))
        drmsd_values.append(drmsd)
    
    return np.array(drmsd_values)

def atom_entity_to_index(atom_entity: str, topology_path: str) -> int:
    """
    Get the MDAnalysis index (starting at 0) of the atom given by an atom label.

    Parameters
    ----------
    atom_entity : str
        Atom label of the form ``@atomName_atomResid`` (e.g. ``"@CA_256"``).
    topology_path : str
        Path to the topology file.

    Returns
    -------
    atom_index : int
        Index of the atom in the topology. If several atoms match, the first one is used.

    Raises
    ------
    ValueError
        If no atom matches the label.
    """
    
    # Find atom name and resid from the entity name
    atom_name = atom_entity.split('_')[0][1:]  # Remove '@' and get the atom name
    resid = int(atom_entity.split('_')[1])      # Get the resid
    
    # Load topology
    u = mda.Universe(topology_path)
    
    # Select the atom based on name and resid
    atom = u.select_atoms(f"name {atom_name} and resid {resid}")
    
    # Check if the atom was found
    if len(atom) == 0:
        logger.error(f"Atom entity '{atom_entity}' not found in topology '{topology_path}'.")
        raise ValueError(f"Atom entity '{atom_entity}' not found in topology '{topology_path}'.")
    
    # Return the index of the atom (MDAnalysis indices start at 0)
    return int(atom.indices[0])

def map_sensitivity_to_structure(
    per_atom_sensitivities: Dict[int, float],
    topology_path: str,
    output_folder: str
    ) -> None:
    """
    Write a PDB file with per-atom sensitivity values in the B-factor column.

    Values are scaled between 0 and 100 so the structure can be colored in PyMOL or VMD.
    Atoms without a value get 0. The file is saved as ``sensitivity_structure.pdb``.

    Parameters
    ----------
    per_atom_sensitivities : dict
        Sensitivity value for each atom, keyed by MDAnalysis atom index (starting at 0).
        The dictionary is modified in place with the scaled values.
    topology_path : str
        Path to the topology file.
    output_folder : str
        Folder where the PDB file is saved.
    """
    
    # Take all the sensitivity values and scale them between 0.0 and 100.0
    sens_values = np.array(list(per_atom_sensitivities.values()))
    
    # Check all sensitivity values are positive
    if np.any(sens_values < 0):
        logger.warning("Some sensitivity values are negative. They will be set to 0.0 for visualization.")
        sens_values[sens_values < 0] = 0.0

    # Find indices of atoms and min/max sensitivity values
    atom_indices = list(per_atom_sensitivities.keys())
    min_sens = np.min(sens_values)
    max_sens = np.max(sens_values)
    
    # Scale the sensitivity value to be between 0.0 and 100.0
    for atom_index in atom_indices:
        sens_value = per_atom_sensitivities[atom_index]
        scaled_value = (sens_value - min_sens) / (max_sens - min_sens) * 100.0
        per_atom_sensitivities[atom_index] = scaled_value

    # Load topology
    u = mda.Universe(topology_path)
    
    # New structure file with sensitivity values in the B-factor column
    new_structure_path = os.path.join(output_folder, "sensitivity_structure.pdb")
    
    # Create a new PDB file with the sensitivity values in the B-factor column
    with mda.Writer(new_structure_path, n_atoms=u.atoms.n_atoms, format='PDB') as writer:
        for atom in u.atoms:
            if atom.index in per_atom_sensitivities:
                atom.bfactor = per_atom_sensitivities[atom.index]
            else:
                atom.bfactor = 0.0  # Default value for atoms not in the sensitivity dictionary
        writer.write(u)
    
    return 

# Other
def to_entity_name(mda_selection: str) -> str:
    """
    Turn an MDAnalysis selection string into a name that can be used in feature labels.

    Spaces and symbols are replaced by text (e.g. ``' '`` by ``'_'`` and ``'-'`` by ``'minus'``).
    See ``mda_to_entity_map``.

    Parameters
    ----------
    mda_selection : str
        MDAnalysis selection string.

    Returns
    -------
    entity_name : str
        Cleaned entity name.
    """
    
    for key, value in mda_to_entity_map.items():
        mda_selection = mda_selection.replace(key, value)
        
    return mda_selection
    
def to_mda_selection(entity_name: str) -> str:
    """
    Turn an entity name back into an MDAnalysis selection string.

    This is the reverse of ``to_entity_name``.

    Parameters
    ----------
    entity_name : str
        Entity name string.

    Returns
    -------
    mda_selection : str
        MDAnalysis selection string.
    """
    
    for key, value in mda_to_entity_map.items():
        entity_name = entity_name.replace(value, key)
        
    return entity_name
    
mda_to_entity_map = {
    ' ': '_',
    ':': 'to',
    '-': 'minus',
    '<': 'lt',
    '>': 'gt',
    '==': 'eq',
    '<=': 'leq',
    '>=': 'geq',
    '!=': 'neq'
}