# Import modules
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Literal
from pathlib import Path

import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.analysis.align
from MDAnalysis.lib.distances import calc_bonds

import deep_cartograph.modules.plumed as plumed

# Set logger
logger = logging.getLogger(__name__)

# Constants for the module
covalent_bond_threshold = 2.0


# To find labels from topologies and MDAnalysis selections
def find_distances(topology_path: str, selection1: str, selection2: str, stride1: int, stride2: int, skip_neighbors: bool, skip_bonded_atoms: bool) -> List[str]:
    '''
    The function finds all the pairwise distances between two selections in the topology, skipping bonded atoms or atoms pertaining to neighboring residues if requested.
    
    It returns a list of strings defining those distances: 
    
    @atom1Name_atom1Resid-@atom2Name_atom2Resid

    Input
    -----

        topology_path      : path to the topology file.
        selection1         : first selection of atoms.
        selection2         : second selection of atoms.
        stride1            : stride for the first selection.
        stride2            : stride for the second selection.
        skip_neighbors     : skip distances between atoms in the same residue or neighboring residues.
        skip_bonded_atoms  : skip distances between bonded atoms.

    Output
    ------

        all_labels : list of labels.
    '''

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
    '''
    This function finds all real or virtual dihedrals in a selection of heavy atoms from a given topology.
    
    It returns a list of strings defining those dihedral angles:
    
    @atom1Name_atom1Resid-@atom2Name_atom2Resid-@atom3Name_atom3Resid-@atom4Name_atom4Resid

    The 'search_mode' controls the type of dihedrals to search for:

        virtual:          find all virtual dihedrals in the selection assuming the atoms in the topology are connected in order. Intended for coarse-grained models (e.g. C-alpha atoms).
        protein_backbone: find all backbone dihedrals (psi, phi) in the selection assuming the system is a protein with standard residues. Intended for all-atom protein models.
        real:             find all real dihedrals between heavy atoms in the selection. Each dihedral will be defined by a set of 4 bonded atoms. Intended for all-atom models.

    Input
    -----

        topology_path  : path to the topology file.
        selection      : selection of atoms.
        search_mode    : mode to search for dihedrals. Options: (virtual, protein_backbone, real)
    
    Output
    ------

        dihedral_labels (list): list of dihedral labels.
    '''
    
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
    '''
    This function finds all the coordinates of atoms in a selection in the topology.
    
    It returns a list of strings defining those coordinates: 
    
    @atomName_atomResid

    Input
    -----

        topology_path : path to the topology file.
        selection     : selection of atoms.
        stride        : stride for the selection.

    Output
    ------

        all_labels : list of labels.
    '''

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
    '''
    Takes as input a path to a topology file and a selection and returns all the virtual dihedrals 
    between heavy atoms in that selection.
    
    The function assumes that the atoms in the topology are connected in order.

    It returns a list of strings defining those dihedral angles:
    
    @atom1Name_atom1Resid-@atom2Name_atom2Resid-@atom3Name_atom3Resid-@atom4Name_atom4Resid

    Input
    -----

        topology_path: path to the topology file.
        selection    : MDAnalysis selection of atoms.

    Output
    ------

        dihedral_labels (list): list of dihedral labels.
    '''

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
    '''
    Takes as input a topology path and a selection and returns all the backbone dihedrals in the selection.

    This will only make sense if the selection is part of a protein.

    It returns a list of strings defining those dihedral angles:

    @phi_resid
    
    @psi_resid
    
    Note that the phi angle of a given residue requires the existence of the previous residue,
    and the psi angle of a given residue requires the existence of the next residue.

    Input
    -----

        topology_path  : path to the topology file.
        selection      : selection of atoms.

    Output
    ------

        dihedral_labels: list of dihedral labels.
    '''

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
    '''
    Takes as input a topology path and a selection and returns all the real dihedral angles between heavy atoms in the selection.

    It returns a list of strings defining those dihedral angles: 
    
    @atom1Name_atom1Resid-@atom2Name_atom2Resid-@atom3Name_atom3Resid-@atom4Name_atom4Resid

    To find the dihedrals, the function looks for all sets of 4 bonded atoms in the list of atoms.
    
    If the topology contains bonds, it uses the bonds to find the dihedrals. 
    
    If the topology does not contain bonds, it uses a distance criterion to guess the bonds.

    Input
    -----

        topology_path  : path to the topology file.
        selection      : selection of atoms.

    Output
    ------

        dihedral_labels (list): list of dihedral labels.
    '''
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
    '''
    This function finds the rotatable dihedrals involving heavy atoms in a selection of a PDB structure
    and returns a list with the labels.

    Inputs
    ------

        topology_path        : path to the topology file.
        dihedrals_definition : dictionary containing the definition of the group of dihedrals.

    Output
    ------

        dihedral_labels (list): list of dihedral labels.
    '''

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
    '''
    This function returns the list of distance labels for a group of distances defined in a dictionary.

    Input
    -----

        topology_path        : path to the topology file.
        distances_definition : dictionary containing the definition of the group of distances.
    
    Output
    ------

        distance_labels (list): list of distance labels.
    '''

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
    '''
    This function returns the list of coordinate labels for a group of coordinates defined in a dictionary.

    Input
    -----

        topology_path       : path to the topology file.
        coordinate_definition : dictionary containing the definition of the group of coordinates.
    
    Output
    ------

        coordinate_labels (list): list of coordinate labels.
    '''
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
    Get a list of feature labels from a features_configuration dictionary and a topology file.

    The labels can be used both as a name for the feature and as a definition of the feature. I.e. there is a
    unique label for each feature.
    
    Each PLUMED feature should have a unique label, such that there is a bijective mapping between the
    PLUMED command computing the feature and the label of the feature.
    
    Input
    -----
        features_configuration (dict): dictionary containing the features to extract organized by groups.
        topology_path          (str): path to the topology file.
    
    Output
    ------
        features_labels (list): list containing the feature labels.
    """

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
    
    return features_labels
    
# Working with trajectories
def extract_XTC(trajectory_path: str, topology_path: str, traj_frames: list, new_traj_path: str):
    """ 
    Extract frames from a trajectory and save them in a new trajectory file in XTC format. 
    By default the frames will be ordered such that earlier frames come first.

    Input
    -----

        trajectory_path (str): path to the original trajectory file.
        topology_path   (str): path to the original topology file.
        traj_frames    (list): list of frames to extract from the trajectory.
        new_traj_path   (str): path to the new trajectory file.
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
    Extract PDB structure from a trajectory and save it in a new PDB file. Erasing CONECT records if present.

    Input
    -----

        trajectory_path (str): path to the original trajectory file.
        topology_path   (str): path to the original topology file.
        pdb_frame       (int): frame number to extract from the trajectory.
        pdb_path        (str): path to the new PDB file.
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
    Function that returns the number of frames in a trajectory.
    
    Input
    -----
        trajectory_path (str): path to the trajectory file.
        topology_path   (str): path to the topology file.
    
    Output
    ------
        num_frames (int): number of frames in the trajectory.
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
    Function that returns the number of atoms in the selection. 
    If no selection is given, it returns the total number of atoms in the topology.

    Input
    -----
        topology (str): path to the topology file.
        selection (str): selection of atoms.
    
    Output
    ------
        num_atoms (int): number of atoms in the selection.
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
    '''
    Function that returns the indices of the atoms in the selection. The indices 
    returned are starting at 1 (as in plumed or in most MD engines) and are of type int.

    MDAnalysis indices start at 0, hence the +1.

    Input
    -----
        topology (str):  path to the topology file.
        selection (str): selection of atoms.

    Output
    ------
        indices (list): list of indices of atoms in the selection.
    '''

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


# I/O functions
def find_supported_traj(parent_path, filename = None):
    """
    Find all trajectories that comply with supported formats by MDAnalysis inside a parent folder.

    Input
    -----
        parent_path (str): path to parent folder with trajectories.
        filename    (str): (Optional) generic name of trajectories. To select a subset of trajectories.

    Output
    ------
        all_supported_trajs (list): sorted list of paths to supported trajectories.
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
    Find all topologies that comply with supported formats by MDAnalysis inside a parent folder.

    Input
    -----
        parent_path (str): path to parent folder.
        filename (str): generic name of topologies.
    
    Output
    ------
        all_supported_tops (list): sorted list of paths to supported topologies.
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
    Creates a PDB file from a structure file using MDAnalysis.
    
    Input
    -----
        structure_path (str): path to the structure file.
        file_name (str): name of the PDB file.
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
    Create a PLUMED input file to calculate the RMSD of with respect to a reference structure.

    Input
    -----
        topology_path  (str): path to the original topology file.
        output_path    (str): path to the new pdb template used by PLUMED.
        align_selection (str): selection of atoms to align to before calculating the RMSD.
        rmsd_selection  (str): selection of atoms to calculate the RMSD.
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

# Analysis
def RMSD(trajectory_path: str, topology_path: str, selection: str, fitting_selection: str) -> np.array:
    """
    Calculate the RMSD of the trajectory with respect to a reference structure.

    Input
    -----
        trajectory_path   (str): path to the trajectory file.
        topology_path     (str): path to the topology file.
        rmsd_selection    (str): selection of atoms to calculate the RMSD.
        fitting_selection (str): selection of atoms to fit the trajectory to.
    
    Output
    ------
        rmsd (np.array): array with the RMSD values for each frame.
    """

    # Load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)
        
    ref = mda.Universe(topology_path)
    
    R = MDAnalysis.analysis.rms.RMSD(u, ref, select=fitting_selection, groupselections=[selection]).run()
    
    rmsd = R.results.rmsd.T[3]
    
    return rmsd
 
def RMSF(trajectory_path: str, topology_path: str, selection: str, fitting_selection: str) -> np.array:
    """
    Calculate the RMSF of the trajectory with respect to the average structure

    Input
    -----
        trajectory_path (str): path to the trajectory file.
        topology_path   (str): path to the topology file.
        selection       (str): selection of atoms to calculate the RMSF.
    
    Output
    ------
        rmsf (np.array): array with the RMSF values for each residue
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


# Other
def to_entity_name(mda_selection: str) -> str:
    """ 
    Transform an MDanalysis selection string into an entity name string.
    
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
    Transform an entity name string into an MDAnalysis selection string.
    
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