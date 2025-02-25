# Import modules
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.analysis.align
from MDAnalysis.lib.distances import calc_bonds

# Set logger
logger = logging.getLogger(__name__)

# Constants for the module
covalent_bond_threshold = 2.0

# Working with structures
def get_indices(topology: str, selection: str = None) -> list:
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

def find_distances(topology_path: str, selection1: str, selection2: str, stride1: int, stride2: int, skip_neighbors: bool, skip_bonded_atoms: bool, atoms_format: str) -> List[str]:
    '''
    This function does the following:

        1. Finds two selections of atoms in the topology.
        2. Find all the pairwise distances between the atoms in the selections skipping bonded atoms or atoms pertaining to neighboring residues if requested.
        3. Returns a list with the distance labels in the format specified by the user.

    The format for the atomic_definitions will depend on atoms_format and be one of the following: 
        
        < atoms_format value > : example
        
        name:       "@atom_name-atom_resid,@atom_name-atom_resid"
        
        index:      "atom_index,atom_index"      -----------------> Note that MDAnalysis indexing starts at 0, thus we sum one to the index.

    Input
    -----

        topology_path      : path to the topology file.
        selection1         : first selection of atoms.
        selection2         : second selection of atoms.
        stride1            : stride for the first selection.
        stride2            : stride for the second selection.
        skip_neighbors     : skip distances between atoms in the same residue or neighboring residues.
        skip_bonded_atoms  : skip distances between bonded atoms.
        atoms_format       : format of the atom labels defining each distance. Options: (index, name)

    Output
    ------

        atomic_definitions : list of distance labels.
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

    atomic_definitions = []

    # Create all possible pairs of atoms without repetition
    for first_atom in first_atoms:

        for second_atom in second_atoms:
            
            # Check both atoms are different
            if first_atom.index != second_atom.index:

                # Create atomic label for this distance
                if atoms_format == "index":
                    atom_label = f"{first_atom.index + 1},{second_atom.index + 1}"
                    equivalent_atom_label = f"{second_atom.index + 1},{first_atom.index + 1}"
                elif atoms_format == "name":
                    atom_label = f"@{first_atom.name}-{first_atom.resid},@{second_atom.name}-{second_atom.resid}"
                    equivalent_atom_label = f"@{second_atom.name}-{second_atom.resid},@{first_atom.name}-{first_atom.resid}"
                else:
                    raise ValueError(f"atoms_format {atoms_format} not supported.")

                # Check distance is not repeated
                if atom_label not in atomic_definitions and equivalent_atom_label not in atomic_definitions:
                    
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
                    atomic_definitions.append(atom_label)
                    
    return atomic_definitions

def find_dihedrals(topology_path: str, selection: str, search_mode: str, atoms_format: str) -> List[str]:
    '''
    This function does the following:

        1. Finds a selection of atoms in the topology.
        2. Keeps just the heavy atoms in the selection.
        4. Finds all real or virtual dihedrals in the selection. See search_mode.
        5. Returns a list with the dihedral labels in the format specified by the user.
    
    The format for the atomic_definitions will depend on atoms_format and be one of the following: 
        
        < atoms_format value > : example
        
        atom_name:  "@atom_name-atom_resid,@atom_name-atom_resid,@atom_name-atom_resid,@atom_name-atom_resid"
        
        index:      "atom_index,atom_index,atom_index,atom_index" -----------------> Note that MDAnalysis indexing starts at 0, thus we sum one to the index.

    search_mode can be one of the following:

        virtual:          find all virtual dihedrals in the selection assuming the atoms in the topology are connected and in order. Intended for coarse-grained models (e.g. C-alpha atoms).
        protein_backbone: find all backbone dihedrals in the selection assuming the system is a protein with standard residues. Intended for all-atom protein models.
        real:             find all real dihedrals in the selection. Each dihedral will be defined by a set of 4 bonded atoms. Intended for all-atom models.

    Input
    -----

        topology_path  : path to the topology file.
        selection      : selection of atoms.
        search_mode    : mode to search for dihedrals. Options: (virtual, protein_backbone, real)
        atoms_format   : format of the atom labels defining each dihedral. Options: (atom_name, protein or index)
    
    Output
    ------

        atomic_definitions (list): list of dihedral labels.
    '''
    
    if search_mode == "virtual":
    
        # Find virtual dihedrals
        dihedrals = get_virtual_dihedral_labels(topology_path, selection, atoms_format)

    elif search_mode == "protein_backbone":

        # Find protein backbone dihedrals
        dihedrals = get_protein_back_dihedrals(topology_path, selection)
    
    elif search_mode == "real":

        # Find real dihedrals
        dihedrals = get_all_real_dihedrals(topology_path, selection, atoms_format)

    else:

        raise ValueError(f"search_mode {search_mode} not supported. Options: (virtual, protein_backbone, real)")

    return dihedrals

def get_virtual_dihedral_labels(topology_path: str, selection: str, atoms_format: str) -> List[str]:
    '''
    Takes as input a path to a topology file and a selection and returns the virtual backbone dihedral plumed labels for all the backbone dihedrals in the selection.

    The format for the labels defining each dihedral will be one of the following:

    < atoms_format > : < example >
            
            name             : "@atom_name-atom_resid,@atom_name-atom_resid,@atom_name-atom_resid,@atom_name-atom_resid"
            
            index            : "atom_index,atom_index,atom_index,atom_index" -----------------> Note that MDAnalysis indexing starts at 0, thus we sum one to the index.

    Input
    -----

        atoms            : list of atoms as an AtomGroup.
        atoms_format: format of the atom labels defining each dihedral. Options: (name or index)

    Output
    ------

        atomic_definitions (list): list of dihedral labels.
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
    
    atomic_definitions = []

    # Iterate over all CA atoms skipping the first 3
    for i in range(3, len(heavy_atoms)):

        # Create atom label
        if atoms_format == "index":
            dihedral_label = f"{atoms[i-3].index + 1},{atoms[i-2].index + 1},{atoms[i-1].index + 1},{atoms[i].index + 1}"
        elif atoms_format == "name":
            dihedral_label = f"@{atoms[i-3].name}-{atoms[i-3].resid},@{atoms[i-2].name}-{atoms[i-2].resid},@{atoms[i-1].name}-{atoms[i-1].resid},@{atoms[i].name}-{atoms[i].resid}"
        else:
            raise ValueError(f"atoms_format {atoms_format} not supported. Options for virtual dihedrals: (name, index)")

        # Add atomic definition of the dihedral
        atomic_definitions.append(dihedral_label)

    return atomic_definitions

def get_protein_back_dihedrals(topology_path: str, selection: str) -> List[str]:
    '''
    Takes as input a topology path and a selection and returns all the backbone dihedrals in the selection.

    This will only make sense if the selection is part of a protein.

    The format for the backbone dihedral labels will be the following:

        "@phi-resid,@psi-resid,@omega-resid"

    Input
    -----

        topology_path  : path to the topology file.
        selection      : selection of atoms.

    Output
    ------

        atomic_definitions: list of dihedral labels.
    '''

    # Load topology
    u = mda.Universe(topology_path)

    # Select atoms
    atoms = u.select_atoms(selection)

    dihedrals = ['phi', 'psi', 'omega']

    atomic_definitions = []

    # Find all residues in the list of atoms
    residues = np.unique([atom.resid for atom in atoms])
    
    # Iterate over all residues
    for residue in residues:

        # Iterate over all backbone dihedrals
        for dihedral in dihedrals:

            # Create dihedral label
            dihedral_label = f"@{dihedral}-{residue}"

            # Add dihedral definition
            atomic_definitions.append(dihedral_label)
    
    return atomic_definitions

def get_all_real_dihedrals(topology_path: str, selection: str, atoms_format: str) -> List[str]:
    '''
    Takes as input a path to a topology and a selection and returns the real dihedral plumed labels for all the dihedrals in the selection.

    The format for the labels defining each dihedral will be one of the following: 
        
        < atoms_format > : < example >
        
        name             : "@atom_name-atom_resid,@atom_name-atom_resid,@atom_name-atom_resid,@atom_name-atom_resid"
        
        index            : "atom_index,atom_index,atom_index,atom_index" -----------------> Note that MDAnalysis indexing starts at 0, thus we sum one to the index.

    To find the dihedrals, the function looks for all sets of 4 bonded atoms in the list of atoms.
    If the topology contains bonds, it uses the bonds to find the dihedrals. If the topology does not contain bonds, it uses a distance criterion to guess the bonds.

    Input
    -----

        topology_path  : path to the topology file.
        selection      : selection of atoms.
        atoms_format   : format of the atom labels defining each dihedral. Options: (name or index)

    Output
    ------

        atomic_definitions (list): list of dihedral labels.
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

    atomic_definitions = []

    # Iterate over all bonds 
    for bond in all_bonds:
        
        # Get indices of the bonded atoms
        i_index, j_index = bond.indices

        # Check if both atoms are heavy atoms
        if i_index in heavy_atom_indices and j_index in heavy_atom_indices:

            # Debug:
            logger.debug(f"Bond atom index i: {i_index}, atom index j: {j_index}")

            # Get neighbors of each atom
            i_neighbors = neighbors_dict[i_index]
            j_neighbors = neighbors_dict[j_index]

            # Debug:
            logger.debug(f"Neighbors of atom i: {i_neighbors}")
            logger.debug(f"Neighbors of atom j: {j_neighbors}")

            # For each possible set of 4 bonded atoms around the bond (i, j)
            for i_neighbor in i_neighbors:
                if i_neighbor != j_index:
                    for j_neighbor in j_neighbors:
                        if j_neighbor != i_index and j_neighbor != i_neighbor:

                            # Create dihedral label for atoms with indices i_neighbor, i_index, j_index, j_neighbor
                            if atoms_format == "index":
                                dihedral_label = f"{i_neighbor},{i_index},{j_index},{j_neighbor}"
                                equivalent_dihedral_label = f"{j_neighbor},{j_index},{i_index},{i_neighbor}"
                            elif atoms_format == "name":
                                dihedral_label = f"@{heavy_atom_dict[i_neighbor].name}-{heavy_atom_dict[i_neighbor].resid},@{heavy_atom_dict[i_index].name}-{heavy_atom_dict[i_index].resid},@{heavy_atom_dict[j_index].name}-{heavy_atom_dict[j_index].resid},@{heavy_atom_dict[j_neighbor].name}-{heavy_atom_dict[j_neighbor].resid}"
                                equivalent_dihedral_label = f"@{heavy_atom_dict[j_neighbor].name}-{heavy_atom_dict[j_neighbor].resid},@{heavy_atom_dict[j_index].name}-{heavy_atom_dict[j_index].resid},@{heavy_atom_dict[i_index].name}-{heavy_atom_dict[i_index].resid},@{heavy_atom_dict[i_neighbor].name}-{heavy_atom_dict[i_neighbor].resid}"
                            else:
                                raise ValueError(f"atoms_format {atoms_format} not supported. Options for real dihedrals: (name, index)")
                            
                            # Check dihedral is not repeated
                            if dihedral_label not in atomic_definitions and equivalent_dihedral_label not in atomic_definitions:
                                atomic_definitions.append(dihedral_label)

    return atomic_definitions    

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

# Working with trajectories
def extract_frames(trajectory_path: str, topology_path: str, traj_frames: list, 
                   new_traj_path: str, top_frame: int, new_top_path: str):
    """ 
    Extract frames from a trajectory and save them in a new trajectory file. By default the frames will be ordered
    such that earlier frames come first.

    Input
    -----

        trajectory_path (str): path to the original trajectory file.
        topology_path   (str): path to the original topology file.
        traj_frames    (list): list of frames to extract from the trajectory.
        new_traj_path   (str): path to the new trajectory file.
        top_frames     (list): list of frames to extract from the topology.
        new_top_path    (str): path to the new topology file.
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
            
    # Load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)
    
    # Save new topology to a temporary PDB file including CONECT records
    tmp_topology_path = os.path.join(Path(new_top_path).parent, "tmp.pdb")
    
    # Write temporary PDB topology
    with mda.Writer(tmp_topology_path, n_atoms=u.atoms.n_atoms, format='PDB') as writer:
        u.trajectory[top_frame]
        writer.write(u)

    # Remove CONECT records from the temporary topology
    with open(tmp_topology_path, 'r') as f:
        lines = f.readlines()

    with open(new_top_path, 'w') as f:
        for line in lines:
            if not line.startswith("CONECT"):
                f.write(line)

    return

def extract_clusters_from_traj(trajectory_path: str, topology_path: str, traj_df: pd.DataFrame, samples_per_frame: float = 1, centroids_df: pd.DataFrame = None,
                               cluster_label: str = 'cluster', frame_label: str = 'frame', output_folder: str = 'clustered_traj'):
    """
    Extract all frames from the trajectory pertaining to each cluster and save them in new trajectory files (XTC).

    This function assumes that the traj_df contains a row for each frame in the trajectory and a column with the cluster label.

    Input
    -----
        trajectory_path     (str): path to the trajectory file with the frames to cluster.
        topology_path       (str): path to the topology file of the trajectory.
        traj_df       (DataFrame): DataFrame containing the cluster and frame labels for each frame.
        samples_per_frame (float): number of samples per frame in the trajectory.
        centroids_df  (DataFrame): DataFrame containing the centroids of each cluster.
        cluster_label       (str): name of the column containing the cluster label.
        frame_label         (str): name of the column containing the frame index.
        output_folder       (str): path to the output folder.
    """
    
    # Check the existence of trajectory and topology files
    if not os.path.exists(trajectory_path):
        logger.warning(f"Trajectory file {trajectory_path} not found.")
        return

    if not os.path.exists(topology_path):
        logger.warning(f"Topology file {topology_path} not found.")
        return

    # Check the number of samples from the trajectory
    num_frames = get_num_frames(trajectory_path, topology_path)
    traj_samples = int(num_frames*samples_per_frame)
    
    # Check the number of samples from the colvars file
    colvars_samples = len(traj_df)
    
    # Check the match
    if traj_samples != colvars_samples:
        logger.warning(f"Number of samples in the colvars file: {colvars_samples} does not match the number of samples in the trajectory: {traj_samples} (num_frames x num_samples_per_frame).") 
        logger.warning(f"Review the traj file, the colvars file and the num_samples_per_frame setting.")
        return
    
    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find the different clusters from the trajectory
    clusters = np.unique(traj_df[cluster_label])
    
    # Print the number of clusters
    logger.info(f"Number of clusters: {len(clusters)}")
    
    # Print the number of centroids
    if centroids_df is not None:
        logger.info(f"Number of centroids: {len(centroids_df)}")

    # Extract frames for each cluster
    for cluster in clusters:

        # Skip if cluster is -1 (noise cluster)
        if cluster == -1:
            continue

        # Find frames for this cluster
        cluster_samples = traj_df[traj_df[cluster_label] == cluster][frame_label]
        cluster_samples = list(cluster_samples)

        # Find topology frame
        if centroids_df is not None:
            # Make it the centroid frame if available
            topology_samples_df = centroids_df[centroids_df[cluster_label] == cluster][frame_label]
            if len(topology_samples_df) > 0:
                topology_samples = topology_samples_df.values[0]
            else:
                # Pick the first frame from the cluster if centroids are not available
                topology_samples = cluster_samples[0]
        else:
            # Pick the first frame from the cluster if centroids are not available
            topology_samples = cluster_samples[0]

        # Create file name
        cluster_traj_name = f"cluster_{cluster}.xtc"
        cluster_top_name = f"cluster_{cluster}.pdb"

        # Create paths
        cluster_traj_path = os.path.join(output_folder, cluster_traj_name)
        cluster_top_path = os.path.join(output_folder, cluster_top_name)
        
        # Adjust sample number to frame number
        cluster_frames = [int(sample * samples_per_frame) for sample in cluster_samples]
        topology_frame = int(topology_samples * samples_per_frame)

        # Extract frames
        extract_frames(trajectory_path, topology_path, cluster_frames, cluster_traj_path, topology_frame, cluster_top_path)

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