# Import modules

import os
import logging
from pathlib import Path

# Import local modules
from deep_cartograph.modules.md import md

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# Get labels for features/cvs
# ---------------------------
#
# Return lists with feature names and the atom-wise definition to be used in a PLUMED input file
def get_dihedral_labels(topology_path: str, dihedrals_definition: dict):
    '''
    This function does the following:
    
        1. Finds rotatable dihedrals involving heavy atoms in a selection of a PDB structure.
        2. Returns two lists with the names and atoms of each dihedral (dihedral_names and atomic_definitions respectively)

    Inputs
    ------

        topology_path        : path to the topology file.
        dihedrals_definition : dictionary containing the definition of the group of dihedrals.

    Output
    ------

        dihedral_names        (list): list of command labels for each dihedral.
        atomic_definitions    (list): list of atom labels for each dihedral.
    '''

    # Read dihedral group definition
    selection = dihedrals_definition.get('selection', 'all')
    search_mode = dihedrals_definition.get('search_mode', 'real')

    atomic_definitions = md.find_dihedrals(topology_path, selection, search_mode)
    
    # Define command labels
    dihedral_names = []
    replace_chars = {',': '-', ' ': '', "-": "_"}
    for label in atomic_definitions:
        for key, value in replace_chars.items():
            label = label.replace(key, value)
        dihedral_names.append(label)

    return dihedral_names, atomic_definitions

def get_distance_labels(topology_path: str, distances_definition: dict):
    '''
    This function does the following:
    
        1. Finds pairs of atoms in a selection of a PDB structure
        2. Returns two lists with the names and atoms of each distance (distance_names and atomic_definitions respectively)


    Input
    -----

        topology_path        : path to the topology file.
        distances_definition : dictionary containing the definition of the group of distances.
    
    Output
    ------

        distance_names        (list): list of command labels for each distance.
        atomic_definitions    (list): list of atom labels for each distance.
    '''

    # Read distance group definition
    selection1 = distances_definition.get('first_selection', 'all')
    selection2 = distances_definition.get('second_selection', 'all')
    stride1 = distances_definition.get('first_stride', 1)
    stride2 = distances_definition.get('second_stride', 1)
    skip_neighbors = distances_definition.get('skip_neigh_residues', False)
    skip_bonded_atoms = distances_definition.get('skip_bonded_atoms', False)

    atomic_definitions = md.find_distances(topology_path, selection1, selection2, stride1, stride2, skip_neighbors, skip_bonded_atoms)
    
    # Define command labels
    distance_names = []
    replace_chars = {',': '-', ' ': '', "-": "_"}
    for label in atomic_definitions:
        for key, value in replace_chars.items():
            label = label.replace(key, value)
        distance_names.append(f"dist_{label}")

    return distance_names, atomic_definitions

# OTHER
def get_traj_flag(traj_path):
    """
    Get trajectory flag from trajectory path. Depending on the extension of the trajectory,
    the flag will be different.
    """ 
    
    # Extensions supported by the molfile plugin
    molfile_extensions = {
        ".dcd" : "--mf_dcd",
        ".crd" : "--mf_crd",
        ".pdb" : "--mf_pdb",
        ".crdbox" : "--mf_crdbox",
        ".gro" : "--mf_gro",
        ".g96" : "--mf_g96",
        ".trr" : "--mf_trr",
        ".trj" : "--mf_trj",
        ".xtc" : "--mf_xtc"
    }
    
    # Extensions supported by the xdrfile plugin
    xdrfile_extensions = {
        ".xtc" : "--ixtc",
        ".trr" : "--itrr"
    }

    # Extensions and flags supported by PLUMED
    other_extensions = {
        ".xyz" : "--ixyz",
        ".gro" : "--igro",
        ".dlp4": "--idlp4"
    }
    # Get extension
    extension = Path(traj_path).suffix
        
    # Get flag
    traj_flag = molfile_extensions.get(extension)
    if traj_flag is None:
        traj_flag = xdrfile_extensions.get(extension)
        if traj_flag is None:
            traj_flag = other_extensions.get(extension)
    
    if traj_flag is None:
        raise Exception("Extension of trajectory not supported by PLUMED.")

    return traj_flag

def check_CRYST1_record(pdb_path, output_folder) -> str:
    """
    Check if a PDB file has a meaningless CRYST1 record and remove it if so.
    
    PDB bank requires the CRYST1 record to be present, so some tools will write a dummy CRYST1 record (like MDAnalysis)
    
    Dummy CRYST1 record: 
    
        CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1
        
    PLUMED will use this record to obtain the box dimensions and correct for periodic boundary conditions when computing
    variables. So any present CRYST1 record must be meaningful.
    
    Parameters
    ----------
    
        pdb_path    (str):  path to the PDB file
        output_folder (str): path to the output folder where the new PDB file will be written if needed
    
    Returns
    -------
    
        pdb_path    (str):  path to the PDB file with the CRYST1 record removed if needed
    """
    
    dummy_cryst1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00" # NOTE: Maybe check for the values of the box dimensions and angles rather than the specific string

    # Read PDB file
    with open(pdb_path, 'r') as pdb_file:
        pdb_lines = pdb_file.readlines()

    # Check if CRYST1 record is present
    dummy_record = None
    for line in pdb_lines:
        if line.startswith(dummy_cryst1):
            dummy_record = line
            break
        
    # If dummy record is present, remove it
    if dummy_record is not None:
        
        pdb_lines.remove(dummy_record)
        new_pdb_path = os.path.join(output_folder, Path(pdb_path).name)
        
        # Write new PDB file
        with open(new_pdb_path, 'w') as pdb_file:
            pdb_file.writelines(pdb_lines)
        
        logger.warning(f"Dummy CRYST1 record removed from {pdb_path}")
    else:
        new_pdb_path = pdb_path

    return new_pdb_path