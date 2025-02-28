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

        
def to_atomgroup(entity_name: str) -> str:
    """ 
    Convert entity name to atom group: @CA_1 -> @CA-1
    
    Replaces "_" by "-".
    """        
    return entity_name.replace("_", "-")
    

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