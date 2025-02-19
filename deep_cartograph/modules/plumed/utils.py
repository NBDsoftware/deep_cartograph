# Import modules
import io 
import os
import sys
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Import local modules
from deep_cartograph.modules.md import md

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# I/O CSV Handlers
# ------------
#
# Functions to handle PLUMED CSV files
def read_as_pandas(colvars_path: str) -> pd.DataFrame:
    '''
    Function that reads a COLVARS file and returns a pandas DataFrame with the same column names as in the COLVARS file.    
    The time column in ps will be converted to ns.

    If the logger level is set to DEBUG, information about the column names will be printed.

    Inputs
    ------

        colvars_path    (str):          COLVARS file path

    Outputs
    -------

        colvars_df      (pandas DataFrame):      COLVARS data
    '''

    # Read column names
    column_names = read_column_names(colvars_path)

    # Read COLVARS file
    colvars_df = pd.read_csv(colvars_path, sep='\s+', dtype=np.float32, comment='#', header=None, names=column_names)

    # Convert time from ps to ns - working with integers to avoid rounding errors
    colvars_df["time"] = colvars_df["time"] * 1000 / 1000000

    # Show info of traj_df
    writtable_info = io.StringIO()
    colvars_df.info(buf=writtable_info)
    logger.debug(f"{writtable_info.getvalue()}")

    return colvars_df

def read_column_names(colvars_path: str) -> list:
    '''
    Reads the column names from a COLVARS file. 

    Inputs
    ------

        colvars_path    (str):          COLVARS file path

    Outputs
    -------

        column_names    (list of str):  list with the column names
    '''

    # Read first line of COLVARS file
    with open(colvars_path, 'r') as colvars_file:
        first_line = colvars_file.readline()

    # Separate first line by spaces
    first_line = first_line.split()

    # The first element is "#!" and the second is "FIELDS" - remove them
    column_names = first_line[2:]

    return column_names

def write_as_csv(dataframe, path):
    """
    Writes a pandas DataFrame to a CSV file keeping the same column names but in PLUMED format.
    Note that the time column is assumed to be in ns and will be converted to ps!

    If the file already exists, it will append the new data to the existing file.

    Inputs
    ------

        dataframe   (pandas DataFrame):    DataFrame to be written
        path                     (str):    path to the CSV file including the file name

    """

    # Convert time from ns to ps
    dataframe["time"] = dataframe["time"] * 1000

    # Check if file already exists
    if not os.path.isfile(path):

        # Create csv file with header line
        header_line = "#! FIELDS " + " ".join(dataframe.columns)
        with open(path, 'w') as csv_file:
            csv_file.write(header_line + "\n")
    
    else:
        # Erase first dataframe row and add time offset
        # Find last time in csv file
        with open(path, 'r') as csv_file:
            last_line = csv_file.readlines()[-1]
        last_time = float(last_line.split()[0])

        # Erase first row - repeated sample for same initial conditions 
        dataframe = dataframe.drop(dataframe.index[0])

        # Add time offset to dataframe
        dataframe["time"] = dataframe["time"] + last_time

    # Close csv file
    csv_file.close()

    # Append data to csv file
    dataframe.to_csv(path, mode='a', header=False, index=False, sep=' ', float_format='%.6f')

    return


# PLUMED driver
# -------------
#
# Returns the corresponding PLUMED driver shell command
def get_driver_command(plumed_input: str, traj_path: str = None, topology_path: str = None):
    '''
    Function that creates a PLUMED DRIVER Shell command. It returns the command as a string

    Example:

        "driver --plumed plumed_input --ixyz traj_path --natoms num_atoms"

    Inputs
    ------

        plumed_input     (str):              PLUMED input file path
        traj_path        (str):              path to trajectory file
        topology_path    (str):              path to topology file

    Outputs
    -------

        driver_command   (str):              PLUMED DRIVER command
    '''

    # Initialize
    driver_command = []
        
    # Add driver flag
    driver_command.append("driver")

    # Add plumed flag
    driver_command.append("--plumed")

    # Make sure plumed input is given with the absolute path
    plumed_input = os.path.abspath(plumed_input)

    # Add plumed input
    driver_command.append(plumed_input)

    # Add trajectory or noatoms flag
    if traj_path is None:
        driver_command.append("--noatoms")
    else:
        traj_flag = get_traj_flag(traj_path)
        driver_command.append(traj_flag)
        
        if Path(traj_path).suffix == ".pdb":
            traj_path = check_CRYST1_record(traj_path, Path(topology_path).parent)
            
        traj_path = os.path.abspath(traj_path)
        driver_command.append(traj_path)

    # Find the number of atoms in the system if topology is given (some traj formats do not need this)
    if topology_path is not None:
        num_atoms = md.get_number_atoms(topology_path)
        driver_command.append("--natoms")
        driver_command.append(str(num_atoms))

    # Join command
    driver_command = " ".join(driver_command)

    return driver_command 

def run_driver_command(driver_command: str, plumed_settings: dict, plumed_timeout: int) -> None:
    """
    Function that runs a PLUMED DRIVER command. It adds the necessary environment variables and modules
    before calling the plumed binary with the PLUMED DRIVER command.

    Inputs
    ------

        driver_command  (str):               PLUMED DRIVER command
        plumed_settings (dict):              settings for PLUMED (binaries, kernel, etc.)
        plumed_timeout  (int):               timeout for PLUMED in seconds
    """

    all_commands = []

    plumed_binary = "plumed"

    # If settings are given
    if plumed_settings is not None:

        # Add environment commands
        if plumed_settings.get('env_commands', []) != []:
            
            # Join environment commands with &&
            env_commands = " && ".join(plumed_settings.get('env_commands'))

            # Add environment commands to all commands
            all_commands.append(env_commands)
                
        # Set environment variable PLUMED_KERNEL
        if plumed_settings.get('kernel_path') is not None:
            os.environ['PLUMED_KERNEL'] = plumed_settings.get('kernel_path')

        # Add bin path
        if plumed_settings.get('bin_path') is not None:
            plumed_binary = plumed_settings.get('bin_path')

    # Add binary to driver command
    driver_command = plumed_binary + " " + driver_command

    # Add driver command to all commands
    all_commands.append(driver_command)

    # Join all commands with &&
    all_commands = " && ".join(all_commands)

    # Log execution information
    logger.info(f"Executing PLUMED driver command: {all_commands}")

    # Find level of logging
    if logger.isEnabledFor(logging.DEBUG):
        # Execute PLUMED redirecting output to the log file
        completed_process = subprocess.run(args=all_commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=plumed_timeout) # cwd=str(Path(input_path).parent),
        
        # Send standard output to the log file
        logger.info(completed_process.stdout.decode('utf-8'))
    else:
        # Execute PLUMED without redirecting output
        completed_process = subprocess.run(args=all_commands, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=plumed_timeout) # cwd=str(Path(input_path).parent), 

    # Check if PLUMED failed
    if completed_process.returncode != 0:
        logger.error("PLUMED failed! :( \n")
        logger.error(completed_process.stderr.decode('utf-8'))
        sys.exit()

    return


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
    atoms_format = dihedrals_definition.get('atoms_format', 'index')
    search_mode = dihedrals_definition.get('search_mode', 'real')

    atomic_definitions = md.find_dihedrals(topology_path, selection, search_mode, atoms_format)
    
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
    atoms_format = distances_definition.get('atoms_format', 'index')

    atomic_definitions = md.find_distances(topology_path, selection1, selection2, stride1, stride2, skip_neighbors, skip_bonded_atoms, atoms_format)
    
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

    # Extensions and flags supported by PLUMED
    supported_extensions = {
        ".xyz" : "--ixyz",
        ".gro" : "--igro",
        ".dlp4": "--idlp4",
        ".xtc" : "--ixtc",
        ".trr" : "--itrr",
        ".dcd" : "--mf_dcd",
        ".crd" : "--mf_crd",
        ".pdb" : "--mf_pdb",
    }

    # Get extension
    extension = Path(traj_path).suffix

    # Check if extension is supported
    if extension not in supported_extensions.keys():
        raise Exception("Extension of trajectory not supported by PLUMED.")
    else:
        traj_flag = supported_extensions[extension]

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
    
    dummy_cryst1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00"

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