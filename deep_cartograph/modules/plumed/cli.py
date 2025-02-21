import os
import sys
import subprocess
from pathlib import Path

import logging

from deep_cartograph.modules.md import md
from deep_cartograph.modules.plumed.utils import get_traj_flag, check_CRYST1_record

# Set logger
logger = logging.getLogger(__name__)

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

    # Find the number of atoms if topology is given (some traj formats do not need this)
    if topology_path is not None:
        num_atoms = md.get_number_atoms(topology_path)
        driver_command.append("--natoms")
        driver_command.append(str(num_atoms))

    # Join command
    driver_command = " ".join(driver_command)

    return driver_command 

def run_plumed(plumed_command: str, plumed_settings: dict, plumed_timeout: int) -> None:
    """
    Runs PLUMED through command line, setting up the necessary environment variables and modules.

    Inputs
    ------

        plumed_command  (str):               PLUMED command to execute (See Command Line Tools in PLUMED manual)
        plumed_settings (dict):              (Optional) Settings for PLUMED (binaries, kernel, etc.)
        plumed_timeout  (int):               (Optional) timeout for PLUMED in seconds
    
    Returns
    -------
    
        tuple: (stdout, stderr) from the PLUMED execution
    """

    all_commands = []
    plumed_binary = plumed_settings.get('bin_path', 'plumed') if plumed_settings else 'plumed'
    
    if plumed_settings:
        if plumed_settings.get('env_commands'):
            all_commands.append(" && ".join(plumed_settings.get('env_commands')))
        
        if plumed_settings.get('kernel_path'):
            os.environ['PLUMED_KERNEL'] = plumed_settings.get('kernel_path')
    
    all_commands.append(f"{plumed_binary} {plumed_command}")
    command_str = " && ".join(all_commands)
    
    logger.info(f"Executing PLUMED command: {command_str}")

    try:
        # Execute PLUMED redirecting output to the log file
        completed_process = subprocess.run(args=command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=plumed_timeout, text=True) 
        
        stdout, stderr = completed_process.stdout, completed_process.stderr
        
        if logger.isEnabledFor(logging.DEBUG):
            # Send standard output to the log file 
            logger.info(stdout)
            
        # Check if PLUMED failed
        if completed_process.returncode != 0:
            logger.error("PLUMED execution failed! \n")
            logger.error(stderr)
            sys.exit(1)
            
        return stdout, stderr
    
    except subprocess.TimeoutExpired:
        logger.error("PLUMED execution timed out!")
        return None, "TimeoutExpired"
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, str(e)

