"""
Build and run PLUMED command-line tools (such as ``plumed driver``) from Python.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Union, Optional

import logging

from deep_cartograph.modules.md import md
from deep_cartograph.modules.plumed.utils import get_traj_flag, sanitize_CRYST1_record

# Set logger
logger = logging.getLogger(__name__)

# PLUMED driver
# -------------
#
# Returns the corresponding PLUMED driver shell command
def get_driver_command(
    plumed_input: str, 
    traj_path: Optional[str] = None, 
    num_atoms: Optional[int] = None, 
    output_path: Optional[str] = None
    ) -> str:
    """
    Build the arguments for the PLUMED ``driver`` tool, which runs a PLUMED input file on a trajectory.

    Example: ``"driver --plumed /abs/plumed.dat --mf_xtc /abs/traj.xtc --natoms 1000"``

    Parameters
    ----------
    plumed_input : str
        Path to the PLUMED input file.
    traj_path : str, optional
        Path to the trajectory. If None, ``--noatoms`` is used and PLUMED only reads the
        colvars files named in the input file.
    num_atoms : int, optional
        Number of atoms in the system. Some trajectory formats need it.
    output_path : str, optional
        Folder where a cleaned copy of a PDB trajectory is written if its CRYST1 record is a dummy one.

    Returns
    -------
    driver_command : str
        Command without the PLUMED binary (see ``run_plumed``).
    """

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

    # Add trajectory or --noatoms flag
    if traj_path:
        # Add trajectory
        traj_flag = get_traj_flag(traj_path)
        driver_command.append(traj_flag)
        if Path(traj_path).suffix == ".pdb":
            # If the trajectory has a dummy CRYST1 record, we need to remove it
            traj_path = sanitize_CRYST1_record(traj_path, output_path)
        traj_path = os.path.abspath(traj_path)
        driver_command.append(traj_path)
    else:
        # Add --noatoms flag. Don't read in a trajectory. Just use colvar files as specified in the input file
        driver_command.append("--noatoms")

    # Find the number of atoms if topology is given (some traj formats do not need this)
    if num_atoms:
        driver_command.append("--natoms")
        driver_command.append(str(num_atoms))

    # Join command
    driver_command = " ".join(driver_command)

    return driver_command 

def run_plumed(
    plumed_command: str, 
    working_dir: Optional[str] = None, 
    plumed_settings: Optional[Dict] = {}, 
    plumed_timeout: Optional[int] = 604800
    ) -> None:
    """
    Run a PLUMED command-line tool in a shell.

    The PLUMED binary, extra environment commands (e.g. ``module load``) and the PLUMED kernel
    are taken from ``plumed_settings``. The program exits if PLUMED returns an error.

    Parameters
    ----------
    plumed_command : str
        PLUMED command to run, without the binary (e.g. the output of ``get_driver_command``).
    working_dir : str, optional
        Folder where the command is run. The original folder is restored afterwards.
        Default is None (current folder).
    plumed_settings : dict, optional
        Settings for PLUMED. Keys used: ``bin_path`` (default ``'plumed'``), ``env_commands``
        (list of shell commands run first) and ``kernel_path`` (sets ``PLUMED_KERNEL``).
    plumed_timeout : int, optional
        Timeout in seconds. Default is 604800 (one week).

    Returns
    -------
    stdout : str or None
        Standard output of PLUMED, or None if it timed out or failed to start.
    stderr : str
        Standard error of PLUMED, or a short error message.
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

    # Store the original working directory
    original_cwd = os.getcwd()
  
    try:
        # Change working directory if specified
        if working_dir:
            logger.info(f"Changing working directory to: {working_dir}")
            os.chdir(working_dir)

        # Execute PLUMED redirecting output
        completed_process = subprocess.run(
            args=command_str, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=plumed_timeout, 
            text=True
        )

        stdout, stderr = completed_process.stdout, completed_process.stderr

        if logger.isEnabledFor(logging.DEBUG):
            logger.info(stdout)

        if completed_process.returncode != 0:
            logger.error("PLUMED execution failed!")
            logger.error(stderr)
            sys.exit(1)

        return stdout, stderr

    except subprocess.TimeoutExpired:
        logger.error("PLUMED execution timed out!")
        return None, "TimeoutExpired"

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, str(e)

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {original_cwd}")

