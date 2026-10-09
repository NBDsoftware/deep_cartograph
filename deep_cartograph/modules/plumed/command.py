"""
Functions that build single PLUMED commands (actions) as strings.

Each function returns one command, ending with a newline, ready to be added to a PLUMED input file.
"""
# Import modules
import sys
import math
import logging
import numpy as np

from typing import List, Union, Optional

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# PLUMED Commands
# ---------------
#
# Functions to create PLUMED commands
def molinfo(
    topology: str, 
    moltype: str = None
    ) -> str:
    """
    Create a PLUMED MOLINFO command.

    MOLINFO reads a structure file so that atoms can be referred to by name (e.g. ``@CA-12``) in other commands.

    Parameters
    ----------
    topology : str
        Path to the structure file (usually a PDB).
    moltype : str, optional
        Molecule type (MOLTYPE keyword). Default is None (not added).

    Returns
    -------
    command : str
        PLUMED MOLINFO command.
    """
    command = f"MOLINFO STRUCTURE={topology}"

    if moltype is not None:
        command += f" MOLTYPE={moltype}"
    
    command += "\n"

    return command

def wholemolecules(indices: List[int]) -> str:
    """
    Create a PLUMED WHOLEMOLECULES command.

    It rebuilds molecules broken by periodic boundary conditions. All atoms from the first
    to the last index are treated as a single entity.

    NOTE: If needed, this could be extended to consider multiple entities. (e.g. WHOLEMOLECULES ENTITY0=1-10 ENTITY1=11-20)

    Parameters
    ----------
    indices : list of int
        Atom indices (starting at 1). Only the first and last are used.

    Returns
    -------
    command : str
        PLUMED WHOLEMOLECULES command.
    """
    command = f"WHOLEMOLECULES ENTITY0={indices[0]}-{indices[-1]} \n"

    return command

def fit_to_template(
    template_path: str
    ) -> str:
    """
    Create a PLUMED FIT_TO_TEMPLATE command.

    It aligns the system to a reference structure at every step, which is needed for
    features that depend on orientation, such as atomic coordinates.

    Parameters
    ----------
    template_path : str
        Path to the reference PDB. Its occupancy column marks the atoms used for the alignment.

    Returns
    -------
    command : str
        PLUMED FIT_TO_TEMPLATE command.
    """
    
    command = f"FIT_TO_TEMPLATE STRIDE=1 REFERENCE={template_path} TYPE=OPTIMAL\n"

    return command

def position (
    command_label: str,
    atom: str
    ) -> str:
    """
    Create a PLUMED POSITION command, which gives the x, y and z coordinates of an atom.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atom : str
        Atom definition (index or MOLINFO shortcut such as ``@CA-12``).

    Returns
    -------
    position_command : str
        PLUMED POSITION command.
    """
    position_command = command_label + ": POSITION ATOM=" + str(atom) + " NOPBC\n"
    
    return position_command

def distance(
    command_label: str, 
    atoms: Union[List[str], str]
    ) -> str:
    """
    Create a PLUMED DISTANCE command between two atoms (or centers).

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms, as a list of atom definitions or a single comma-separated string.

    Returns
    -------
    distance_command : str
        PLUMED DISTANCE command.
    """
  
    # Check if atoms is a list of strings or a string
    if isinstance(atoms, list):
        # Convert all atoms to strings
        atoms = [str(atom) for atom in atoms]
        # Create DISTANCE command
        distance_command = command_label + ": DISTANCE ATOMS=" + ",".join(atoms)

    elif isinstance(atoms, str):
        # Convert atoms to string
        atoms = str(atoms)
        # Create DISTANCE command
        distance_command = command_label + ": DISTANCE ATOMS=" + atoms

    else:
        logger.error("Atoms must be a list of strings or a string.")
        sys.exit()

    # Add newline
    distance_command += " NOPBC\n"

    return distance_command

def custom(
    command_label: str, 
    expression: str, 
    arguments: List[str],
    periodic: bool = False
    ) -> str:
    """
    Create a PLUMED CUSTOM command, which applies a math expression to other variables.

    Parameters
    ----------
    command_label : str
        Label of the command.
    expression : str
        Math expression (e.g. ``"sin(x)"``).
    arguments : list of str
        Labels of the input variables.
    periodic : bool, optional
        Whether the result is periodic. Default is False.

    Returns
    -------
    custom_command : str
        PLUMED CUSTOM command.
    """
    
    # Create CUSTOM command
    custom_command = command_label + ": CUSTOM ARG=" + ",".join(arguments)

    # Add expression
    custom_command += " FUNC=" + expression
    
    if periodic:
        custom_command += " PERIODIC=YES"
    else:
        custom_command += " PERIODIC=NO"

    # Add newline
    custom_command += "\n"

    return custom_command

def torsion(
    command_label: str, 
    atoms: Union[List[str], str]
    ) -> str:
    """
    Create a PLUMED TORSION command, which gives a dihedral angle.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms, as a list of atom definitions or a single comma-separated string.
        Either 4 atoms or a single shortcut such as ``@phi-12``.

    Returns
    -------
    torsion_command : str
        PLUMED TORSION command.
    """
    
    # Check if atoms is a list of strings or a string
    if isinstance(atoms, list):
        # Convert all atoms to strings
        atoms = [str(atom) for atom in atoms]
        # Create TORSION command
        torsion_command = command_label + ": TORSION ATOMS=" + ",".join(atoms)

    elif isinstance(atoms, str):
        # Convert atoms to string
        atoms = str(atoms)
        # Create TORSION command
        torsion_command = command_label + ": TORSION ATOMS=" + atoms

    else:
        logger.error("Atoms must be a list of strings or a string.")
        sys.exit()

    # Add newline
    torsion_command += "\n"

    return torsion_command
  
def sin_old(
    command_label: str, 
    atoms: Union[List[str], str]
    ) -> str:
    """
    Old way to compute a sine-like feature of a dihedral using the PLUMED ALPHABETA command.

    Calls ``alphabeta`` with a reference angle of -pi/2, which gives
    ``0.5*(1+cos(phi+pi/2)) = 0.5*(1-sin(phi))``, where phi is the dihedral angle.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms that define the dihedral.

    Returns
    -------
    str
        PLUMED ALPHABETA command.
    """
    return alphabeta(command_label, atoms, reference = -round(math.pi/2,4))

def cos_old(
    command_label: str, 
    atoms: Union[List[str], str]
    ) -> str:
    """
    Old way to compute a cosine-like feature of a dihedral using the PLUMED ALPHABETA command.

    Calls ``alphabeta`` with a reference angle of 0, which gives ``0.5*(1+cos(phi))``,
    where phi is the dihedral angle.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms that define the dihedral.

    Returns
    -------
    str
        PLUMED ALPHABETA command.
    """
    return alphabeta(command_label, atoms, reference = 0)

def alphabeta(
    command_label: str, 
    atoms: Union[List[str], str], 
    reference: float
    ) -> str:
    """
    Create a PLUMED ALPHABETA command.

    It gives the cosine of a dihedral shifted and scaled to the range [0, 1]:
    ``0.5*(1+cos(phi-ref))``, where phi is the dihedral angle and ref the reference angle.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms that define the dihedral.
    reference : float
        Reference angle in radians.

    Returns
    -------
    alphabeta_command : str
        PLUMED ALPHABETA command.
    """

    # Check if atoms is a list of strings or a string
    if isinstance(atoms, list):
        # Convert all atoms to strings
        atoms = [str(atom) for atom in atoms]
        # Create ALPHABETA command
        alphabeta_command = command_label + ": ALPHABETA ATOMS1=" + ",".join(atoms)

    elif isinstance(atoms, str):
        # Convert atoms to string
        atoms = str(atoms)
        # Create ALPHABETA command
        alphabeta_command = command_label + ": ALPHABETA ATOMS1=" + atoms
   
    else:
        logger.error("Atoms must be a list of strings or a string.")
        sys.exit()

    # Add reference
    alphabeta_command += " REFERENCE=" + str(reference)

    # Add newline
    alphabeta_command += "\n"

    return alphabeta_command

def read(
    command_label, 
    file_path, 
    values, 
    ignore_time
    ) -> str:
    """
    Create a PLUMED READ command, which reads values from a colvars file.

    Parameters
    ----------
    command_label : str
        Label of the command.
    file_path : str
        Path to the file to read.
    values : str
        Name of the column(s) to read.
    ignore_time : bool
        If True, the time column of the file is ignored.

    Returns
    -------
    read_command : str
        PLUMED READ command.
    """

    # Create READ command
    read_command = command_label + ": READ FILE=" + file_path + " VALUES=" + values

    # Add IGNORE_TIME keyword
    if ignore_time:
        read_command += " IGNORE_TIME"

    # Add newline
    read_command += "\n"

    return read_command

def combine(
    command_label: str,
    arguments: List[str],
    coefficients: Optional[np.array] = None,
    parameters: Optional[np.array] = None,
    powers: Optional[np.array] = None,
    periodic: bool = False
    ) -> str:
    """
    Create a PLUMED COMBINE command, which computes a combination of variables.

    The result is ``C = Sum_i [ c_i * (x_i - a_i)^p_i ]``. Used for example to build linear CVs
    or to normalize a variable.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the input variables (x_i).
    coefficients : array-like, optional
        Coefficients (c_i). Default is None (PLUMED default).
    parameters : array-like, optional
        Offsets (a_i). Default is None (PLUMED default).
    powers : array-like, optional
        Powers (p_i). Default is None (PLUMED default).
    periodic : bool, optional
        Whether the result is periodic. Default is False.

    Returns
    -------
    combine_command : str
        PLUMED COMBINE command.
    """
    
    # Create COMBINE command
    combine_command = command_label + ": COMBINE ARG=" + ",".join(arguments)

    if coefficients is not None:
        # Add coefficients
        combine_command += " COEFFICIENTS="
        for coefficient in coefficients:
            combine_command += f"{coefficient:.17g},"
        combine_command = combine_command[:-1]
    
    if parameters is not None:
        # Add parameters
        combine_command += " PARAMETERS="
        for parameter in parameters:
            combine_command += f"{parameter:.17g},"
        combine_command = combine_command[:-1]
    
    if powers is not None:
        # Add powers
        combine_command += " POWERS="
        for power in powers:
            combine_command += f"{power:.10g},"
        combine_command = combine_command[:-1]

    # Add periodic keyword
    if periodic:
        combine_command += " PERIODIC=YES"
    else:
        combine_command += " PERIODIC=NO"
        
    # Add newline
    combine_command += "\n"
    
    return combine_command

def rmsd(
    command_label: str, 
    reference: str, 
    type: str = "OPTIMAL"
    ) -> str:
    """
    Create a PLUMED RMSD command, which gives the RMSD with respect to a reference structure.

    Parameters
    ----------
    command_label : str
        Label of the command.
    reference : str
        Path to the reference PDB. Occupancy marks alignment atoms and B-factor marks RMSD atoms.
    type : str, optional
        Type of RMSD calculation. Default is ``"OPTIMAL"``.

    Returns
    -------
    rmsd_command : str
        PLUMED RMSD command.
    """

    # Create RMSD command
    rmsd_command = command_label + ": RMSD REFERENCE=" + reference + " TYPE=" + type + " \n"

    return rmsd_command

def upper_walls(
    command_label: str, 
    arguments: List[str],
    at_eqs:  Optional[List[float]] = None,
    kappas:  Optional[List[float]] = None,
    exponents: Optional[List[int]] = None,
    epsilons: Optional[List[float]] = None,
    offsets: Optional[List[float]] = None
    ) -> str:
    """
    Create a PLUMED UPPER_WALLS command.

    It adds a bias that pushes each variable back when it goes above a given value.
    Each list has one value per argument. Keywords left as None are not added.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the variables to restrain.
    at_eqs : list of float, optional
        Position of the wall for each variable (AT).
    kappas : list of float, optional
        Force constants (KAPPA).
    exponents : list of int, optional
        Exponents (EXP).
    epsilons : list of float, optional
        Rescaling factors (EPS).
    offsets : list of float, optional
        Offsets (OFFSET).

    Returns
    -------
    upper_walls_command : str
        PLUMED UPPER_WALLS command.
    """
    
    # Create UPPER_WALLS command
    upper_walls_command = command_label + ": UPPER_WALLS ARG=" + ",".join(arguments)
    
    if at_eqs is not None:
        # Add at_eqs
        upper_walls_command += " AT="
        for at_eq in at_eqs:
            upper_walls_command += f"{at_eq:.10g},"
        upper_walls_command = upper_walls_command[:-1]
        
    if kappas is not None:
        # Add kappas
        upper_walls_command += " KAPPA="
        for kappa in kappas:
            upper_walls_command += f"{kappa:.10g},"
        upper_walls_command = upper_walls_command[:-1]
    
    if exponents is not None:
        # Add exponents
        upper_walls_command += " EXP="
        for exponent in exponents:
            upper_walls_command += f"{exponent:.10g},"
        upper_walls_command = upper_walls_command[:-1]
    
    if epsilons is not None:
        # Add epsilons
        upper_walls_command += " EPS="
        for epsilon in epsilons:
            upper_walls_command += f"{epsilon:.10g},"
        upper_walls_command = upper_walls_command[:-1]
    
    if offsets is not None:
        # Add offsets
        upper_walls_command += " OFFSET="
        for offset in offsets:
            upper_walls_command += f"{offset:.10g},"
        upper_walls_command = upper_walls_command[:-1]
    
    # Add newline
    upper_walls_command += "\n"

    return upper_walls_command

def print(
    arguments: List[str], 
    file_path: str, 
    stride: int = 1, 
    fmt: str = "%.4f"
    ) -> str:
    """
    Create a PLUMED PRINT command, which writes variables to a colvars file.

    A colvars (COLVAR) file is a text table written by PLUMED with one column per variable
    and one row per saved step.

    Parameters
    ----------
    arguments : list of str
        Labels of the variables to print.
    file_path : str
        Path to the output colvars file.
    stride : int, optional
        Print every ``stride`` steps. Default is 1.
    fmt : str, optional
        Number format. Default is ``"%.4f"``.

    Returns
    -------
    print_command : str
        PLUMED PRINT command.
    """

    # Create PRINT command
    print_command = "PRINT ARG="

    # Add arguments
    for arg in arguments:
        print_command += arg + ","

    # Remove last comma
    print_command = print_command[:-1]

    # Add file name
    print_command += " FILE=" + file_path

    # Add stride
    print_command += " STRIDE=" + str(stride)
    
    # Add FMT
    print_command += f" FMT={fmt}"

    # Add newline
    print_command += "\n"

    return print_command 

def histogram(
    command_label,
    arguments,
    grid_mins,
    grid_maxs,
    stride,
    kernel,
    normalization,
    grid_bins = [500],
    bandwidths = [0.01],
    weights_label = None,
    clear_freq = None
    ) -> str:
    """
    Create a PLUMED HISTOGRAM command, which accumulates a histogram of variables on a grid.

    Grid lists have one value per argument.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the variables.
    grid_mins : list of float
        Lower limit of the grid.
    grid_maxs : list of float
        Upper limit of the grid.
    stride : int
        Use one of every ``stride`` steps (1 = use all).
    kernel : str
        Kernel used for kernel density estimation (e.g. ``"GAUSSIAN"``).
    normalization : str
        Type of normalization.
    grid_bins : list of int, optional
        Number of bins. Default is ``[500]``.
    bandwidths : list of float, optional
        Kernel bandwidths, only used with the ``GAUSSIAN`` kernel. Default is ``[0.01]``.
    weights_label : str, optional
        Label of the log-weights (LOGWEIGHTS), used to reweight biased data. Default is None.
    clear_freq : int, optional
        Clear the accumulated data every ``clear_freq`` steps (used in block analysis). Default is None.

    Returns
    -------
    histogram_command : str
        PLUMED HISTOGRAM command.
    """

    # Create HISTOGRAM command
    histogram_command = command_label + ": HISTOGRAM ARG="

    # Add arguments
    for arg in arguments:
        histogram_command += arg + ","

    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add stride
    histogram_command += " STRIDE=" + str(stride)

    # Add weights label if present
    if weights_label is not None:
        histogram_command += " LOGWEIGHTS=" + weights_label

    # Add min grid keyword
    histogram_command += " GRID_MIN=" 
    
    # Add grid min values
    for grid_min in grid_mins:
        histogram_command += f"{grid_min:.10g},"

    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add max grid keyword
    histogram_command += " GRID_MAX="

    # Add grid max values
    for grid_max in grid_maxs:
        histogram_command += f"{grid_max:.10g},"

    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add grid bin keyword
    histogram_command += " GRID_BIN="

    # Add grid bin values
    for grid_bin in grid_bins:
        histogram_command += f"{grid_bin:.10g}," 
    
    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add kernel keyword
    histogram_command += " KERNEL=" + kernel

    if kernel == "GAUSSIAN":
        
        # Add bandwidth keyword
        histogram_command += " BANDWIDTH=" 
    
        # Add bandwidth values
        for bandwidth in bandwidths:
            histogram_command += f"{bandwidth:.10g},"

        # Remove last comma
        histogram_command = histogram_command[:-1]

    # Add normalization keyword
    histogram_command += " NORMALIZATION=" + normalization

    # Add clear frequency keyword
    if clear_freq is not None:
        histogram_command += " CLEAR=" + str(clear_freq)

    # Add newline
    histogram_command += "\n"

    return histogram_command

def dumpgrid(
    arguments, 
    file_path, 
    stride = None
    ) -> str:
    """
    Create a PLUMED DUMPGRID command, which writes a grid (e.g. a histogram or FES) to a file.

    Parameters
    ----------
    arguments : list of str
        Labels of the grids to write.
    file_path : str
        Path to the output file.
    stride : int, optional
        Write every ``stride`` steps. Default is None (write only at the end).

    Returns
    -------
    dumpgrid_command : str
        PLUMED DUMPGRID command.
    """

    # Create DUMPGRID command
    dumpgrid_command = "DUMPGRID GRID="

    # Add arguments
    for arg in arguments:
        dumpgrid_command += arg + ","

    # Remove last comma
    dumpgrid_command = dumpgrid_command[:-1]

    # Add file name
    dumpgrid_command += " FILE=" + file_path

    # Add default format
    dumpgrid_command += f" FMT={DEFAULT_FMT}"

    # Add stride
    if stride is not None:
        dumpgrid_command += " STRIDE=" + str(stride)

    # Add newline
    dumpgrid_command += "\n"

    return dumpgrid_command

def convert_to_fes(
    command_label, 
    arguments, 
    temp, 
    mintozero = True
    ) -> str:
    """
    Create a PLUMED CONVERT_TO_FES command, which turns a histogram into a free energy surface.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the input grids.
    temp : float
        Temperature.
    mintozero : bool, optional
        If True, shift the FES so its minimum is zero. Default is True.

    Returns
    -------
    convert_to_fes_command : str
        PLUMED CONVERT_TO_FES command.
    """

    # Create CONVERT_TO_FES command
    convert_to_fes_command = command_label + ": CONVERT_TO_FES GRID="

    # Add arguments
    for arg in arguments:
        convert_to_fes_command += arg + ","

    # Remove last comma
    convert_to_fes_command = convert_to_fes_command[:-1]

    # Add temperature
    convert_to_fes_command += " TEMP=" + str(temp)

    # Add mintozero
    if mintozero:
        convert_to_fes_command += " MINTOZERO"

    # Add newline
    convert_to_fes_command += "\n"

    return convert_to_fes_command 

def reweight_bias(
    command_label, 
    arguments, 
    temp
    ) -> str:
    """
    Create a PLUMED REWEIGHT_BIAS command, which computes weights to remove the effect of a bias.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the bias variables.
    temp : float
        Temperature.

    Returns
    -------
    reweight_bias_command : str
        PLUMED REWEIGHT_BIAS command.
    """

    # Create REWEIGHT_BIAS command
    reweight_bias_command = command_label + ": REWEIGHT_BIAS ARG="

    # Add arguments
    for arg in arguments:
        reweight_bias_command += arg + ","

    # Remove last comma
    reweight_bias_command = reweight_bias_command[:-1]

    # Add temperature
    reweight_bias_command += " TEMP=" + str(temp)

    # Add newline
    reweight_bias_command += "\n"

    return reweight_bias_command

def external(
    command_label, 
    arguments, 
    file
    ) -> str:
    """
    Create a PLUMED EXTERNAL command, which applies a bias read from a grid file.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the biased variables.
    file : str
        Path to the grid file with the bias.

    Returns
    -------
    external_command : str
        PLUMED EXTERNAL command.
    """

    # Create EXTERNAL command
    external_command = command_label + ": EXTERNAL ARG=" 

    # Add arguments
    for arg in arguments:
        external_command += arg + ","

    # Remove last comma
    external_command = external_command[:-1]

    # Add file name
    external_command += " FILE=" + file

    # Add newline
    external_command += "\n"

    return external_command

def opes_metad(
    command_label: str, 
    arguments: List[str], 
    temperature: float, 
    pace: int, 
    sigmas: float, 
    barrier: float, 
    compression_threshold: float
    ) -> str:
    """
    Create a PLUMED OPES_METAD command, which applies an OPES enhanced sampling bias.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the biased variables (usually the CV components).
    temperature : float
        Temperature.
    pace : int
        Update the bias every ``pace`` steps.
    sigmas : list of float
        Initial kernel width, one per argument.
    barrier : float
        Expected height of the free energy barriers to overcome.
    compression_threshold : float
        Threshold used to merge nearby kernels.

    Returns
    -------
    opes_metad_command : str
        PLUMED OPES_METAD command.
    """

    # Start OPES_METAD command
    opes_metad_command = "OPES_METAD ...\n"

    # Add command label
    opes_metad_command += " LABEL=" + command_label + "\n"

    # Add arguments
    opes_metad_command += " ARG=" + ",".join(arguments) + "\n"

    # Add temperature
    opes_metad_command += " TEMP=" + f"{temperature:.10g}\n"

    # Add pace
    opes_metad_command += " PACE=" + str(pace) + "\n"

    # Add sigmas
    opes_metad_command += " SIGMA=" + ",".join([f"{sigma:.10g}" for sigma in sigmas]) + "\n"

    # Add barrier
    opes_metad_command += " BARRIER=" + f"{barrier:.10g}\n"

    # Add compression threshold
    opes_metad_command += " COMPRESSION_THRESHOLD=" + f"{compression_threshold:.10g}\n"

    # End OPES_METAD command
    opes_metad_command += "... OPES_METAD\n"

    return opes_metad_command

def opes_metad_explore(
    command_label: str, 
    arguments: List[str], 
    temperature: float, 
    pace: int, 
    sigmas: float, 
    barrier: float, 
    compression_threshold: float
    ) -> str:
    """
    Create a PLUMED OPES_METAD_EXPLORE command.

    This OPES variant explores the CV space faster, at the cost of slower convergence.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the biased variables (usually the CV components).
    temperature : float
        Temperature.
    pace : int
        Update the bias every ``pace`` steps.
    sigmas : list of float
        Initial kernel width, one per argument.
    barrier : float
        Expected height of the free energy barriers to overcome.
    compression_threshold : float
        Threshold used to merge nearby kernels.

    Returns
    -------
    opes_metad_explore_command : str
        PLUMED OPES_METAD_EXPLORE command.
    """

    # Start OPES_METAD_EXPLORE command
    opes_metad_explore_command = "OPES_METAD_EXPLORE ...\n"

    # Add command label
    opes_metad_explore_command += " LABEL=" + command_label + "\n"

    # Add arguments
    opes_metad_explore_command += " ARG=" + ",".join(arguments) + "\n"

    # Add temperature
    opes_metad_explore_command += " TEMP=" + f"{temperature:.10g}\n"

    # Add pace
    opes_metad_explore_command += " PACE=" + str(pace) + "\n"

    # Add sigma
    opes_metad_explore_command += " SIGMA=" + ",".join([f"{sigma:.10g}" for sigma in sigmas]) + "\n"

    # Add barrier
    opes_metad_explore_command += " BARRIER=" + f"{barrier:.10g}\n"

    # Add compression threshold
    opes_metad_explore_command += " COMPRESSION_THRESHOLD=" + f"{compression_threshold:.10g}\n"

    # End OPES_METAD_EXPLORE command
    opes_metad_explore_command += "... OPES_METAD_EXPLORE\n"

    return opes_metad_explore_command

def opes_expanded(
    command_label: str, 
    arguments: List[str], 
    pace: int, 
    observation_steps: int
    ) -> str:
    """
    Create a PLUMED OPES_EXPANDED command.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the expansion collective variables.
    pace : int
        Update the bias every ``pace`` steps.
    observation_steps : int
        Number of steps used to gather information before applying the bias.

    Returns
    -------
    opes_expanded_command : str
        PLUMED OPES_EXPANDED command.
    """

    # Start OPES_EXPANDED command
    opes_expanded_command = "OPES_EXPANDED ...\n"

    # Add command label
    opes_expanded_command += " LABEL=" + command_label + "\n"

    # Add arguments
    opes_expanded_command += " ARG=" + ",".join(arguments) + "\n"

    # Add pace
    opes_expanded_command += " PACE=" + str(pace) + "\n"

    # Add observation steps
    opes_expanded_command += " OBSERVATION_STEPS=" + str(observation_steps) + "\n"

    # End OPES_EXPANDED command
    opes_expanded_command += "... OPES_EXPANDED\n"

    return opes_expanded_command

def metad(
    command_label: str, 
    arguments: List[str], 
    sigmas: List[float], 
    height: float, 
    bias_factor: int, 
    temperature: float, 
    pace: int, 
    grid_mins: List[float], 
    grid_maxs: List[float], 
    grid_bins: List[int]
    ) -> str:  
    """
    Create a PLUMED METAD command for (well-tempered) metadynamics.

    Metadynamics adds Gaussian hills along the CVs to push the system out of visited states.
    The command also asks PLUMED to compute c(t) (CALC_RCT), used for reweighting.
    List arguments have one value per CV component.

    Parameters
    ----------
    command_label : str
        Label of the command.
    arguments : list of str
        Labels of the biased variables.
    sigmas : list of float
        Width of the Gaussian hills.
    height : float
        Height of the Gaussian hills.
    bias_factor : int
        Well-tempered bias factor.
    temperature : float
        Temperature.
    pace : int
        Add a hill every ``pace`` steps.
    grid_mins : list of float
        Lower limit of the grid used to store the bias.
    grid_maxs : list of float
        Upper limit of the grid.
    grid_bins : list of int
        Number of grid bins.

    Returns
    -------
    metad_command : str
        PLUMED METAD command.
    """

    # Start METAD command
    metad_command = "METAD ...\n"

    # Add command label
    metad_command += "LABEL=" + command_label + "\n"

    # Add arguments
    metad_command += "ARG="

    for arg in arguments:
        metad_command += arg + ","

    # Remove last comma
    metad_command = metad_command[:-1]

    # Add sigmas 
    metad_command += "\nSIGMA=" + ",".join([f"{sigma:.6g}" for sigma in sigmas])

    # Add height
    metad_command += "\nHEIGHT=" + f"{height:.10g}"

    # Add bias_factor
    metad_command += "\nBIASFACTOR=" + f"{bias_factor:.10g}"

    # Add temperature
    metad_command += "\nTEMP=" + f"{temperature:.10g}"

    # Add pace
    metad_command += "\nPACE=" + str(pace)

    # Add grid mins using .join()
    metad_command += "\nGRID_MIN=" + ",".join([f"{grid_min:.10g}" for grid_min in grid_mins])

    # Add grid maxs
    metad_command += "\nGRID_MAX=" + ",".join([f"{grid_max:.10g}" for grid_max in grid_maxs])

    # Add grid bins
    metad_command += "\nGRID_BIN=" + ",".join([f"{grid_bin:.10g}" for grid_bin in grid_bins])

    # Add c(t) calculation
    metad_command += "\nCALC_RCT"

    # End METAD command
    metad_command += "\n... METAD\n"

    return metad_command

def com(
    command_label, 
    atoms
    ) -> str:
    """
    Create a PLUMED COM command, which defines a virtual atom at the center of mass of a group of atoms.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms, as a list of atom definitions or a single comma-separated string.

    Returns
    -------
    com_command : str
        PLUMED COM command.
    """

    # Check if atoms is a list of strings or a string
    if isinstance(atoms, list):

        # Convert all atoms to strings
        atoms = [str(atom) for atom in atoms]

        # Create COM command
        com_command = command_label + ": COM ATOMS=" + ",".join(atoms)

    elif isinstance(atoms, str):

        # Convert atoms to string
        atoms = str(atoms)

        # Create COM command
        com_command = command_label + ": COM ATOMS=" + atoms
    
    else:
        logger.error("Atoms must be a list of strings or a string.")
        sys.exit()
    
    # Add newline
    com_command += "\n"

    return com_command

def center(
    command_label, 
    atoms
    ) -> str:
    """
    Create a PLUMED CENTER command, which defines a virtual atom at the geometric center of a group of atoms.

    Parameters
    ----------
    command_label : str
        Label of the command.
    atoms : list or str
        Atoms, as a list of atom definitions or a single comma-separated string.

    Returns
    -------
    center_command : str
        PLUMED CENTER command.
    """

    # Check if atoms is a list of strings or a string
    if isinstance(atoms, list):

        # Convert all atoms to strings
        atoms = [str(atom) for atom in atoms]

        # Create CENTER command
        center_command = command_label + ": CENTER ATOMS=" + ",".join(atoms)

    elif isinstance(atoms, str):

        # Convert atoms to string
        atoms = str(atoms)

        # Create CENTER command
        center_command = command_label + ": CENTER ATOMS=" + atoms
    
    else:
        logger.error("Atoms must be a list of strings or a string.")
        sys.exit()

    # Add newline
    center_command += "\n"

    return center_command

def pytorch_model(
    command_label, 
    arguments, 
    model_path
    ) -> str:
    """
    Create a PLUMED PYTORCH_MODEL command, which evaluates a TorchScript model (e.g. a trained CV).

    Parameters
    ----------
    command_label : str
        Label of the command. The outputs are called ``<label>.node-0``, ``<label>.node-1``, etc.
    arguments : list of str
        Labels of the input variables (features), in the order the model expects.
    model_path : str
        Path to the TorchScript model file.

    Returns
    -------
    pytorch_model_command : str
        PLUMED PYTORCH_MODEL command.
    """

    # Create PYTORCH_MODEL command
    pytorch_model_command = command_label + ": PYTORCH_MODEL "

    # Add FILE with model_path
    pytorch_model_command += "FILE=" + model_path + " "

    # Add ARGs
    pytorch_model_command += "ARG="
    pytorch_model_command = pytorch_model_command + ",".join(arguments)

    # Add newline
    pytorch_model_command += "\n"

    return pytorch_model_command
