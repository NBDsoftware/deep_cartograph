# Import modules
import sys
import math
import logging
import numpy as np

from typing import List, Union

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# PLUMED Commands
# ---------------
#
# Functions to create PLUMED commands
def molinfo(topology: str, moltype: str = None) -> str:
    '''
    Function that creates a PLUMED MOLINFO command.

    Inputs
    ------

        topology    (str):  path to the topology file
        moltype     (str):  molecule type

    Returns
    -------

        molinfo_command (str):  PLUMED MOLINFO command
    '''   
    command = f"MOLINFO STRUCTURE={topology}"

    if moltype is not None:
        command += f" MOLTYPE={moltype}"
    
    command += "\n"

    return command

def wholemolecules(indices: List[int]) -> str:
    '''
    Function that creates a PLUMED WHOLEMOLECULES command.

    NOTE: If needed, this could be extended to consider multiple entities. (e.g. WHOLEMOLECULES ENTITY0=1-10 ENTITY1=11-20)
    
    Inputs
    ------

        indices     :  list of molecule indices

    Returns
    -------

        wholemolecules_command (str):  PLUMED WHOLEMOLECULES command
    '''   
    command = f"WHOLEMOLECULES ENTITY0={indices[0]}-{indices[-1]} \n"

    return command

def distance(command_label: str, atoms: Union[List[str], str]) -> str:
    '''
    Function that creates a PLUMED DISTANCE command.

    Inputs
    ------

        command_label   :  command label
        atoms           :  list of strings or string defining the atoms 
    
    Outputs
    -------

        distance_command    (str):          PLUMED DISTANCE command
    '''
  
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
    distance_command += "\n"

    return distance_command

def torsion(command_label: str, atoms: Union[List[str], str]) -> str:
    '''
    Function that creates a PLUMED TORSION command.

    Inputs
    ------

        command_label   :                 command label
        atoms           :  list of strings or string defining the atoms

    Outputs
    -------

        torsion_command :                 PLUMED TORSION command
    '''
    
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

def sin(command_label: str, atoms: Union[List[str], str]) -> str:
    """
    Proxy for the PLUMED ALPHABETA command using a reference angle of -pi/2 radians to convert the cosinus to a sinus.
    
    alphabeta = 0.5*(1+cos(phi-pi/2)) 

    where phi is the torsion angle defined by the atoms

    Inputs
    ------

        command_label   :  command label
        atoms           :  list of strings or string defining the atoms that define the phi torsion angle
    
    Outputs
    -------

        sin_command     :                 PLUMED ALPHABETA command for the sinus
    """
    return alphabeta(command_label, atoms, reference = -round(math.pi/2,4))

def cos(command_label: str, atoms: Union[List[str], str]) -> str:
    """
    Proxy for the PLUMED ALPHABETA command using a reference angle of 0 radians.
    
    alphabeta = 0.5*(1+cos(phi-0)) 

    where phi is the torsion angle defined by the atoms

    Inputs
    ------

        command_label   :  command label
        atoms           :  list of strings or string defining the atoms that define the phi torsion angle
    
    Outputs
    -------

        cos_command     :                 PLUMED ALPHABETA command for the cosinus
    """
    return alphabeta(command_label, atoms, reference = 0)

def alphabeta(command_label: str, atoms: Union[List[str], str], reference: float) -> str:
    """
    Function that creates an PLUMED ALPHABETA command. The command returns the cosinus moved up and squished between 0 and 1:
    
    alphabeta = 0.5*(1+cos(phi-ref)) 
    
    where phi is the torsion angle defined by the atoms and ref is the reference angle in radians.

    Inputs
    ------

        command_label   :  command label
        atoms           :  list of strings or string defining the atoms that define the phi torsion angle
        reference       :  list of floats defining the reference values
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

def read(command_label, file_path, values, ignore_time):
    '''
    Function that creates a PLUMED READ command.

    Inputs
    ------

        command_label   (str):              command label
        file_path       (str):              file path
        values          (str):              values
        ignore_time     (bool):             True if time must be ignored

    Outputs
    -------

        read_command    (str):              PLUMED READ command
    '''

    # Create READ command
    read_command = command_label + ": READ FILE=" + file_path + " VALUES=" + values

    # Add IGNORE_TIME keyword
    if ignore_time:
        read_command += " IGNORE_TIME"

    # Add newline
    read_command += "\n"

    return read_command

def combine(command_label: str, arguments: List[str], coefficients: Union[np.array, None] = None, parameters:  Union[np.array, None] = None, 
            powers: Union[np.array, None] = None, periodic: bool = False):
    '''
    Function that creates a PLUMED COMBINE command. The output from this command is the linear combination of the input arguments:
    
        C = Sum_i [ c_i * (x_i - a_i)^p_i ] 

    Inputs
    ------

        command_label   :       command label
        arguments       :       arguments -> x_i
        coefficients    :       coefficients -> c_i
        parameters      :       parameters -> a_i
        powers          :       powers -> p_i
        periodic        :       True if the RC is periodic

    Outputs
    -------

        combine_command (str):              PLUMED COMBINE command
    '''
    
    # Create COMBINE command
    combine_command = command_label + ": COMBINE ARG=" + ",".join(arguments)

    if coefficients is not None:
        # Add coefficients
        combine_command += " COEFFICIENTS="
        for coefficient in coefficients:
            combine_command += str(round(coefficient, 5)) + ","
        combine_command = combine_command[:-1]
    
    if parameters is not None:
        # Add parameters
        combine_command += " PARAMETERS="
        for parameter in parameters:
            combine_command += str(round(parameter, 5)) + ","
        combine_command = combine_command[:-1]
    
    if powers is not None:
        # Add powers
        combine_command += " POWERS="
        for power in powers:
            combine_command += str(round(power, 5)) + ","
        combine_command = combine_command[:-1]

    # Add periodic keyword
    if periodic:
        combine_command += " PERIODIC=YES"
    else:
        combine_command += " PERIODIC=NO"
        
    # Add newline
    combine_command += "\n"
    
    return combine_command

def print(arguments, file_path, stride = 1, fmt = "%.4f"):
    '''
    Function that creates a PLUMED PRINT command.

    Inputs
    ------

        arguments       (list of str):      arguments
        file_path       (str):              output colvars file path
        stride          (int):              stride

    Outputs
    -------

        print_command   (str):              PLUMED PRINT command
    '''

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

def histogram(command_label, arguments, grid_mins, grid_maxs, stride, kernel, normalization, grid_bins = [500], bandwidths = [0.01], weights_label = None, clear_freq = None):
    '''
    Function that creates a PLUMED HISTOGRAM command.

    Inputs
    ------

        command_label     (str):             command label
        arguments         (list of str):     list of arguments
        grid_mins         (list of float):   list of values for grid minimum
        grid_maxs         (list of float):   list of values for grid maximum
        stride            (int):             stride for command (1=read all)
        kernel            (str):             kernel used for kernel density estimation
        normalization     (str):             type of normalization
        grid_bins         (list of int):     list of values for grid bins
        bandwidths        (list of float):   list of values for bandwidth
        weights_label     (str):             weights label
        clear_freq        (int):             frequency for clearing accumulated data to compute the histogram - used in block analysis

    Outputs
    -------

        histogram_command (str):            PLUMED HISTOGRAM command
    '''

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
        histogram_command += str(round(grid_min,3)) + ","

    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add max grid keyword
    histogram_command += " GRID_MAX="

    # Add grid max values
    for grid_max in grid_maxs:
        histogram_command += str(round(grid_max,3)) + ","

    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add grid bin keyword
    histogram_command += " GRID_BIN="

    # Add grid bin values
    for grid_bin in grid_bins:
        histogram_command += str(int(grid_bin)) + ","
    
    # Remove last comma
    histogram_command = histogram_command[:-1]

    # Add kernel keyword
    histogram_command += " KERNEL=" + kernel

    if kernel == "GAUSSIAN":
        
        # Add bandwidth keyword
        histogram_command += " BANDWIDTH=" 
    
        # Add bandwidth values
        for bandwidth in bandwidths:
            histogram_command += str(round(bandwidth,3)) + ","

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

def dumpgrid(arguments, file_path, stride = None):
    '''
    Function that creates a PLUMED DUMPGRID command.

    Inputs
    ------

        arguments       (list of str):      arguments
        file_path       (str):              file name

    Outputs
    -------

        dumpgrid_command (str):             PLUMED DUMPGRID command
    '''

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

def convert_to_fes(command_label, arguments, temp, mintozero = True):
    '''
    Function that creates a PLUMED CONVERT_TO_FES command.

    Inputs
    ------

        command_label   (str):              command label
        arguments       (list of str):      arguments
        temp            (float):            temperature
        mintozero       (bool):             whether to set minimum to zero

    Outputs
    -------

        convert_to_fes_command (str):       PLUMED CONVERT_TO_FES command
    '''

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

def reweight_bias(command_label, arguments, temp):
    '''
    Function that creates a PLUMED REWEIGHT_BIAS command.

    Inputs
    ------

        command_label   (str):              command label
        arguments       (list of str):      arguments
        temp            (float):            temperature

    Outputs
    -------

        reweight_bias_command (str):        PLUMED REWEIGHT_BIAS command
    '''

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

def external(command_label, arguments, file):
    '''
    Function that creates a PLUMED EXTERNAL command.

    Inputs
    ------

        command_label          (str):   command label
        arguments      (list of str):   arguments
        file                   (str):   file name
    
    Outputs
    -------

        external_command (str):         PLUMED EXTERNAL command
    '''

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

def metad(command_label, arguments, sigmas, height, biasfactor, temp, pace, grid_mins, grid_maxs, grid_bins):
    """
    Function that creates a PLUMED METAD command.

    Inputs
    ------

        command_label   (str):                 command label
        arguments       (list of str):         arguments
        sigma           (list of  float):      sigmas
        height          (float):               height
        biasfactor      (int):                 biasfactor
        temp            (float):               temperature
        pace            (int):                 pace
        grid_mins       (list of float):       grid mins
        grid_maxs       (list of float):       grid maxs
        grid_bins       (list of int):         grid bins
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
    metad_command += "\nSIGMA=" + ",".join([str(round(sigma,2)) for sigma in sigmas])

    # Add height
    metad_command += "\nHEIGHT=" + str(round(height,4))

    # Add biasfactor
    metad_command += "\nBIASFACTOR=" + str(biasfactor)

    # Add temperature
    metad_command += "\nTEMP=" + str(temp)

    # Add pace
    metad_command += "\nPACE=" + str(pace)

    # Add grid mins using .join()
    metad_command += "\nGRID_MIN=" + ",".join([str(round(grid_min,4)) for grid_min in grid_mins])

    # Add grid maxs
    metad_command += "\nGRID_MAX=" + ",".join([str(round(grid_max,4)) for grid_max in grid_maxs])

    # Add grid bins
    metad_command += "\nGRID_BIN=" + ",".join([str(grid_bin) for grid_bin in grid_bins])

    # Add c(t) calculation
    metad_command += "\nCALC_RCT"

    # End METAD command
    metad_command += "\n... METAD\n"

    return metad_command

def com(command_label, atoms) -> str:
    """
    Function that creates a PLUMED COM command.

    Inputs
    ------

        command_label   (str):                 command label
        atoms           (list of str or str):  list of strings or string defining the atoms
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

def center(command_label, atoms) -> str:
    """
    Function that creates a PLUMED CENTER command.

    Inputs
    ------

        command_label   (str):                 command label
        atoms           (list of str or str):  list of strings or string defining the atoms
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

def pytorch_model(command_label, arguments, model_path) -> str:
    """
    Function that creates a PLUMED PYTORCH_MODEL command.

    Inputs
    ------

        command_label   (str):                 command label
        arguments       (list of str):         list of arguments
        model_path      (str):                 path to the PyTorch model
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
