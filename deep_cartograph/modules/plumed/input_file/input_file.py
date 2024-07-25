# Import modules
import os
import logging
from typing import Dict, Tuple

# Import local modules
from deep_cartograph.modules.plumed.command import command as plumed_command
from deep_cartograph.modules.plumed import utils as plumed_utils
from deep_cartograph.modules.common import common
from deep_cartograph.modules.md import md

# Set logger
logger = logging.getLogger(__name__)

# Set constants
DEFAULT_FMT = '%14.10f'

# PLUMED Input Builders
# ---------------------
# 
# PLUMED input files are text files that contain the commands to be executed by PLUMED.
# These functions are used to build different PLUMED input files.

# Input Builders for new MD simulations
def track_features(configuration: Dict, topology_path: str, colvars_path: str, output_folder: str) -> Tuple[str, str]:
    '''
    PLUMED input file builder to track a collection of features (distances, torsion angles...)
    during an MD simulation.

    Inputs
    ------

        configuration       (dict): configuration dictionary
        topology_path        (str): topology file path
        colvars_path         (str): path to the colvars file
        output_folder        (str): path to the output folder
    '''

    ############### 
    # Create file #
    ###############

    # Define plumed input file name
    plumed_file_path = os.path.join(output_folder, f"{configuration['input_name']}.dat") 
    plumed_file = open(plumed_file_path, "w")

    ###################
    # Create topology #
    ###################

    # Define new topology path
    plumed_topology_path = os.path.join(output_folder, "plumed_topology.pdb")

    # Create new topology file
    md.create_pdb(topology_path, plumed_topology_path)

    ##################
    # Absolute paths #
    ##################

    # Find absolute paths
    plumed_topology_path = os.path.abspath(plumed_topology_path)
    colvars_path = os.path.abspath(colvars_path)

    ################
    # WRITE HEADER #
    ################

    # Write MOLINFO keyword - to use shortcuts for atoms
    molinfo_command = plumed_command.molinfo(plumed_topology_path, configuration['moltype'])
    plumed_file.write(molinfo_command)

    # Get the indices of the molecules that should be made whole
    whole_mol_indices = md.get_indices(plumed_topology_path, configuration['whole_molecule_selection'])

    # Write WHOLEMOLECULES keyword
    wholemolecules_command = plumed_command.wholemolecules(whole_mol_indices)
    plumed_file.write(wholemolecules_command)

    #############
    # DISTANCES #
    #############

    # Accumulate all command labels
    distance_commands = []
    
    # Check for distance groups
    if configuration['features'].get('distance_groups', {}) != {}:

        # Find list of groups
        distance_group_names = configuration['features']['distance_groups'].keys()

        # Iterate over groups
        for distance_group_name in distance_group_names:    

            # Comment name of group
            plumed_file.write(f"# {distance_group_name} distances \n")

            # Find group definition
            distance_group = configuration['features']['distance_groups'][distance_group_name]

            # Find atom and command labels for all pairs in this group
            distance_names, atomic_definitions = plumed_utils.get_distance_labels(plumed_topology_path, distance_group, distance_group_name)

            logger.info(f"Found {len(atomic_definitions)} features for {distance_group_name}")
            
            # Iterate over labels
            for command_label, atom_label in zip(distance_names, atomic_definitions):

                # Get distance command
                distance_command = plumed_command.distance(command_label, atom_label)

                # Write distance command to PLUMED input file
                plumed_file.write(distance_command)

            # Save command labels
            distance_commands.extend(distance_names)

            plumed_file.write(f"\n")

    #############
    # DIHEDRALS #
    #############

    # Accumulate all command labels
    dihedral_commands = []

    # Check for dihedral groups
    if configuration['features'].get('dihedral_groups', {}) != {}:

        # Find list of groups
        dihedral_group_names = configuration['features']['dihedral_groups'].keys()

        # Iterate over groups
        for dihedral_group_name in dihedral_group_names:

            # Comment name of group
            plumed_file.write(f"# {dihedral_group_name} dihedrals \n")

            # Find group definition
            dihedral_group = configuration['features']['dihedral_groups'][dihedral_group_name]

            # Find atom and command labels for all dihedrals in this group
            dihedral_names, atomic_definitions = plumed_utils.get_dihedral_labels(plumed_topology_path, dihedral_group, dihedral_group_name)
            
            if dihedral_group.get('periodic_encoding', True):
                logger.info(f"Found {2*len(atomic_definitions)} features for {dihedral_group_name}") # Each dihedral has 2 features (sin and cos)
            else:
                logger.info(f"Found {len(atomic_definitions)} features for {dihedral_group_name}")

            # Iterate over labels
            for command_label, atom_label in zip(dihedral_names, atomic_definitions):
                
                if dihedral_group.get('periodic_encoding', True):

                    # Get the commands for the sinus and cosinus of the torsion angle
                    sin_command = plumed_command.sin(f"sin_{command_label}", atom_label)
                    cos_command = plumed_command.cos(f"cos_{command_label}", atom_label)

                    # Write commands to PLUMED input file
                    plumed_file.write(sin_command)
                    plumed_file.write(cos_command)

                    # Save commands label
                    dihedral_commands.append(f"sin_{command_label}")
                    dihedral_commands.append(f"cos_{command_label}")

                else:
                    # Get the command for the torsion angle
                    dihedral_command = plumed_command.torsion(command_label, atom_label)

                    # Write torsion command to PLUMED input file
                    plumed_file.write(dihedral_command)

                    # Save command label
                    dihedral_commands.append(command_label)

            plumed_file.write(f"\n")

    ######################
    # DISTANCE TO CENTER #
    ######################

    # Accumulate all command labels
    distance_to_center_commands = []

    # Check for distance to com groups
    if configuration['features'].get('distance_to_center_groups', {}) != {}:

        # Find the list of groups
        distance_to_center_group_names = configuration['features']['distance_to_center_groups'].keys()

        # Iterate over groups
        for distance_to_center_group_name in distance_to_center_group_names:

            # Comment name of group
            plumed_file.write(f"# {distance_to_center_group_name} distances to com \n")

            # Find group definition
            distance_to_center_group = configuration['features']['distance_to_center_groups'][distance_to_center_group_name]

            # Get the CENTER command
            center_command_label = f"{distance_to_center_group_name}_com"
            center_command = plumed_command.center(center_command_label, md.get_indices(plumed_topology_path, distance_to_center_group['center_selection']))

            # Write CENTER command to PLUMED input file
            plumed_file.write(center_command)

            # Find atoms in selection to compute the distance to the CENTER 
            atoms = md.get_indices(plumed_topology_path, distance_to_center_group['selection'])

            logger.info(f"Found {len(atoms)} features for {distance_to_center_group_name}")

            # Iterate over atoms
            for atom in atoms:
                
                # Get distance command
                distance_command_label = f"d_{atom}_{center_command_label}"
                distance_command = plumed_command.distance(distance_command_label, [atom, center_command_label])

                # Write distance command to PLUMED input file
                plumed_file.write(distance_command)

                # Save command label
                distance_to_center_commands.append(distance_command_label)

    # Merge all command labels
    all_command_labels = distance_commands + dihedral_commands + distance_to_center_commands

    # Print command
    print_command = plumed_command.print(all_command_labels, colvars_path, configuration.get('traj_stride', 1))

    # Write print command to PLUMED input file
    plumed_file.write(print_command)

    return plumed_file_path, plumed_topology_path