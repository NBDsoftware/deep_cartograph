# Import modules
import os
import re
import sys
import math
import logging
import numpy as np
from pathlib import Path
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

    # Define new topology path
    plumed_topology_path = os.path.join(output_folder, "plumed_topology.pdb")

    # Create new topology file
    md.create_pdb(topology_path, plumed_topology_path)

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
    if configuration['features'].get('distance_groups') is not None:

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
    if configuration['features'].get('dihedral_groups') is not None:

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
    if configuration['features'].get('distance_to_center_groups') is not None:

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
    """ 
    PLUMED input file builder to analyze a CV from a colvars file.
    Computes its time series, biased histogram, unbiased histogram and free energy profile.

    Additionally, it computes the error in the free energy profile using block averaging.

        See:

        [1]: Quantifying Uncertainty and Sampling Quality in Biomolecular Simulations, Alan Grossfield and Daniel M. Zuckerman
        [2]: Plumed tutorial (Masterclass 21.2) on Statistical errors in MD: https://www.plumed.org/doc-v2.8/user-doc/html/masterclass-21-2.html

    Inputs
    ------

        colvars_metadata_list  (list of dict): list with dictionaries with colvar information
        biases                  (list of str): list with the names of the biases
        plumed_settings                (dict): dictionary with PLUMED settings
        md_settings                    (dict): dictionary with MD settings
        colvars_path                    (str): path to the colvars file
        output_folder                   (str): path to the output folder
        
    Outputs
    -------

        output_paths (dict): dictionary with the following keys:

            "b_histo"    : path to the colvars biased histogram file
            "u_histo"    : path to the colvars unbiased histogram file
            "fes"        : path to the colvars FES file
            "fes_err"    : path to the colvars FES error file
        
        plumed_input_path (str): path to the PLUMED input file
    """
    
    # Find labels of the collective variables
    colvar_labels = [metadata['label'] for metadata in colvars_metadata_list]

    # Define PLUMED input file name
    if md_settings['biased']:
        input_name = "plumed." + "_".join(colvar_labels) + "_analysis_biased.dat"
    else:
        input_name = "plumed." + "_".join(colvar_labels) + "_analysis.dat"
    plumed_file_path = common.create_output_path(input_name, output_folder, "statistics", '_'.join(colvar_labels), "plumed_data")

    # Find colvar dimension
    colvar_dimension = f"{len(colvar_labels)}D"

    # Find PLUMED analysis configuration
    stride = plumed_settings[colvar_dimension]['stride']
    kernel = str(plumed_settings[colvar_dimension]['kernel'])
    normalization = str(plumed_settings[colvar_dimension]['normalization'])
    bins = plumed_settings[colvar_dimension]['bins']
    bandwidth = plumed_settings[colvar_dimension]['bandwidth']
    # Find MD configuration
    total_time = md_settings['total_time']*1000 # Find total simulation time in ps
    total_time = total_time*1000/(md_settings['time_step']*md_settings['plumed_stride']) # Convert total simulation time to time units used in input colvars file

    # Find minimum block size: plumed supports a max of 101 blocks
    min_block_size = math.ceil(total_time/101)

    # Define output filenames
    b_histo_file = f"{'_'.join(colvar_labels)}_bhisto.dat"
    u_histo_file = f"{'_'.join(colvar_labels)}_uhisto.dat"
    fes_file = f"{'_'.join(colvar_labels)}_fes.dat"
    fes_err_file = f"{'_'.join(colvar_labels)}_fes_err.dat"

    # Define output file paths
    b_histo_path = common.create_output_path(b_histo_file, output_folder, "statistics", '_'.join(colvar_labels), "plumed_data")
    u_histo_path = common.create_output_path(u_histo_file, output_folder, "statistics", '_'.join(colvar_labels), "plumed_data")
    fes_path = common.create_output_path(fes_file, output_folder, "statistics", '_'.join(colvar_labels), "plumed_data")
    fes_err_path = common.create_output_path(fes_err_file, output_folder, "statistics", '_'.join(colvar_labels), "plumed_data")
    output_paths = {"b_histo": b_histo_path, "u_histo": u_histo_path, "fes": fes_path, "fes_err": fes_err_path}

    # Create PLUMED input file
    plumed_file = open(plumed_file_path, "w")

    # Read time series
    for colvar_label in colvar_labels:
        read_command = plumed_command.read(command_label=colvar_label, file_path=colvars_path, values=colvar_label, ignore_time=True)
        plumed_file.write(read_command)

    plumed_file.write("\n")
        
    # If previous MD was biased, re-weight the old biases
    if md_settings['biased']:
        
        # Define read labels for the old biases 
        reweight_args = []

        # For each bias
        for bias_label in biases:
        
            # Find bias component
            if ".rbias" in bias_label:
                bias_component = ".rbias"
                bias_name = bias_label.replace(".rbias", "")
            elif ".bias" in bias_label:
                bias_component = ".bias"   
                bias_name = bias_label.replace(".bias", "")
            
            read_command = plumed_command.read(command_label=bias_name, file_path=colvars_path, values=bias_label, ignore_time=True)
            plumed_file.write(read_command)

            # Add read label to list
            reweight_args.append(bias_name+bias_component)

        plumed_file.write("\n")

        # Compute weights
        reweight_bias_command = plumed_command.reweight_bias(command_label="weights", arguments=reweight_args, temp=md_settings['temperature'])
        plumed_file.write(reweight_bias_command)
        plumed_file.write("\n")
    
    grid_mins = []
    grid_maxs = []
    num_bins = []
    bandwidths = []
    relaxation_times = []

    # Find histogram parameters for each colvar
    for colvars_metadata in colvars_metadata_list:
        
        if "min" in colvars_metadata and "max" in colvars_metadata:
            grid_min = colvars_metadata["min"] - 0.05 * (colvars_metadata["max"] - colvars_metadata["min"])
            grid_max = colvars_metadata["max"] + 0.05 * (colvars_metadata["max"] - colvars_metadata["min"])

        grid_min, grid_max = colvars_metadata.get('range', [None, None])

        if grid_min is None or grid_max is None:
            logger.error(f"Range not defined and time series not computed for colvar {colvars_metadata['label']}. PLUMED needs the range to compute the histogram.")
            sys.exit(1)

        grid_mins.append(grid_min)
        grid_maxs.append(grid_max)
        num_bins.append(bins)
        bandwidths.append(bandwidth)

        if colvars_metadata.get('relaxation_time') is not None:
            relaxation_times.append(colvars_metadata["relaxation_time"])
        else:
            logger.warning(f"       Relaxation time not computed for colvar {colvars_metadata['label']} but calculation of FES error requested. The relaxation time is used to check the block size for the block analysis is big enough.")
    
    # If relaxation times is not empty
    if len(relaxation_times) > 0:

        # Keep largest relaxation time (ps)
        relaxation_time = max(relaxation_times)

        # Convert relaxation time to time units used in input colvars file
        relaxation_time = relaxation_time*1000/(md_settings['time_step']*md_settings['plumed_stride'])

        # Estimate appropriate block size from relaxation time - take into account that we might use a stride in the histogram command
        block_size = max(1000*math.ceil(relaxation_time/(stride*1000)), min_block_size)
    else:
        block_size = min_block_size

    # Create HISTOGRAM commands   
    if md_settings['biased']:

        biased_histogram_command = plumed_command.histogram("biased_histo", colvar_labels, grid_mins, grid_maxs,
                                                         stride, kernel, normalization, num_bins, bandwidths, clear_freq = block_size)
        unbiased_histogram_command = plumed_command.histogram("unbiased_histo", colvar_labels, grid_mins, grid_maxs, 
                                                           stride, kernel, normalization, num_bins, bandwidths, "weights", clear_freq = block_size)
        
        plumed_file.write(unbiased_histogram_command)
        plumed_file.write("\n")
        plumed_file.write(biased_histogram_command)
        plumed_file.write("\n")

        biased_dumpgrid_command = plumed_command.dumpgrid(arguments=["biased_histo"], file_path=b_histo_path, stride = block_size)
        plumed_file.write(biased_dumpgrid_command)
        plumed_file.write("\n")
    
    else:
        unbiased_histogram_command = plumed_command.histogram("unbiased_histo", colvar_labels, grid_mins, grid_maxs, 
                                                           stride, kernel, normalization, num_bins, bandwidths, stride = block_size)
        
        # Set file path to None
        b_histo_path = None

        plumed_file.write(unbiased_histogram_command)
        plumed_file.write("\n")

    # Create DUMPGRID command
    unbiased_dumpgrid_command = plumed_command.dumpgrid(arguments=["unbiased_histo"], file_path=u_histo_path, stride = block_size)

    # Write command
    plumed_file.write(unbiased_dumpgrid_command)
    plumed_file.write("\n")

    # Close PLUMED input file
    plumed_file.close()

    return output_paths, plumed_file_path