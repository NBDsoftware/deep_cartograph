"""
Align trajectories tool: superimposes trajectories onto a reference topology using
the CA atoms of the residues they share (found through sequence alignment).
"""
import os
import time
import logging.config
from pathlib import Path
from typing import List, Optional, Union

import MDAnalysis as mda
import MDAnalysis.analysis.align as mda_align

from deep_cartograph.modules.bio import PDBTopologyMapper
from deep_cartograph.modules.common import check_data

########
# HELPERS #
########

def find_common_resids(ref_topology: str, topologies: List[str]) -> List[int]:
    """
    Find the residues of the reference topology that are present in all the target topologies.

    The reference sequence is aligned to each target sequence to match residues.

    Parameters
    ----------
    ref_topology : str
        Path to the reference topology (PDB file).

    topologies : List[str]
        Paths to the target topologies (PDB files).

    Returns
    -------
    List[int]
        Sorted residue IDs (numbered as in the reference) found in every target topology.
        Empty if `topologies` is empty.
    """
    logger = logging.getLogger("deep_cartograph")

    if not topologies:
        return []

    mapper = PDBTopologyMapper(ref_topology, topologies[0])
    common_resids = set(mapper.mapping.keys())
    logger.debug(f"Topology {Path(topologies[0]).name}: {len(common_resids)} residues mapped to reference.")

    for top in topologies[1:]:
        mapper = PDBTopologyMapper(ref_topology, top)
        current_resids = set(mapper.mapping.keys())
        logger.debug(f"Topology {Path(top).name}: {len(current_resids)} residues mapped to reference.")
        common_resids  = common_resids.intersection(current_resids)
        logger.debug(f"Common residues after processing {Path(top).name}: {len(common_resids)}")

    return sorted(common_resids)


def build_ca_selection(resids: List[int]) -> str:
    """
    Build an MDAnalysis selection string for the CA atoms of the given residues.

    Parameters
    ----------
    resids : List[int]
        Residue IDs to include.

    Returns
    -------
    str
        MDAnalysis selection string.
    """
    resid_str = " ".join(str(r) for r in resids)
    return f"backbone and name CA and resid {resid_str}"


########
# TOOL #
########

def align_trajectories(trajectory_data: Optional[Union[List[str], str]] = None,
                       topology_data: Optional[Union[List[str], str]] = None, 
                       ref_topology: Optional[str] = None,
                       output_folder: str = 'align_trajectories') -> None:
    """
    Align trajectories and topologies to a reference topology and save them in the output folder.

    The residues shared by all topologies are found through sequence alignment, and
    their CA atoms are used to superimpose each trajectory onto the reference. This
    works even if the systems have different sequences or residue numbering.

    For each input trajectory, the aligned trajectory (same file name) and its first
    frame as a PDB topology are written to `output_folder`.

    Parameters
    ----------
    trajectory_data : str or List[str], optional
        Trajectory file(s) or folder with trajectories to align.

    topology_data : str or List[str], optional
        Topology file(s) or folder with topologies for the trajectories. A single topology
        is used for all trajectories; otherwise each topology must have the same name as
        its trajectory.

    ref_topology : str, optional
        Reference topology to align to. Defaults to the first topology.

    output_folder : str, optional
        Path to the output folder. Default: 'align_trajectories'.
    """

    # Set logger
    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("==================")
    logger.info("Align Trajectories")
    logger.info("==================")

    # Start timer
    start_time = time.time()

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Check main data
    trajectories, topologies = check_data(trajectory_data, topology_data)

    if not trajectories:
        logger.warning("No trajectories provided. Nothing to align.")
        return

    # Determine reference topology
    if ref_topology is None:
        ref_topology = topologies[0]
        logger.info(f"No reference topology provided. Using first topology as reference: {Path(ref_topology).name}")

    logger.info(f"Reference topology: {Path(ref_topology).name}")

    # Find reference residue IDs that are present in all topologies
    logger.info("Finding common residues across all topologies via sequence alignment...")
    common_ref_resids = find_common_resids(ref_topology, topologies)
    logger.info(f"Found {len(common_ref_resids)} common residues across all topologies.")

    if not common_ref_resids:
        logger.error("No common residues found across topologies. Cannot align trajectories.")
        return

    # Reference CA selection (indexed by reference resids)
    ref_selection = build_ca_selection(common_ref_resids)
    logger.debug(f"Reference CA selection: {ref_selection}")

    # Load reference universe once (used for all alignments)
    ref_universe = mda.Universe(ref_topology)

    # Align each trajectory to the reference
    for traj, top in zip(trajectories, topologies):

        logger.info(f"Aligning trajectory '{Path(traj).name}' with topology '{Path(top).name}'...")

        # Map common reference resids to residue IDs in this topology
        mapper = PDBTopologyMapper(ref_topology, top)
        target_resids = [mapper.map_residue(r) for r in common_ref_resids]
        target_resids = [r for r in target_resids if r is not None]

        if not target_resids:
            logger.error(f"No mappable residues found for topology '{Path(top).name}'. Skipping.")
            continue

        target_selection = build_ca_selection(target_resids)
        logger.debug(f"Mobile CA selection for '{Path(top).name}': {target_selection}")

        # Load mobile universe
        mobile_universe = mda.Universe(top, traj)

        # Define output paths
        output_traj = os.path.join(output_folder, Path(traj).name)
        output_top = os.path.join(output_folder, Path(top).stem + ".pdb")

        # Align entire trajectory to the reference
        mda_align.AlignTraj(
            mobile = mobile_universe,
            reference = ref_universe,
            select = {'mobile': target_selection, 'reference': ref_selection},
            filename = output_traj,
            match_atoms = False
        ).run()

        logger.info(f"Aligned trajectory saved to: {output_traj}")

        # Write the first frame of the aligned trajectory as the aligned topology
        aligned_universe = mda.Universe(top, output_traj)
        aligned_universe.trajectory[0]
        aligned_universe.atoms.write(output_top)

        logger.info(f"Aligned topology saved to: {output_top}")

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Analyze geometry): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return

def set_logger(verbose: bool, log_path: str):
    """
    Set up logging for Deep Cartograph.

    Parameters
    ----------
    verbose : bool
        If True, log at DEBUG level. Otherwise, log at INFO level.

    log_path : str
        Path to the log file.

    Raises
    ------
    FileNotFoundError
        If the logging configuration files in `log_config/` are missing.
    """
    # Issue warning if logging is already configured
    if logging.getLogger().hasHandlers():
        logging.warning("Logging has already been configured in the root logger. This may lead to unexpected behavior.")
    
    # Get the path to this file
    file_path = Path(os.path.abspath(__file__))

    # Get the path to the package
    tool_path = file_path.parent
    all_tools_path = tool_path.parent
    package_path = all_tools_path.parent

    info_config_path = os.path.join(package_path, "log_config/info_configuration.ini")
    debug_config_path = os.path.join(package_path, "log_config/debug_configuration.ini")
    
    # Check the existence of the configuration files
    if not os.path.exists(info_config_path):
        raise FileNotFoundError(f"Configuration file not found: {info_config_path}")
    if not os.path.exists(debug_config_path):
        raise FileNotFoundError(f"Configuration file not found: {debug_config_path}")
    
    # Pass the log_path to the fileConfig using the 'defaults' parameter
    config_path = debug_config_path if verbose else info_config_path
    logging.config.fileConfig(
        config_path,
        defaults={'log_path': log_path},
        disable_existing_loggers=True
    )

    logger = logging.getLogger("deep_cartograph")
    logger.info("Deep Cartograph: package for analyzing MD simulations using collective variables.")

########
# MAIN #
########

def main():
    """Entry point of the align trajectories command: read the arguments, then run the tool."""
    
    import argparse

    parser = argparse.ArgumentParser("Deep Cartograph: Align Trajectories", description="Align trajectories using MDAnalysis.")

    parser.add_argument(
        '-traj_data', dest='trajectory_data', required=False, nargs='+',
        help=(
            "List of trajectory paths or path to folder with trajectories to align. "
            "These trajectories will be aligned to the reference topology or to the first topology in the list. "
            "Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd."
        )
    )

    parser.add_argument(
        '-top_data', dest='topology_data', required=False, nargs='+',
        help=(
            "List of topology paths or path to folder with topologies for the trajectories. "
            "If a folder is provided, each topology should have the same name as the "
            "corresponding trajectory in -traj_data. If a single topology file is provided, it will be used for all trajectories. "
            "Accepted format: .pdb."
        )
    )

    parser.add_argument(
        '-ref_top', dest='reference_topology', required=False,
        help=(
            "Path to reference topology file (.pdb) to align the trajectories and topologies. "
            "Defaults to the first topology in topology_data. Accepted format: .pdb."
        )
    )

    parser.add_argument('-output', dest='output_folder', help="Path to the output folder", required=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Set the logging level to DEBUG", default=False)

    args = parser.parse_args()

    # Determine output folder
    output_folder = args.output_folder if args.output_folder else 'align_trajectories'
    os.makedirs(output_folder, exist_ok=True)

    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Run Align Trajectories tool
    align_trajectories(
        trajectory_data=args.trajectory_data,
        topology_data=args.topology_data,
        ref_topology=args.reference_topology,
        output_folder = output_folder)

if __name__ == "__main__":

    main()