# Import modules
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import MDAnalysis as mda


# Set logger
logger = logging.getLogger(__name__)

def extract_frames(trajectory_path: str, topology_path: str, frames: list, new_traj_path: str, file_format: str = 'XTC'):
    """ 
    Extract frames from a trajectory and save them in a new trajectory file.

    Input
    -----

        trajectory_path (str): path to the trajectory file.
        topology_path   (str): path to the topology file.
        frames          (list): list of frames to extract.
        new_traj_path   (str): path to the new trajectory file.
        file_format     (str): format of the new trajectory file.
    """

    # Try to load trajectory
    try:
        u = mda.Universe(topology_path, trajectory_path)
    except Exception as e:
        logger.error(f"Error loading trajectory {trajectory_path}. {e}")
        sys.exit(1)

    # Check if any frames were requested
    if len(frames) == 0:
        logger.warning(f"No frames requested for {new_traj_path}.")
        return
    
    # If requested format is PDB, save to temporary file (including CONECT records)
    if file_format == 'PDB':
        final_traj_path = new_traj_path
        new_traj_path = os.path.join(Path(new_traj_path).parent, "tmp.pdb")

    # Save subset of frames to new trajectory
    with mda.Writer(new_traj_path, n_atoms=u.atoms.n_atoms, format=file_format) as writer:
        for frame in frames:
            u.trajectory[frame]
            writer.write(u)

    # If requested format is PDB, remove CONECT records
    if file_format == 'PDB':
        with open(new_traj_path, 'r') as f:
            lines = f.readlines()
        with open(final_traj_path, 'w') as f:
            for line in lines:
                if not line.startswith("CONECT"):
                    f.write(line)

    return

def extract_clusters_from_traj(trajectory_path: str, topology_path: str, traj_df: pd.DataFrame, centroids_df: pd.DataFrame = None,
                               cluster_label: str = 'cluster', frame_label: str = 'frame', output_folder: str = 'clustered_traj'):
    """
    Extract all frames from the trajectory pertaining to each cluster and save them in a new trajectory files (XTC).

    This function assumes that the traj_df contains a row for each frame in the trajectory and a column with the cluster label.

    Input
    -----
        trajectory_path     (str): path to the trajectory file.
        topology_path       (str): path to the topology file.
        traj_df       (DataFrame): DataFrame containing the frames pertaining to each cluster.
        centroids_df  (DataFrame): DataFrame containing the centroids of each cluster.
        cluster_label       (str): name of the column containing the cluster label.
        frame_label         (str): name of the column containing the frame index.
        output_folder       (str): path to the output folder.
    """
    
    # Check the existence of trajectory and topology files
    if not os.path.exists(trajectory_path):
        logger.warning(f"Trajectory file {trajectory_path} not found.")
        return

    if not os.path.exists(topology_path):
        logger.warning(f"Topology file {topology_path} not found.")
        return

    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find the different clusters from the trajectory
    clusters = np.unique(traj_df[cluster_label])

    # Extract frames for each cluster
    for cluster in clusters:

        # Skip if cluster is -1 (noise cluster)
        if cluster == -1:
            continue

        # Find frames for this cluster
        cluster_frames = traj_df[traj_df[cluster_label] == cluster][frame_label]

        # Find topology frame
        if centroids_df is not None:
            # Make it the centroid frame if available
            topology_frame = [centroids_df[centroids_df[cluster_label] == cluster][frame_label].values[0]]
        else:
            # Pick the first frame from the cluster is centroids are not available
            topology_frame = [cluster_frames.values[0]]


        # Create file name
        cluster_traj_name = f"cluster_{cluster}.xtc"
        cluster_top_name = f"cluster_{cluster}.pdb"

        # Create paths
        cluster_traj_path = os.path.join(output_folder, cluster_traj_name)
        cluster_top_path = os.path.join(output_folder, cluster_top_name)

        # Extract frames 
        extract_frames(trajectory_path, topology_path, cluster_frames, cluster_traj_path)
        extract_frames(trajectory_path, topology_path, topology_frame, cluster_top_path, file_format='PDB')

def find_supported_traj(parent_path, filename = None):
    """
    Find all trajectories that comply with supported formats by MDAnalysis inside a parent folder.

    Input
    -----
        parent_path (str): path to parent folder with trajectories.
        filename    (str): (Optional) generic name of trajectories. To select a subset of trajectories.

    Output
    ------
        all_supported_trajs (list): sorted list of paths to supported trajectories.
    """

    # List supported formats (MDAnalysis supported)
    traj_formats = ['.xtc', '.dcd', '.xyz', '.arc', '.trr', '.tng', '.nc', '.netcdf', '.mdcrd', '.crd', '.h5', '.hdf5', '.lh5']

    if filename is None:
        filename = "*"  

    # Find all ensemble trajectories using generic name given in global_parameters
    all_trajectories = glob.glob(os.path.join(parent_path, filename))

    # Check if all_trajectories is None
    if all_trajectories is None or len(all_trajectories) == 0:
        logger.warning(f"No trajectories found in {parent_path}.")

    # Find all trajectories that comply with supported formats
    all_supported_trajs = [file for file in all_trajectories if Path(file).suffix in traj_formats]

    # Complain if no supported trajectories are found
    if len(all_supported_trajs) == 0:
        logger.warning(f"No supported trajectories found in {parent_path}.")

    all_supported_trajs.sort()

    return all_supported_trajs

def find_supported_top(parent_path, filename = None):
    """
    Find all topologies that comply with supported formats by MDAnalysis inside a parent folder.

    Input
    -----
        parent_path (str): path to parent folder.
        filename (str): generic name of topologies.
    
    Output
    ------
        all_supported_tops (list): sorted list of paths to supported topologies.
    """

    # List supported formats
    top_formats = ['.pdb', '.gro', '.psf', '.prmtop', '.parm7', '.top', '.itp']

    if filename is None:
        filename = "*"

    # Find all ensemble topologies using generic name given in global_parameters
    all_topologies = glob.glob(os.path.join(parent_path, filename))

    # Check if topology is None
    if all_topologies is None or len(all_topologies) == 0:
        logger.warning(f"No topology found in {parent_path}.")
    
    # Find all topologies that comply with supported formats
    all_supported_tops = [file for file in all_topologies if Path(file).suffix in top_formats]

    # Complain no supported topology is found
    if len(all_supported_tops) == 0:
        logger.warning(f"No supported topology found in {parent_path}.")
    
    # Sort them to make sure we pair the right trajectory and topology
    all_supported_tops.sort()

    return all_supported_tops

def create_pdb(structure_path, file_name):
    """
    Creates a PDB file from a structure file using MDAnalysis.
    
    Input
    -----
        structure_path (str): path to the structure file.
        file_name (str): name of the PDB file.
    """

    # Load structure
    u = mda.Universe(structure_path)

    # Save structure in PDB format
    u.atoms.write(file_name)

    return