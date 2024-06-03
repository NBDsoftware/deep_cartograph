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

def extract_clusters_from_traj(trajectory_path: str, topology_path: str, traj_name: str, traj_df: pd.DataFrame, centroids_df, metastable_df, output_folder: str, just_minima: bool = False):
    """
    Extract subset of frames for each cluster and save them in xtc format.
    Extract centroid frame for each cluster and save them in pdb format.
    Extract metastable frames for each cluster and save them in xtc format.

    Input
    -----
        trajectory_path     (str): path to the trajectory file.
        topology_path       (str): path to the topology file.
        traj_name           (str): name of the trajectory.
        traj_df       (DataFrame): DataFrame containing the frames pertaining to each cluster.
        centroids_df  (DataFrame): DataFrame containing the centroid frame pertaining to each cluster.
        metastable_df (DataFrame): DataFrame containing the metastable frame pertaining to each cluster.
        output_folder       (str): path to the output folder.
    """

    # Create ensembles and states folders in output folder
    os.makedirs(os.path.join(output_folder, 'ensembles'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'states'), exist_ok=True)
    
    # Find the different clusters from the trajectory
    clusters = np.unique(traj_df['cluster_label'])

    # Extract frames for each cluster
    for cluster in clusters:

        # Select frames
        cluster_frames = traj_df[traj_df['cluster_label'] == cluster]['frame']
        centroid_frame = centroids_df[centroids_df['cluster_label'] == cluster]['frame']

        # Create file name
        cluster_traj_name = f"{traj_name}_ensemble_{cluster}.xtc"
        centroid_name = f"{traj_name}_ensemble_{cluster}_centroid.pdb"

        # Create paths
        cluster_traj_path = os.path.join(output_folder, 'ensembles', cluster_traj_name)
        centroid_path = os.path.join(output_folder, 'ensembles', centroid_name)

        # Extract frames 
        if not just_minima:
            extract_frames(trajectory_path, topology_path, cluster_frames, cluster_traj_path)
            extract_frames(trajectory_path, topology_path, centroid_frame, centroid_path, file_format='PDB')

        if metastable_df is not None:
            # Select metastable frames
            cluster_metastable_frames = metastable_df[metastable_df['cluster_label'] == cluster]
        else:
            cluster_metastable_frames = []

        # If there are any metastable frames in this cluster, extract them
        if len(cluster_metastable_frames) > 0:

            # Find all unique states in the cluster
            cluster_states = np.unique(cluster_metastable_frames['state'])
            
            # For each state, extract metastable frames
            for state in cluster_states:

                # Extract state frames
                state_name = f"{traj_name}_ensemble_{cluster}_state_{state}.xtc"
                state_path = os.path.join(output_folder, 'states', state_name)
                state_frames = cluster_metastable_frames[cluster_metastable_frames['state'] == state]['frame']
                extract_frames(trajectory_path, topology_path, state_frames, state_path)

                # Extract state topology
                state_topology_name = f"{traj_name}_ensemble_{cluster}_state_{state}.pdb"
                state_topology_path = os.path.join(output_folder, 'states', state_topology_name)
                extract_frames(trajectory_path, topology_path, [state_frames.iloc[0]], state_topology_path, file_format='PDB')


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