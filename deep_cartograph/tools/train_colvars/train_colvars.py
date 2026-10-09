"""
Train colvars tool: trains collective variables (CVs) from colvars files,
mainly with the mlcolvar library.
"""
# Import modules
import os
import time
import argparse
import logging.config
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional

from deep_cartograph.tools.train_colvars.train_colvars_workflow import TrainColvarsWorkflow
from deep_cartograph.modules.common import (
    get_unique_path, 
    read_configuration,
    read_features_list
)

########
# TOOL #
########

def train_colvars(
    configuration: Dict,
    train_colvars_paths: Union[str, List[str]],
    train_topologies: Optional[List[str]] = None,
    trajectory_names: Optional[List[str]] = None,
    val_colvars_paths: Optional[Union[str, List[str]]] = None,
    val_topologies: Optional[List[str]] = None,
    sup_topologies: Optional[List[str]] = None,
    sup_traj_names: Optional[List[str]] = None, 
    waypoint_structures: Optional[List[str]] = None,
    reference_topology: Optional[str] = None,
    features_list: Optional[List[str]] = None,
    dimension: Optional[int] = None,
    cvs: Optional[List[Literal['pca', 'ae', 'vae', 'tica', 'htica', 'deep_tica']]] = None,
    n_models: Optional[int] = None,
    frames_per_sample: Optional[int] = 1,
    output_folder: str = 'train_colvars'
) -> Dict[str, List[str]]:
    """
    Train collective variables (CVs) from colvars files.

    For each CV, the model is trained, the Free Energy Surface (FES) along the CV is
    computed from the training data, a sensitivity analysis shows which input features
    matter most, the training trajectories are projected onto the CV and PLUMED input
    files are written (also for the supplementary topologies, if given).

    Supported collective variables (CVs):
        - pca (Principal Component Analysis)
        - ae (Autoencoder)
        - vae (Variational Autoencoder)
        - tica (Time-lagged Independent Component Analysis)
        - htica (Hierarchical Time-lagged Independent Component Analysis)
        - deep_tica (Deep Time-lagged Independent Component Analysis)
        - umap (Uniform Manifold Approximation and Projection)

    Parameters
    ----------
    configuration : Dict
        Configuration dictionary (see `default_config.yml` for more information).

    train_colvars_paths : str or List[str]
        Path or list of paths to colvars files with the training data (features for each frame).

    train_topologies : Optional[List[str]], default=None
        Topology files of the training trajectories (same order as `train_colvars_paths`).

    trajectory_names : Optional[List[str]], default=None
        Names of the trajectories behind the colvars files, used to name the output folders.
        If `None`, the colvars file names are used.

    val_colvars_paths : Optional[Union[str, List[str]]], default=None
        Path or list of paths to colvars files with the validation data.

    val_topologies : Optional[List[str]], default=None
        Topology files of the validation trajectories (same order as `val_colvars_paths`).

    sup_topologies : Optional[List[str]], default=None
        Topologies of supplementary systems. PLUMED input files are written for each of them.

    sup_traj_names : Optional[List[str]], default=None
        Names of the supplementary systems, used to name their output folders.
        If `None`, the topology file names are used.

    waypoint_structures : Optional[List[str]], default=None
        Structure files (e.g. PDB) of intermediate states of the transition. Used to add an
        experimental RMSD restraint to the PLUMED inputs (see `add_rmsd_restraint` in the
        `bias` section of the configuration).

    reference_topology : Optional[str], default=None
        Reference topology used to match feature names across topologies.
        If `None`, the first training topology is used.

    features_list : Optional[List[str]], default=None
        Features to use for training.
        If `None`, all features except `*labels`, `time`, `*bias`, and `*walker` are used.

    dimension : Optional[int], default=None
        Dimension of the CVs. If `None`, the value in the configuration is used.

    cvs : Optional[List[str]], default=None
        CVs to train or compute (see the list above). If `None`, the ones in the configuration are used.

    n_models : Optional[int], default=None
        Number of models to train as an ensemble, for the neural network CVs ('ae', 'vae', 'deep_tica').
        The training trajectories are split into `n_models` groups (folds) and model i is trained on
        all folds except fold i. Each model is saved in its own '{cv_name}_{i}' folder.
        If `None`, the value in the configuration is used (`training.general.num_models`, default 1).

    frames_per_sample : Optional[int], default=1
        Number of trajectory frames between two consecutive rows of the colvars files
        (the stride used when computing the features). Used to label the frames of the projected data.

    output_folder : str, default='train_colvars'
        Path to the output folder.

    Returns
    -------
    Dict[str, Dict]
        A dictionary keyed by CV name. For each CV: 'output_folder', 'model_path' and
        'traj_paths' (projected training trajectories, CSV files) refer to the first ensemble
        member, and 'ensemble_output_folders' / 'ensemble_model_paths' list every member.
    """

    logger = logging.getLogger("deep_cartograph")

    # Title
    logger.info("================================")
    logger.info("Training of Collective Variables")
    logger.info("================================")
    logger.info("Training of collective variables using the mlcolvar library.")

    # Start timer
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    if isinstance(train_colvars_paths, str):
        train_colvars_paths = [train_colvars_paths]
    
    # Create a TrainColvarsWorkflow object 
    workflow = TrainColvarsWorkflow(
        configuration=configuration,
        train_colvars_paths=train_colvars_paths,
        train_topology_paths=train_topologies,
        trajectory_names=trajectory_names,
        val_colvars_paths=val_colvars_paths,
        val_topology_paths=val_topologies,
        sup_topology_paths=sup_topologies,
        sup_names=sup_traj_names,
        waypoint_structures=waypoint_structures,
        ref_topology_path=reference_topology,
        features_list=features_list,
        cv_dimension=dimension,
        cvs=cvs,
        num_models=n_models,
        frames_per_sample=frames_per_sample,
        output_folder=output_folder
    )

    # Run the workflow
    output_paths = workflow.run()

    # End timer
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time (Train colvars): %s', time.strftime("%H h %M min %S s", time.gmtime(elapsed_time)))

    return output_paths

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
    
def parse_arguments():
    """Parse the command-line arguments of the train colvars command."""
    parser = argparse.ArgumentParser(
        prog="Deep Cartograph:  Train Collective Variables",
        description=("Train collective variables using the mlcolvar library."
        )
    )
    
    # Required input files
    parser.add_argument(
        '-conf', '-configuration', dest='configuration_path', type=str, required=True,
        help="Path to configuration file (.yml)."
    )
    parser.add_argument(
        '-colvars', dest='train_colvars_path', type=str, required=True,
        help="Path to the input colvars file used for training the collective variables."
    )
    
    # Optional arguments
    parser.add_argument(
        '-trajectory', dest='trajectory_name', type=str, required=False,
        help=("Name of the trajectory corresponding to the colvars file. " 
              "Used to identify the origin of the samples in the colvars file."
        )
    )
    parser.add_argument(
        '-topology', dest='topology', type=str, required=False,
        help="Path to topology file of the trajectory."
    )
    parser.add_argument(
        '-reference_topology', dest='reference_topology', type=str, required=False,
        help="Path to reference topology file. If None, the first topology file is used as reference."
    )
    parser.add_argument(
        '-frames_per_sample', dest='frames_per_sample', type=int, required=False,
        help="Number of trajectory frames between two consecutive samples in the colvars file (default: 1)."
    )
    parser.add_argument(
        '-features_path', type=str, required=False,
        help="Path to a file containing the list of features that should be used (these are used if the path is given)"
    )
    parser.add_argument(
        '-dim', '-dimension', dest='dimension', type=int, required=False,
        help="Dimension of the CV to train or compute"
    )
    parser.add_argument(
        '-cvs', nargs='+', required=False,
        help="Collective variables to train or compute (pca, ae, vae, tica, htica, deep_tica, umap)"
    )
    parser.add_argument(
        '-n_models', dest='n_models', type=int, required=False,
        help=("Number of models to train as an ensemble, for the neural network CVs (ae, vae, deep_tica). "
              "The training trajectories are split into n_models disjoint folds and each member is trained "
              "on all folds but its own. Overrides the configuration input YML.")
    )
    parser.add_argument(
        '-out', '-output', dest='output_folder', required=False,
        help="Path to the output folder"
    )
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', required=False,
        help="Set the logging level to DEBUG."
    )

    return parser.parse_args()

########
# MAIN #
########

def main():
    """Entry point of the train colvars command: read the arguments and configuration, then run the tool."""

    args = parse_arguments()

    # Create new output folder
    output_folder = args.output_folder if args.output_folder else 'train_colvars'
    output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set logger
    log_path = os.path.join(output_folder, 'deep_cartograph.log')
    set_logger(verbose=args.verbose, log_path=log_path)

    # Read configuration
    configuration = read_configuration(args.configuration_path)

    # Read features to use
    features_list = read_features_list(args.features_path)

    # Trajectory names should be list or None - see train_colvars API
    trajectory_names = None
    if args.trajectory:
        trajectory_names = [args.trajectory_name]

    # Topologies should be list or None - see train_colvars API
    train_topologies = None
    if args.topology:
        train_topologies = [args.topology]

    # Run Train Colvars tool
    train_colvars(
        configuration = configuration,
        train_colvars_paths = args.colvars_path,
        train_topologies = train_topologies,
        trajectory_names = trajectory_names,
        reference_topology = args.reference_topology,
        features_list = features_list,
        dimension = args.dimension,
        cvs = args.cvs,
        n_models = args.n_models,
        frames_per_sample = args.frames_per_sample,
        output_folder = output_folder)
    
if __name__ == "__main__":

    main()
    