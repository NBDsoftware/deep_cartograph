# Deep Cartograph workflow

Map trajectories onto collective variables: featurize, filter, train CVs, project and cluster in one command.

## Description

`deep_carto` runs the whole Deep Cartograph analysis in one go. You give it your MD
[trajectories and topologies](https://nbdsoftware.github.io/deep_cartograph/concepts.html#trajectories-and-topologies)
and a configuration file. It returns one or more
[collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables)
(CVs: a few numbers that summarize the conformational state of the selected region), 2D maps of your
simulations in the space of these CVs, representative conformations, and ready-to-adapt PLUMED input files to enhance the sampling of the CVs.

The steps run in this order. Each one is also available as a separate tool:

1. [`analyze_geometry`](https://nbdsoftware.github.io/deep_cartograph/tools/analyze_geometry.html):
   RMSD, RMSF and dRMSD plots of the input trajectories, as a quick sanity check.
2. [`traj_augmentation`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_augmentation.html):
   adds interpolated frames to the *seed* trajectories (`-seed_traj_data`), if any, and adds the
   result to the training data.
3. [`compute_features`](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html):
   measures [features](https://nbdsoftware.github.io/deep_cartograph/concepts.html#features)
   (distances, torsion angles, ...) on every frame, using PLUMED. Only the features that exist in all
   the topologies are kept, so that all trajectories are described in the same way.
4. [`filter_features`](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html):
   keeps only the informative features.
5. [`train_colvars`](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html):
   builds each requested CV from the filtered features, projects the training trajectories onto
   it, plots the
   [free energy surface](https://nbdsoftware.github.io/deep_cartograph/concepts.html#free-energy-surface)
   (a map of the most and least visited regions) and writes the PLUMED input files.
6. [`traj_projection`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_projection.html):
   projects the *supplementary* trajectories (`-sup_traj_data`), if any, onto the trained CVs.
7. [`traj_cluster`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_cluster.html):
   [clusters](https://nbdsoftware.github.io/deep_cartograph/concepts.html#clustering) the frames in
   the space of each CV (groups similar frames into states) and saves representative structures.

Use `deep_carto` for a new system or a new set of simulations. Use the individual tools when you
only need one step, or want to re-run one step with different settings.

## Usage

### Command line

```bash
conda activate deep_cartograph
deep_carto -conf config.yml -traj_data trajectories/ -top_data topologies/ -out output/
```

Compute only PCA and TICA with two dimensions, and project a set of experimental structures onto the CVs:

```bash
deep_carto -conf config.yml -traj_data trajectories/ -top_data topologies/ \
           -sup_traj_data experimental/ -sup_top_data experimental/ \
           -cvs pca tica -dim 2 -out output/
```

Full `deep_carto -h` output:

```
usage: Deep Cartograph [-h] -conf CONFIGURATION_PATH
                       [-traj_data TRAJECTORY_DATA [TRAJECTORY_DATA ...]]
                       [-top_data TOPOLOGY_DATA [TOPOLOGY_DATA ...]]
                       [-val_traj_data VALIDATION_TRAJECTORY_DATA [VALIDATION_TRAJECTORY_DATA ...]]
                       [-val_top_data VALIDATION_TOPOLOGY_DATA [VALIDATION_TOPOLOGY_DATA ...]]
                       [-seed_traj_data SEED_TRAJECTORY_DATA [SEED_TRAJECTORY_DATA ...]]
                       [-seed_top_data SEED_TOPOLOGY_DATA [SEED_TOPOLOGY_DATA ...]]
                       [-sup_traj_data SUPPLEMENTARY_TRAJ_DATA [SUPPLEMENTARY_TRAJ_DATA ...]]
                       [-sup_top_data SUPPLEMENTARY_TOP_DATA [SUPPLEMENTARY_TOP_DATA ...]]
                       [-ref_top REFERENCE_TOPOLOGY]
                       [-waypoints_data WAYPOINTS_DATA [WAYPOINTS_DATA ...]] [-restart]
                       [-dim DIMENSION] [-cvs CVS [CVS ...]] [-n_models N_MODELS]
                       [-out OUTPUT_FOLDER] [-v]

Map trajectories onto Collective Variables.

options:
  -h, --help            show this help message and exit
  -conf CONFIGURATION_PATH, -configuration CONFIGURATION_PATH
                        Path to configuration file (.yml).
  -traj_data TRAJECTORY_DATA [TRAJECTORY_DATA ...]
                        List of trajectory paths or path to folder with trajectories with data to
                        train CVs. These trajectories will not be modified before using them to
                        train CVs. Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd.
  -top_data TOPOLOGY_DATA [TOPOLOGY_DATA ...]
                        List of topology paths or path to folder with topologies for the
                        trajectories. If a folder is provided, each topology should have the same
                        name as the corresponding trajectory in -traj_data. If a single topology
                        file is provided, it will be used for all trajectories. Accepted format:
                        .pdb.
  -val_traj_data VALIDATION_TRAJECTORY_DATA [VALIDATION_TRAJECTORY_DATA ...]
                        List of trajectory paths or path to folder with trajectories with data to
                        validate CVs during training. Accepted formats: .xtc .dcd .pdb .xyz .gro
                        .trr .crd.
  -val_top_data VALIDATION_TOPOLOGY_DATA [VALIDATION_TOPOLOGY_DATA ...]
                        List of topology paths or path to folder with topologies for the
                        validation trajectories. If a folder is provided, each topology should
                        have the same name as the corresponding trajectory in -val_traj_data. If a
                        single topology file is provided, it will be used for all validation
                        trajectories. Accepted format: .pdb.
  -seed_traj_data SEED_TRAJECTORY_DATA [SEED_TRAJECTORY_DATA ...]
                        List of trajectory paths or path to folder with trajectories with data to
                        augment using the trajectory augmentation tool. These trajectories will be
                        augmented through interpolation before using them to train CVs. Accepted
                        formats: .xtc .dcd .pdb .xyz .gro .trr .crd.
  -seed_top_data SEED_TOPOLOGY_DATA [SEED_TOPOLOGY_DATA ...]
                        List of topology paths or path to folder with topologies for the seed
                        trajectories. If a folder is provided, each topology should have the same
                        name as the corresponding trajectory in -seed_traj_data. Accepted format:
                        .pdb.
  -sup_traj_data SUPPLEMENTARY_TRAJ_DATA [SUPPLEMENTARY_TRAJ_DATA ...]
                        List of supplementary trajectory paths or path to folder with
                        supplementary trajectories. Used to project onto the CV alongside the
                        training data but not used for computing CVs.
  -sup_top_data SUPPLEMENTARY_TOP_DATA [SUPPLEMENTARY_TOP_DATA ...]
                        List of supplementary topology paths or folder with supplementary
                        topologies. If a folder is provided, each topology should match the
                        corresponding supplementary trajectory in -sup_traj_data.
  -ref_top REFERENCE_TOPOLOGY
                        Path to reference topology file. Used to find features from user
                        selections. Defaults to the first topology in topology_data. Accepted
                        format: .pdb.
  -waypoints_data WAYPOINTS_DATA [WAYPOINTS_DATA ...]
                        Path to the folder containing intermediate conformations that define the
                        transition of interest. If given, features that do not change their value
                        across these structures will be filtered out.
  -restart              Restart workflow from the last finished step.
  -dim DIMENSION, -dimension DIMENSION
                        Dimension of the CV to train or compute. Overrides the configuration input
                        YML.
  -cvs CVS [CVS ...]    Collective variables to train or compute (pca, ae, vae, tica, htica,
                        deep_tica, umap). Overrides the configuration input YML.
  -n_models N_MODELS    Number of models to train as an ensemble, for the neural network CVs (ae,
                        vae, deep_tica). The training trajectories are split into n_models
                        disjoint folds and each member is trained on all folds but its own.
                        Overrides the configuration input YML.
  -out OUTPUT_FOLDER, -output OUTPUT_FOLDER
                        Path to the output folder.
  -v, -verbose          Set logging level to DEBUG.
```

### Python API

```python
from deep_cartograph.deep_carto import deep_cartograph
from deep_cartograph.modules.common import read_configuration

configuration = read_configuration("config.yml")

deep_cartograph(
    configuration=configuration,
    trajectory_data="trajectories/",
    topology_data="topologies/",
    supplementary_traj_data="experimental/",
    supplementary_top_data="experimental/",
    cvs=["pca", "tica"],
    dimension=2,
    output_folder="output",
)
```

The function returns nothing. All results are written to `output_folder`.

## Options

### Inputs

Each `*_traj_data` option accepts one or more trajectory files, or a folder with trajectories. Each
`*_top_data` option accepts one topology used for all the trajectories of that group, or several
topologies (or a folder), each with the same file name as its trajectory. You need at least
`-traj_data` or `-seed_traj_data`.

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | Configuration file (`.yml`). See the Configuration section below. |
| `-traj_data` | — | Trajectories used to train the CVs. Formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`. |
| `-top_data` | — | Topologies (`.pdb`) for `-traj_data`. |
| `-val_traj_data` | — | Validation trajectories. They are used to check the training of the neural-network CVs, not to train them. |
| `-val_top_data` | — | Topologies (`.pdb`) for `-val_traj_data`. |
| `-seed_traj_data` | — | Seed trajectories. New frames are interpolated between their frames, and the result is added to the training data. Useful when you only have a few structures, for example a short path between two states. |
| `-seed_top_data` | — | Topologies (`.pdb`) for `-seed_traj_data`. |
| `-sup_traj_data` | — | Supplementary trajectories or structures, for example experimental structures. They are projected onto the CVs but not used to train them. They are also used as references in the RMSD and dRMSD analysis. |
| `-sup_top_data` | — | Topologies (`.pdb`) for `-sup_traj_data`. |
| `-ref_top` | first topology in `-top_data` | Topology (`.pdb`) used to turn the atom selections of the configuration into features. |
| `-waypoints_data` | — | Structures, or a folder with structures, of intermediate states along the transition you are interested in. Features that do not change across these structures are removed. |
| `-out`, `-output` | `deep_cartograph` | Output folder. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-restart` | off | Continue in an existing output folder, reusing the results that are already there. Without it, a new folder is created (`output_1`, `output_2`, ...) if the output folder exists. |
| `-dim`, `-dimension` | from configuration | Number of dimensions of each CV (usually 1 to 3). Overrides the configuration. |
| `-cvs` | from configuration | CVs to build: `pca`, `ae`, `vae`, `tica`, `htica`, `deep_tica`, `umap`. See [types of collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#types-of-collective-variables). Overrides the configuration. |
| `-n_models` | from configuration (1) | Number of models to train for each neural-network CV (`ae`, `vae`, `deep_tica`). Each model is trained on a different part of the training trajectories, which helps you judge how robust the CV is. The projection and clustering steps use the first model. |
| `-v`, `-verbose` | off | Write detailed (debug) messages to the log. |

## Configuration

The workflow reads a YAML configuration file with one section per step. Each section takes the same
options as the configuration of the corresponding tool, so see the tool pages for details. Sections
you leave out use their default values. The default configuration is in
[`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/default_config.yml)
and it is validated against the schema in
[`yaml_schemas/deep_cartograph.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/deep_cartograph.py).

| Section | Controls |
|---------|----------|
| `analyze_geometry` | Which RMSD, RMSF and dRMSD plots to make, and the time between frames. See [`analyze_geometry`](https://nbdsoftware.github.io/deep_cartograph/tools/analyze_geometry.html). |
| `traj_augmentation` | How many frames to create from the seed trajectories, and how. See [`traj_augmentation`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_augmentation.html). |
| `compute_features` | Which features to compute, which frames to use (`plumed_settings.traj_stride`) and where PLUMED is installed (`plumed_environment`). See [`compute_features`](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html). |
| `filter_features` | How strictly to filter the features. See [`filter_features`](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html). |
| `train_colvars` | Which CVs to build, their dimension, training settings and plots. See [`train_colvars`](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html). |
| `traj_projection` | Plots of the supplementary trajectories on the CVs. See [`traj_projection`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_projection.html). |
| `traj_cluster` | Whether to cluster, which clustering method to use and which structures to save. See [`traj_cluster`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_cluster.html). |

## Output

Each step writes to its own subfolder. Folders for optional inputs only appear if you gave them.

```text
output/
├── deep_cartograph.log          # log of the whole run
├── configuration.yml            # full configuration used, including default values
├── analyze_geometry/            # RMSD, RMSF and dRMSD plots and CSV files
├── traj_augmentation/           # augmented seed trajectories (with -seed_traj_data)
├── common_features/             # intermediate files used to find the features shared by all topologies
├── compute_features/            # colvars files with the features of the training trajectories
├── compute_val_features/        # colvars files of the validation trajectories (with -val_traj_data)
├── compute_ref_features/        # colvars files of the supplementary trajectories (with -sup_traj_data)
├── compute_waypoint_features/   # colvars files of the waypoint structures (with -waypoints_data)
├── filter_features/             # list of the features kept after filtering
├── train_colvars/               # one folder per CV: trained model, projected trajectories, FES plots, PLUMED inputs
├── traj_projection/             # supplementary trajectories projected onto each CV (with -sup_traj_data)
└── traj_cluster/                # one folder per CV: cluster labels, plots and representative structures
```

The features are stored in
[colvars files](https://nbdsoftware.github.io/deep_cartograph/concepts.html#colvars-files):
plain-text tables written by PLUMED, with one row per frame and one column per feature.

## Recommendations


**Choosing the CVs**: if you are unsure, start with `-cvs pca tica`, which are fast, and add a
neural-network CV (`ae`, `vae` or `deep_tica`) once you know what the simpler maps show. See
[types of collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#types-of-collective-variables).

**Large data sets**: use `compute_features.plumed_settings.traj_stride` to keep only one of every
*n* frames, and reduce the number of features with the strides of the feature groups. This makes
every step faster.

**Resuming a run**: if a run stops halfway, run the same command again with `-restart` and the same
`-out` folder. The augmentation, feature computation and filtering steps reuse their existing results, so the
slow PLUMED calculations are not repeated. The training, projection and clustering steps run again.
If you change the settings of a step, use a new output folder instead, so that old results are not
reused.

**PLUMED interface**: the resulting Deep Learning CVs can be deployed for
[enhancing sampling](https://nbdsoftware.github.io/deep_cartograph/concepts.html#enhanced-sampling-with-plumed)
with the [PLUMED](https://www.plumed.org/) package via the
[pytorch](https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html) interface,
available since version 2.9.

## Limitations

- **Topology format.** Topologies must be PDB files.
- **PLUMED is required.** Features are computed with PLUMED, which must be installed and set in
  `compute_features.plumed_environment`.
- **UMAP in PLUMED.** UMAP CVs can be used for analysis, but not in PLUMED for enhanced sampling.
