# Train collective variables

Find a few variables that summarize how your system changes, and prepare them for analysis and enhanced sampling.

## Description

A protein in a trajectory can be described by hundreds of [features](https://nbdsoftware.github.io/deep_cartograph/concepts.html#features) (distances, angles, dihedrals...). That is too many to look at or to bias. This tool compresses them into one, two or a few numbers per frame: the [collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables) (CVs). Each type of CV has its own idea of what is important, for example the largest or the slowest motions (see [types of collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#types-of-collective-variables)). You can train several types in one run and compare them.

The input is a [colvars file](https://nbdsoftware.github.io/deep_cartograph/concepts.html#colvars-files): a table with the value of each feature in each frame. For each CV type you ask for, the tool:

- trains the CV and saves it as a model file;
- projects the training trajectory onto the CV (the CV values of each frame);
- plots the [free energy surface](https://nbdsoftware.github.io/deep_cartograph/concepts.html#free-energy-surface) (FES) along the CV, which shows the stable states as low-energy basins;
- ranks the features by how much they affect the CV, so you can see which parts of the protein drive it;
- writes ready-to-use PLUMED input files to compute the CV, or to bias it in an [enhanced sampling](https://nbdsoftware.github.io/deep_cartograph/concepts.html#enhanced-sampling-with-plumed) simulation.

In the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html), this tool comes after [compute_features](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html) (which writes the colvars file) and usually after [filter_features](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html) (which picks the informative features). Its results feed [traj_projection](https://nbdsoftware.github.io/deep_cartograph/tools/traj_projection.html) (to project other trajectories onto the trained CVs) and [traj_cluster](https://nbdsoftware.github.io/deep_cartograph/tools/traj_cluster.html) (to group frames into states).

## Usage

### Command line

```bash
conda activate deep_cartograph
train_colvars -h
```

Train a 2D PCA and a 2D TICA on the features chosen by `filter_features`, and write PLUMED inputs for the system in `protein.pdb`:

```bash
train_colvars -conf config.yml -colvars colvars.dat -topology protein.pdb \
              -trajectory my_traj -features_path filtered_features.txt \
              -cvs pca tica -dim 2 -out train_colvars
```

### Python API

```python
from deep_cartograph.tools.train_colvars import train_colvars
from deep_cartograph.modules.common import read_configuration, read_features_list

results = train_colvars(
    configuration=read_configuration("config.yml"),
    train_colvars_paths=["colvars_rep1.dat", "colvars_rep2.dat"],
    train_topologies=["protein.pdb", "protein.pdb"],
    trajectory_names=["rep1", "rep2"],
    features_list=read_features_list("filtered_features.txt"),
    cvs=["pca", "deep_tica"],
    dimension=2,
    output_folder="train_colvars",
)

# One entry per CV type
results["pca"]["model_path"]   # 'train_colvars/pca/model.zip'
results["pca"]["traj_paths"]   # projected trajectories (CSV), one per colvars file
```

The Python API also accepts several training trajectories, validation data (`val_colvars_paths`), and extra topologies for which PLUMED inputs are written (`sup_topologies`). See the function docstring for the full list.

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | YAML configuration file (see Configuration below). |
| `-colvars` | required | Colvars file with the feature values used to train the CVs. |
| `-topology` | — | Topology of the trajectory behind the colvars file. Needed to write the PLUMED input files and to map feature importance onto the structure. |
| `-reference_topology` | — | Topology used to name the features when trajectories come from different topologies. If not given, the first topology is used. |
| `-features_path` | — | Text file with the features to use, one per line (e.g. the output of `filter_features`). If not given, all features in the colvars file are used. |
| `-out`, `-output` | — | Output folder. If not given, `train_colvars` is used. If the folder exists, a new one with a number suffix is created. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-trajectory` | — | Name for the trajectory, used to name its output folder. If not given, the colvars file name is used. |
| `-frames_per_sample` | — | Number of trajectory frames between two rows of the colvars file (the stride used when computing the features). If not given, 1 is used. |
| `-dim`, `-dimension` | — | Number of CV components (dimensions) to compute. Overrides the value in the configuration. |
| `-cvs` | — | CV types to train, separated by spaces: `pca`, `tica`, `htica`, `ae`, `vae`, `deep_tica`, `umap`. Overrides the list in the configuration. |
| `-n_models` | — | Train this many copies of each neural network CV (`ae`, `vae`, `deep_tica`), each one leaving out part of the training trajectories. Useful to check whether the result is robust. Overrides the configuration. |
| `-v`, `--verbose` | off | Write more detailed messages to the log. |

## Configuration

The configuration file sets which CVs to train and how; any option you leave out takes the default shown below.

| Option | Default | Description |
|--------|---------|-------------|
| `cvs` | all seven types | List of CV types to train (overridden by `-cvs`). |
| `common.dimension` | `2` | Number of CV components (overridden by `-dim`). |
| `common.lag_time` | `1` | For `tica`, `htica` and `deep_tica`: time between the two frames compared, in rows of the colvars file. |
| `common.features_normalization` | `null` | Scale the features before training: `mean_std`, `min_max_range1`, `min_max_range2` or `null` (no scaling). `mean_std` is a good choice when features have different units. |
| `common.input_colvars.stride` | `1` | Use only every n-th row of the colvars file. Also `start` and `stop` to use part of it. |
| `common.training.general.max_epochs` | `1000` | Maximum number of training rounds for the neural network CVs. Training stops earlier if it stops improving. |
| `common.training.general.num_models` | `1` | Number of copies of each neural network CV to train (overridden by `-n_models`). |
| `common.bias.method` | `opes_metad` | Enhanced sampling method written to the biased PLUMED input: `opes_metad`, `opes_metad_explore`, `opes_expanded` or `wt_metadynamics` (well-tempered metadynamics). |
| `common.bias.args.temperature` | `300.0` | Simulation temperature (K) used in the biased PLUMED input. |
| `figures.fes.compute` | `true` | Compute and plot the free energy surface. |
| `figures.traj_projection.plot` | `true` | Plot each trajectory in the CV space, colored by frame (2D CVs only). |

The other bias settings (`sigma`, `pace`, `barrier`, `height`, `bias_factor`, and the grid range
`grid_min`/`grid_max`) are under `common.bias.args`; their defaults are in the files linked below.

Settings under `common` apply to all CV types. To change a setting for one type only, add a section named after it with the same keys, for example `ae: {training: {general: {max_epochs: 5000}}}`.

The example configuration is in [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/train_colvars/default_config.yml). All options and their defaults are in the schema [`yaml_schemas/train_colvars.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/train_colvars.py). They include the network size, learning rate and the settings of each CV type.

## Output

```text
train_colvars/
├── configuration.yml                  # full configuration used, with all defaults filled in
├── deep_cartograph.log                # log of the run (command line only)
└── pca/                               # one folder per CV type (ae_0/, ae_1/... when -n_models > 1)
    ├── model.zip                      # trained CV; give it to traj_projection
    ├── traj_data/
    │   └── my_traj/                   # one folder per training trajectory
    │       ├── projected_trajectory.csv   # CV values of each frame (one column per component)
    │       ├── trajectory.png         # trajectory in the CV space, colored by frame (2D CVs only)
    │       ├── fes/
    │       │   ├── fes_pca_1/fes.png      # free energy along component 1 (one folder per component)
    │       │   └── fes_pca_1_2/fes.png    # free energy map of components 1 and 2 (one folder per pair)
    │       └── plumed_inputs/
    │           ├── plumed_pca_unbiased.zip   # PLUMED files to compute the CV along a trajectory or plain MD
    │           └── plumed_pca_biased.zip     # PLUMED files to run enhanced sampling along the CV
    ├── sensitivity_analysis/          # which features matter most for the CV
    │   ├── sensitivity_analysis.csv   # importance of each feature
    │   ├── top_features_barh.png      # bar plot of the most important features
    │   └── sensitivity_structure.pdb  # structure with importance in the B-factor column (color it in PyMOL/VMD)
    └── training/                      # training curves and scores (neural network CVs)
```

- The FES folders also contain the free energy values as `.npy` files, unless `figures.fes.save` is `false`.
- For the linear CVs (`pca`, `tica`, `htica`), `sensitivity_analysis/` has one subfolder per component.
- The PLUMED zip files contain the PLUMED input, a PDB of the system and, for neural network CVs, the model file PLUMED needs.
- PLUMED files are written only when a topology is given. `umap` writes no PLUMED files and has no sensitivity analysis.

## Recommendations

**Which CV type should I use?**

- [**PCA**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#pca) (`pca`): the directions in which the structure changes the most. Fast, simple and reproducible. A good first try.
- [**TICA**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#tica) (`tica`): the slowest motions, which are often the transitions between states. The trajectory must be in time order; set `common.lag_time`.
- [**HTICA**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#htica) (`htica`): like TICA, but splits the features into groups first. Use it when you have very many features and TICA runs out of memory.
- [**Autoencoder**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#autoencoder-ae) (`ae`): a neural network that learns a compressed description from which the features can be rebuilt. Can capture curved, non-linear changes that PCA misses. Slower, and results change a bit between runs.
- [**Variational autoencoder**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#variational-autoencoder-vae) (`vae`): like the autoencoder, but tends to give a smoother, more evenly spread map. Needs more tuning.
- [**DeepTICA**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#deeptica) (`deep_tica`): the non-linear version of TICA, for the slowest motions. Needs a time-ordered trajectory and enough transitions between states.
- [**UMAP**](https://nbdsoftware.github.io/deep_cartograph/concepts.html#umap) (`umap`): a map that keeps similar frames close together, good for seeing distinct states. For analysis only: it cannot be used in PLUMED.

A practical approach is to train `pca` and `tica` first, then add a neural network CV if the free energy surface does not separate the states you care about.

- **Start with 2 dimensions.** 2D CVs give the most readable plots (FES maps and trajectory plots). For enhanced sampling, 1 or 2 dimensions is usually best.
- **Filter the features first.** Fewer, informative features give faster training and clearer CVs. Use [filter_features](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html) and pass its list with `-features_path`.
- **Match `-frames_per_sample` to the stride** used in `compute_features`, so frame numbers in the outputs match the real trajectory.
- **Check the sensitivity analysis.** It tells you which residues drive each CV. Load `sensitivity_structure.pdb` in PyMOL or VMD and color by B-factor.
- **Check the bias settings before a long run.** Review `sigma`, `pace` and `barrier` (or `height`) for your system, and widen `grid_min`/`grid_max` if you expect the simulation to go beyond the states in the training data.

## Limitations

- **UMAP cannot be used in PLUMED.** No PLUMED files are written for `umap`, so it is for analysis only.
- **Neural network CVs need PLUMED with PyTorch support.** To use `ae`, `vae` or `deep_tica` in a simulation, your PLUMED build must include the PyTorch module. `pca`, `tica` and `htica` work with a standard PLUMED build.
- **Optional packages.** All CV types except `pca` need the optional machine-learning packages installed with Deep Cartograph; if they are missing, those CVs are skipped with a warning.
- **GPU is optional.** Neural network CVs train on CPU; a GPU, if present, is used automatically and makes training faster.
- **Neural network results vary between runs.** Change the seed or use `-n_models` to see how much. Linear CVs (`pca`, `tica`, `htica`) always give the same result for the same input.
- **Coordinates as features.** If the features include atom coordinates, all trajectories must be aligned to the same reference, also when you use the PLUMED files.
