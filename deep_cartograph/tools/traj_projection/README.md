# Trajectory projection

Compute the values of already trained collective variables for new trajectories.

## Description

Once you have trained [collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables) (CVs) with [train_colvars](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html), you may want to see where other simulations fall on the same map. Examples are new replicas, a mutant, or an enhanced sampling run. This tool takes the trained models and computes the CV values of every frame of the new trajectories. This is called projecting the trajectories onto the CVs.

The new trajectories are given as [colvars files](https://nbdsoftware.github.io/deep_cartograph/concepts.html#colvars-files) (tables of [feature](https://nbdsoftware.github.io/deep_cartograph/concepts.html#features) values per frame) computed with [compute_features](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html). They must contain the same features used to train the model. If you also give the projected training trajectories, the tool draws the new trajectories on top of the [free energy surface](https://nbdsoftware.github.io/deep_cartograph/concepts.html#free-energy-surface) (FES) of the training data. This shows at a glance whether the new runs visit the same states or new ones.

In the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html), this tool comes after `train_colvars`. Its output can be clustered with [traj_cluster](https://nbdsoftware.github.io/deep_cartograph/tools/traj_cluster.html).

## Usage

### Command line

```bash
conda activate deep_cartograph
traj_projection -h
```

Project a new replica onto a PCA model trained with `train_colvars`, and plot it on the FES of the training trajectory:

```bash
traj_projection -conf config.yml \
                -colvars new_replica/colvars.dat -top protein.pdb -names new_replica \
                -models train_colvars/pca/model.zip \
                -models_traj train_colvars/pca/traj_data/my_traj/projected_trajectory.csv \
                -out traj_projection
```

The configuration file can be very short; for example, a `config.yml` with only `figures: {}` uses all the defaults.

### Python API

```python
from deep_cartograph.tools.traj_projection import traj_projection

results = traj_projection(
    configuration={},
    colvars_paths=["mutant/colvars.dat", "wild_type/colvars.dat"],
    topologies=["mutant.pdb", "wild_type.pdb"],
    trajectory_names=["mutant", "wild_type"],
    model_paths=["train_colvars/pca/model.zip", "train_colvars/tica/model.zip"],
    # For each model, the projected training trajectories used for the background FES
    model_traj_paths=[
        ["train_colvars/pca/traj_data/my_traj/projected_trajectory.csv"],
        ["train_colvars/tica/traj_data/my_traj/projected_trajectory.csv"],
    ],
    output_folder="traj_projection",
)

# One entry per CV type, one CSV per trajectory
results["pca"]["traj_paths"]  # ['traj_projection/pca/mutant/projected_trajectory.csv', ...]
```

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | YAML configuration file (see Configuration below). |
| `-colvars`, `-colvars_files` | required | One or more colvars files of the trajectories to project. |
| `-top`, `-topology` | — | Topologies of the trajectories, one per colvars file and in the same order. Needed when the trajectories come from a different topology than the training data, to match the feature names. |
| `-models`, `-cvs_models` | required | One or more trained models (`model.zip` files written by `train_colvars`). |
| `-models_traj`, `-cvs_models_traj` | required | Projected training trajectories of the model (`projected_trajectory.csv` files written by `train_colvars`). Used to draw the background FES. |
| `-out`, `-output` | — | Output folder. If not given, `traj_projection` is used. If the folder exists, a new one with a number suffix is created. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-names`, `-trajectory_names` | — | Names for the trajectories, one per colvars file. Used to name the output folders and the plot legends. |
| `-v`, `--verbose` | off | Write more detailed messages to the log. |

## Configuration

The configuration file only controls the plots; any option you leave out takes the default shown below.

| Option | Default | Description |
|--------|---------|-------------|
| `figures.fes.compute` | `true` | Compute and plot the free energy surface of the training data. |
| `figures.fes.temperature` | `300` | Temperature (K) used to compute the free energy surface. |
| `figures.fes.max_fes` | `30` | Highest free energy shown in the plots; higher values are left blank. |
| `figures.traj_projection.plot` | `true` | Plot each trajectory in the CV space, colored by frame (2D CVs only). |

There is no separate default configuration file for this tool. All options are listed in the schema [`yaml_schemas/traj_projection.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/traj_projection.py).

## Output

```text
traj_projection/
├── configuration.yml                  # full configuration used, with all defaults filled in
├── deep_cartograph.log                # log of the run (command line only)
└── pca/                               # one folder per model
    ├── new_replica/                   # one folder per projected trajectory
    │   ├── projected_trajectory.csv   # CV values of each frame (one column per component)
    │   └── trajectory.png             # trajectory in the CV space, colored by frame (2D CVs only)
    └── fes/                           # only if -models_traj is given
        ├── fes_pca_1/fes.png          # FES along component 1, with the new trajectories as histograms
        └── fes_pca_1_2/fes.png        # FES map of components 1 and 2, with the new trajectories as points
```

The FES folders also contain the free energy values as `.npy` files, unless `figures.fes.save` is `false`.

## Recommendations

- **Use the same features as in training.** Compute the colvars files of the new trajectories with the same `compute_features` settings used for the training data. Every feature the model needs must be present.
- **Give topologies for different systems.** For a mutant or a different construct, pass its topology with `-top` so the features can be matched to the right atoms.
- **Cluster the result.** Pass the `projected_trajectory.csv` files to [traj_cluster](https://nbdsoftware.github.io/deep_cartograph/tools/traj_cluster.html) to group the frames into states.

## Limitations

- **Existing results are not recomputed.** If the projected files for a model already exist in the output folder, that model is skipped. Use a new output folder to recompute.
- **One folder per CV type.** Output folders are named after the CV type, so two models of the same type (for example two PCA models) write to the same folder.
- **Frame numbers are row numbers.** The frame used in `trajectory.png` counts rows of the colvars file, not trajectory frames. If you computed the features with a stride, multiply by it.
