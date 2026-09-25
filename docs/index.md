# Deep Cartograph

```{image} _static/DeepCarto_logo.png
:alt: Deep Cartograph
:width: 200px
:align: center
```

Deep Cartograph is a package to analyze and enhance molecular dynamics (MD) simulations. It trains
collective variables (CVs) from simulation data, either to analyze existing trajectories or to
enhance sampling in subsequent simulations. It uses [PLUMED](https://www.plumed.org/) to compute
features and the [mlcolvar](https://github.com/luigibonati/mlcolvar) library to train the CVs.
Developed for the [European BioExcel](http://bioexcel.eu/) project, funded by the European
Commission (EU Horizon Europe [101093290](https://cordis.europa.eu/project/id/101093290)).

Starting from trajectory and topology files, Deep Cartograph can:

1. Featurize the trajectory.
2. Filter the features.
3. Compute and train different collective variables.
4. Project and cluster the trajectory in the CV space.
5. Produce a PLUMED input file to enhance the sampling.

```{image} _static/DeepCarto.png
:alt: Deep Cartograph workflow
:width: 800px
:align: center
```

**New here?** Start with the [Installation](installation.md) guide, then run the full
[`deep_carto`](workflow/deep_carto.md) workflow.

## Workflow

| Command | Purpose |
|---------|---------|
| [`deep_carto`](workflow/deep_carto.md) | Run every step, from featurization to clustering, in one command. |

## Tools

Each step is also available as a standalone command and as a Python function.

| Command | Purpose |
|---------|---------|
| [`align_trajectories`](tools/align_trajectories.md) | Align trajectories to a reference topology. |
| [`analyze_geometry`](tools/analyze_geometry.md) | Simple geometry analysis: RMSD, RMSF and dRMSD. |
| [`traj_augmentation`](tools/traj_augmentation.md) | Augment trajectory samples by interpolating between frames. |
| [`compute_features`](tools/compute_features.md) | Compute features from a trajectory using PLUMED. |
| [`filter_features`](tools/filter_features.md) | Keep the most informative features. |
| [`train_colvars`](tools/train_colvars.md) | Train collective variables with mlcolvar. |
| [`traj_projection`](tools/traj_projection.md) | Project trajectories onto pre-trained CVs. |
| [`traj_cluster`](tools/traj_cluster.md) | Cluster trajectory frames in the CV space. |

```{toctree}
:maxdepth: 2
:hidden:

installation
workflow/deep_carto
```

```{toctree}
:caption: Tools
:maxdepth: 2
:hidden:

tools/align_trajectories
tools/analyze_geometry
tools/traj_augmentation
tools/compute_features
tools/filter_features
tools/train_colvars
tools/traj_projection
tools/traj_cluster
```

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

faq
changelog
```

## Citing

If you use Deep Cartograph, please cite it using the metadata in the
[`CITATION.cff`](https://github.com/NBDsoftware/deep_cartograph/blob/master/CITATION.cff) file
(GitHub's "Cite this repository" button generates APA and BibTeX entries from it).

```{image} _static/bioexcel_logo.png
:alt: BioExcel
:width: 250px
:target: https://bioexcel.eu/
:align: center
:class: only-light
```
```{image} _static/bioexcel_logo_white.png
:alt: BioExcel
:width: 250px
:target: https://bioexcel.eu/
:align: center
:class: only-dark
```

## Licensing

Offered under a dual-license model: free for academic and non-commercial use under
**CC BY-NC-SA 4.0**; a separate commercial license is required for for-profit use
(contact `it@nostrumbiodiscovery.com`). See the `LICENSE` file in the repository.
