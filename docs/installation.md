# Installation

Requirements: `git`, `conda`. Create the `deep_cartograph` conda environment and install the package:

```bash
git clone https://github.com/NBDsoftware/deep_cartograph.git
cd deep_cartograph
conda env create -f environment_detailed.yml
conda activate deep_cartograph
pip install .
```

The `environment_detailed.yml` file was produced with `conda env export --no-builds` and should be
cross-platform compatible. If it fails to solve, create the environment from `environment.yml` instead.

This exposes the `deep_carto` workflow command and the tool commands (`align_trajectories`,
`analyze_geometry`, `compute_features`, `filter_features`, `train_colvars`, `traj_augmentation`,
`traj_cluster`, `traj_projection`).

<!-- TODO: document environment_lite.yml (no torch / mlcolvar, PCA only). -->

## GPU support

To use a GPU, create the environment on a machine that has one available. On a cluster, for example,
open an interactive session on a compute node and install from there, so that conda resolves the
GPU-enabled dependencies.

## Development install

Install the package in editable mode, so that changes in the working directory are reflected in
the environment, and use `environment_develop.yml` to get `pytest` and Jupyter:

```bash
conda env create -f environment_develop.yml
conda activate deep_cartograph
pip install -e .
pytest deep_cartograph/tests
```

## Releasing (maintainers)

Every version string in the repo (`setup.py`, `setup.cfg`, `CITATION.cff`, `docs/conf.py`) is
generated — do not edit them by hand. Instead:

1. Describe the changes under `## [Unreleased]` in `CHANGELOG.md` as you work.
2. **Actions ▸ Release (1/2) prepare ▸ Run workflow**, branch `master`, version e.g. `0.2.0`
   (no leading `v`). This rewrites the version everywhere, renames `[Unreleased]` to the new
   version, and pushes a `release/0.2.0` branch. The job summary links you to the PR.
3. Open that PR (keep the title `release: 0.2.0`) and merge it.
4. **Release (2/2) publish** then fires automatically: it tags the release commit and publishes
   the GitHub Release, using that CHANGELOG section as the notes.

It is split in two because `master` requires pull requests, so the version change cannot be pushed
to it directly. Preview a bump locally with `python scripts/bump_version.py X.Y.Z --dry-run`.
