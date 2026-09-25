# Tools

This folder contains the tools available in the Deep Cartograph project.

## Overview

Each tool can be used to perform a specific task. The tools are designed to be modular and can be used independently from command line or in combination with other tools through the provided python APIs.

## List of Tools

Each tool has its own README, which is also published on the [documentation site](https://nbdsoftware.github.io/deep_cartograph/).

- [**align_trajectories**](align_trajectories/README.md): align trajectories to a reference topology.
- [**analyze_geometry**](analyze_geometry/README.md): perform some simple geometry analysis like RMSD or RMSF.
- [**compute_features**](compute_features/README.md): compute features from a trajectory.
- [**filter_features**](filter_features/README.md): filter features based on bi-modality, standard deviation or entropy of their distribution.
- [**train_colvars**](train_colvars/README.md): compute and train different collective variables from data.
- [**traj_augmentation**](traj_augmentation/README.md): augment trajectory samples by interpolating between existing frames.
- [**traj_projection**](traj_projection/README.md): project trajectories onto collective variable space.
- [**traj_cluster**](traj_cluster/README.md): cluster trajectories based on collective variable space.
