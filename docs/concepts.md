# Concepts

A short, plain-language guide to the terms used across this documentation.

## Trajectories and topologies

A **trajectory** is the output of a molecular dynamics (MD) simulation: the coordinates of all atoms
saved at regular time intervals (the *frames*). A **topology** describes the system itself: which atoms
there are, their names, residues and chains. Deep Cartograph reads topologies as PDB files and
trajectories in the common formats (`.xtc`, `.dcd`, `.trr`, `.pdb`, `.gro`, ...). Each trajectory needs a
matching topology.

## Features

A **feature** is a simple geometric quantity measured on every frame of a trajectory, such as the
distance between two atoms or a backbone torsion angle. Together, the features describe the conformation of
the molecule in each frame in a way a computer can compare and learn from.

Deep Cartograph can compute these kinds of features:

- **Distances** between pairs of atoms, in nm.
- **Dihedral angles**: torsion angles defined by four atoms. They can be the backbone phi/psi
  angles, side-chain torsions, or *virtual* dihedrals. A virtual dihedral is built from four atoms that
  are not bonded, such as four consecutive C-alpha atoms, and is useful to follow the overall fold of a
  chain. Each angle is usually stored as its sine and cosine, so that -180° and 180° get the same
  value.
- **Distances to a center**: from each selected atom to the geometric center of a group of atoms.
- **Coordinates**: the x, y and z position of selected atoms, after aligning the frames.

Computing many features is easy; the hard part is knowing which of them matter. That is why Deep
Cartograph first computes a broad set of features, then [filters](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html)
out the ones that carry little information.

## Colvars files

A **colvars file** is a plain-text table written by [PLUMED](https://www.plumed.org/). It has one row
per frame and one column per feature, plus a `time` column that holds the frame number. The first
line lists the column names, for example `#! FIELDS time dist-@CA_12-@CA_45 sin-@phi_30 ...`. Values
are in PLUMED units: nm for distances and radians for angles. The tools in Deep Cartograph pass
features to each other through colvars files: [`compute_features`](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html)
writes them, and the following tools read them.

## Collective variables

A **collective variable** (CV) is a small number of values, usually one to three, that summarize the
state of the molecule in each frame. A good CV separates the states you care about, such as
open and closed, bound and unbound, or folded and unfolded, and follows the transitions between them.

Deep Cartograph learns CVs automatically from the features. Each CV is a function of the features that
can be computed for any new frame. This lets you:

- **Visualize** a long simulation as a simple 2D map.
- **Compare** different simulations in the same space.
- **Group** frames into states with [clustering](#clustering).
- **Enhance sampling** by biasing new simulations along the CV (see [below](#enhanced-sampling-with-plumed)).

## Types of collective variables

Deep Cartograph offers several methods to build a CV. They differ in what they look for in the data.
When in doubt, start with PCA, since it is fast and easy to interpret, and compare it with one of the others.

### PCA

*Principal component analysis.* It finds the directions in which the structure changes the most. The
CV is a weighted sum of the features, which makes it fast and easy to interpret. Motions that are
large but not relevant can dominate it.

### TICA

*Time-lagged independent component analysis.* It finds the **slowest** motions: the changes that take
the longest to happen, which are often the transitions between states. It needs time-ordered
trajectories, and a **lag time**, which is the time gap between the pairs of frames it compares.

### HTICA

*Hierarchical TICA.* It splits the features into groups, applies TICA to each group and then again to
the results. It is useful when the system is big and there are too many features for a single TICA.

### Autoencoder (AE)

A neural network learns to squeeze the features into a few numbers, the CV, and to rebuild the features
from them. It can capture curved, non-linear changes that PCA misses, but it needs more data and
training time.

### Variational autoencoder (VAE)

A variant of the autoencoder that produces a smoother and more regular CV space.

### DeepTICA

It combines a neural network with TICA to find slow motions that are non-linear. It is the most
powerful option for kinetics.

### UMAP

A dimensionality-reduction method that is popular for visualization. It is good at separating
clusters, but its CVs **cannot be used in PLUMED** for enhanced sampling.

## Free energy surface

The **free energy surface** (FES) is a map of how likely each region of the CV space is. Deep
Cartograph estimates it from how often the simulation visited each region: frequently visited
regions have low free energy and appear as basins, which are stable states, while rarely visited
regions have high free energy and are barriers. The FES plots are a quick way to see how many states
your simulation found and how they connect.

## Clustering

**Clustering** groups frames that sit close together in the CV space, so that each group is roughly one
state of the molecule. [`traj_cluster`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_cluster.html)
does this and saves a representative structure for each cluster, so that you can look at what each
state looks like.

## Enhanced sampling with PLUMED

Plain MD often stays trapped in one state for the whole simulation. **Enhanced sampling** methods,
such as metadynamics and OPES, add a bias along a CV that pushes the system to explore new regions.
[PLUMED](https://www.plumed.org/) is the library that applies this bias inside MD engines like GROMACS
or AMBER.

Deep Cartograph writes ready-to-adapt PLUMED input files for the CVs it trains. You can use them to run
biased simulations, or simply to compute the CV on new trajectories. PLUMED 2.9 or newer, compiled with
its PyTorch interface, is needed to use the neural-network CVs.
