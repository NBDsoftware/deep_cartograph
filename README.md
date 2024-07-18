# Deep Cartograph

This package can be used to train different collective variables from simulation data. Either to analyze existing trajectories or to use them to enhance the sampling in subsequent simulations. It leverages a custom version of [mlcolvars](https://github.com/NBDsoftware/mlcolvar) to compute or train the different collective variables.

Starting from a trajectory and topology files, the package will:

1. Compute a set of features to encode the trajectory in a lower dimensional space invariant to rotations and translations.
2. Filter the features to keep only the most relevant ones.
3. Compute and train different collective variables (CVs) using the filtered features.
4. Cluster the trajectory using the collective variables.

### Project structure

- **data**: contains the data used for the tests and the examples.
- **deep_cartograph**: contains all the tools and modules that form part of the deep_cartograph package.
- **examples**: contains examples of how to use the package.
- **tests**: contains the tests of the package.

## Installation

Using conda, create the environment:

```
conda env create -f environment.yml
```

Then install the custom version of [mlcolvars](https://github.com/NBDsoftware/mlcolvar) in the same environment.

```
conda activate deep_cartograph
git clone https://github.com/NBDsoftware/mlcolvar.git
cd mlcolvar
pip install .
```

## Usage

The main workflow can be used with `deep_cartograph/run.py`, see available options:

```
usage: Deep Cartograph [-h] -conf CONFIGURATION_PATH -traj TRAJECTORY -top TOPOLOGY [-ref REFERENCE_FOLDER] [-use_rl] [-dim DIMENSION] [-cvs CVS [CVS ...]] -out OUTPUT_FOLDER [-v]

Map trajectories onto Collective Variables.

options:
  -h, --help            show this help message and exit
  -conf CONFIGURATION_PATH, -configuration CONFIGURATION_PATH
                        Path to configuration file (.yml)
  -traj TRAJECTORY, -trajectory TRAJECTORY
                        Path to trajectory file, for which the features are computed.
  -top TOPOLOGY, -topology TOPOLOGY
                        Path to topology file.
  -ref REFERENCE_FOLDER, -reference REFERENCE_FOLDER
                        Path to folder with reference data. It should contain structures or trajectories.
  -use_rl, -use_reference_lab
                        Use labels for reference data (names of the files in the reference folder)
  -dim DIMENSION, -dimension DIMENSION
                        Dimension of the CV to train or compute
  -cvs CVS [CVS ...]    Collective variables to train or compute (pca, ae, tica, dtica)
  -out OUTPUT_FOLDER, -output OUTPUT_FOLDER
                        Path to the output folder
  -v, -verbose          Set the logging level to DEBUG
```

An example for the YAML configuration file can be found here `deep_cartograph/default_config.yml`.

**PLUMED interface**: the resulting Deep Learning CVs can be deployed for enhancing sampling with the [PLUMED](https://www.plumed.org/) package via the [pytorch](https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html>`_) interface, available since version 2.9. 
