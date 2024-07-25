# Deep Cartograph

This package can be used to train different collective variables from simulation data. Either to analyze existing trajectories or to use them to enhance the sampling in subsequent simulations. It leverages a custom version of [mlcolvar](https://github.com/NBDsoftware/mlcolvar) to compute or train the different collective variables. See the original publication of the mlcolvar library [here](https://pubs.aip.org/aip/jcp/article-abstract/159/1/014801/2901354/A-unified-framework-for-machine-learning?redirectedFrom=fulltext).

Starting from a trajectory and topology files, deep cartograph can be used to:

1. Compute a set of features to encode the trajectory in a lower dimensional space invariant to rotations and translations.
2. Filter and cluster the features to keep only the most relevant ones. Based on the standard deviation, the entropy, the Hartigan's dip test for unimodality and the mutual information between features.
3. Compute and train different collective variables (CVs) using the filtered features.
4. Project the trajectory onto the CV and cluster it using hierarchical, kmeans or hdbscan algorithms.

### Project structure

- **data**: contains the data used for the tests and the examples.
- **deep_cartograph**: contains all the tools and modules that form part of the deep_cartograph package.
- **examples**: contains examples of how to use the package.
- **tests**: contains the tests of the package.

## Installation

Using conda, create the deep cartograph environment from the `environment.yml` file.

```
git clone https://github.com/NBDsoftware/deep_cartograph.git
cd deep_cartograph
conda env create -f environment.yml
```

Activate the environment and install the custom version of [mlcolvar](https://github.com/NBDsoftware/mlcolvar).

```
cd ../
conda activate deep_cartograph
git clone https://github.com/NBDsoftware/mlcolvar.git
cd mlcolvar
pip install .
```

Finally install the deep cartograph package itself.

```
cd deep_cartograph
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
