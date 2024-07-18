# Deep Cartograph

This package can be used to train different collective variables from simulation data. Either to analyze existing trajectories or to use them to enhance the sampling in subsequent simulations. It leverages the mlcolvars library to compute or train the different collective variables.

Folders:

- **data/**: contains the data used to test the package
- **deep_cartograph/**: contains all the tools and modules that form part of the deep_cartograph package. The source code.
- **tests/**: contains the tests of the package

## Installation

Using conda, create the environment:

```
conda env create -f environment.yml
```

Then install the custom version of [mlcolvars](https://github.com/PabloNA97/mlcolvar) (deep_cartograph branch) in the same environment.

## Usage

To run the full workflow, use the deep_cartograph/run.py script:

```
python run.py -conf config.yaml -traj trajectory.xtc -top topology.pdb -out output
```

Or use one of the tools independently:

