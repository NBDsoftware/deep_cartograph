# Deep Cartograph workflow

Map trajectories onto collective variables: featurize, filter, train CVs, project and cluster in one command.

## Description

<!-- TODO: describe how `deep_carto` chains analyze_geometry, traj_augmentation, compute_features,
filter_features, train_colvars and traj_cluster, and when to use it instead of the individual tools. -->

## Usage

### Command line

```bash
conda activate deep_cartograph
deep_carto -conf config.yml -traj_data trajectories/ -top_data topologies/ -out output/
```

Full `deep_carto -h` output:

```
usage: Deep Cartograph [-h] -conf CONFIGURATION_PATH -traj_data TRAJECTORY_DATA -top_data TOPOLOGY_DATA [-sup_traj_data SUPPLEMENTARY_TRAJ_DATA] [-sup_top_data SUPPLEMENTARY_TOP_DATA]
                       [-ref_top REFERENCE_TOPOLOGY] [-restart] [-dim DIMENSION] [-cvs CVS [CVS ...]] [-out OUTPUT_FOLDER] [-v]

Map trajectories onto Collective Variables.

options:
  -h, --help            show this help message and exit
  -conf CONFIGURATION_PATH, -configuration CONFIGURATION_PATH
                        Path to configuration file (.yml).
  -traj_data TRAJECTORY_DATA
                        Path to trajectory or folder with trajectories to analyze. Accepted formats: .xtc .dcd .pdb .xyz .gro .trr .crd.
  -top_data TOPOLOGY_DATA
                        Path to topology or folder with topology files for the trajectories. If a folder is provided, each topology should have the same name as the corresponding
                        trajectory in -traj_data. Accepted format: .pdb.
  -sup_traj_data SUPPLEMENTARY_TRAJ_DATA
                        Path to supplementary trajectory or folder with supplementary trajectories. Used to project onto the CV alongside 'trajectory_data' but not for computing CVs.
  -sup_top_data SUPPLEMENTARY_TOP_DATA
                        Path to supplementary topology or folder with supplementary topologies. If a folder is provided, each topology should match the corresponding supplementary
                        trajectory in -sup_traj_data.
  -ref_top REFERENCE_TOPOLOGY
                        Path to reference topology file. Used to find features from user selections. Defaults to the first topology in topology_data. Accepted format: .pdb.
  -restart              Restart workflow from the last finished step. Deletes step folders for repeated steps.
  -dim DIMENSION, -dimension DIMENSION
                        Dimension of the CV to train or compute. Overrides the configuration input YML.
  -cvs CVS [CVS ...]    Collective variables to train or compute (pca, ae, tica, htica, deep_tica). Overrides the configuration input YML.
  -out OUTPUT_FOLDER, -output OUTPUT_FOLDER
                        Path to the output folder.
  -v, -verbose          Set logging level to DEBUG.
```

### Python API

```python
from deep_cartograph.deep_carto import deep_cartograph
```

<!-- TODO: minimal working example. -->

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| | | <!-- TODO --> |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| | | <!-- TODO --> |

## Configuration

The workflow reads a YAML configuration file with one section per step (`analyze_geometry`,
`traj_augmentation`, `compute_features`, `filter_features`, `train_colvars`, `clustering`). The
default configuration is in
[`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/default_config.yml)
and it is validated against the schema in
[`yaml_schemas/deep_cartograph.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/deep_cartograph.py).

<!-- TODO: describe the main configuration sections. -->

## Output

<!-- TODO: step folders written to the output folder. -->

## Recommendations

**PLUMED interface**: the resulting Deep Learning CVs can be deployed for enhancing sampling with
the [PLUMED](https://www.plumed.org/) package via the
[pytorch](https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html) interface,
available since version 2.9.

<!-- TODO -->

## Limitations

<!-- TODO -->
