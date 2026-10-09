# Compute features

Compute features from a trajectory using PLUMED.

## Description

This tool measures a set of features in every frame of one or more MD trajectories: distances between atoms, dihedral angles, distances to the center of a group of atoms, or atomic coordinates. You choose which [features](https://nbdsoftware.github.io/deep_cartograph/concepts.html#features) to compute in a YAML configuration file, using atom selections. The calculation itself is done by [PLUMED](https://www.plumed.org/).

For each trajectory, the tool writes a [colvars file](https://nbdsoftware.github.io/deep_cartograph/concepts.html#colvars-files): a text table with one row per frame and one column per feature. You can analyze these tables directly, or use them to build [collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables) (a few numbers that summarize the main changes of the system).

This is the first step of the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html). Its colvars files are the input of [filter_features](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html), which keeps only the most informative features, and then of [train_colvars](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html). If you give several trajectories with different [topologies](https://nbdsoftware.github.io/deep_cartograph/concepts.html#trajectories-and-topologies) (for example, a wild type and a mutant), the tool matches the atoms between them and computes only the features that exist in all of them, so every colvars file has the same columns.

## Usage

### Command line

```bash
conda activate deep_cartograph
compute_features -h
```

Compute the features defined in `config.yml` for one trajectory, using one frame out of every 10:

```bash
compute_features -conf config.yml -traj_data traj.xtc -top_data top.pdb -traj_stride 10 -output compute_features
```

Compute features for all the trajectories in a folder. Each trajectory needs a topology with the same file name (e.g. `trajectories/rep1.xtc` and `topologies/rep1.pdb`):

```bash
compute_features -conf config.yml -traj_data trajectories/ -top_data topologies/ -output compute_features
```

### Python API

```python
from deep_cartograph.tools.compute_features import compute_features
from deep_cartograph.modules.common import read_configuration

configuration = read_configuration("config.yml")

colvars_paths = compute_features(
    configuration=configuration,
    trajectory_data=["rep1.xtc", "rep2.xtc"],
    topology_data="top.pdb",        # one topology shared by all trajectories
    traj_stride=10,                 # optional, overrides plumed_settings.traj_stride
    output_folder="compute_features",
)
# colvars_paths: ['compute_features/rep1/colvars.dat', 'compute_features/rep2/colvars.dat']
```

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | YAML configuration file that defines the features to compute (see [Configuration](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html#configuration)). |
| `-traj_data` | required | One or more trajectory files, or a folder with trajectories. Formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`. |
| `-top_data` | required | One or more topology files, or a folder with topologies (`.pdb`). A single topology is used for all trajectories; otherwise each topology must have the same name as its trajectory. |
| `-output` | `compute_features` | Output folder. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-traj_stride` | — | Use one frame out of every `traj_stride`. Overrides `plumed_settings.traj_stride` in the configuration. |
| `-v`, `--verbose` | off | Write more detailed messages to the log. |

## Configuration

The tool reads a YAML configuration file: the default configuration is in [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/compute_features/default_config.yml); it is validated against the schema in [`yaml_schemas/compute_features.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/compute_features.py).

When you run this tool on its own, `plumed_settings` and `plumed_environment` are at the top level of the file. In a configuration for the full [`deep_carto` workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html), the same content goes under a `compute_features:` section.

### Choosing the features

The main decision is which features to compute. You define them in `plumed_settings.features` as one or more named groups. There are four kinds of groups, and you can combine several groups of the same or different kinds:

- **`distance_groups`**: the distance between every pair of atoms, one atom taken from `first_selection` and the other from `second_selection`. Distances are in nm.
- **`dihedral_groups`**: dihedral (torsion) angles between four atoms, found inside `selection`. The `search_mode` option decides which dihedrals are found:
  - `real`: all dihedrals formed by four bonded heavy atoms, for example side-chain angles. Use it for all-atom systems.
  - `protein_backbone`: the backbone phi and psi angles of each selected residue.
  - `virtual`: dihedrals formed by every four consecutive selected atoms, even if they are not bonded. Use it for models with one bead per residue, or with a selection like `name CA` to follow the shape of the protein chain.
- **`distance_to_center_groups`**: the distance from each atom in `selection` to the geometric center of the atoms in `center_selection`.
- **`coordinate_groups`**: the x, y and z positions of each atom in `selection`. Frames are aligned on the backbone before the positions are read.

Atoms are selected with the [MDAnalysis selection syntax](https://docs.mdanalysis.org/stable/documentation_pages/selections.html), e.g. `"name CA"`, `"resid 10:50 and backbone"` or `"not name H*"`. The selections are applied to the topology.

For example, this configuration computes distances between C-alpha atoms and the backbone angles of residues 20 to 80:

```yaml
plumed_settings:
  traj_stride: 1
  features:
    distance_groups:
      ca_distances:                   # any name you like
        first_selection: "name CA"
        second_selection: "name CA"
        first_stride: 2               # use every 2nd C-alpha of the first selection
        second_stride: 2
        skip_neigh_residues: True     # skip pairs in the same or neighboring residues
    dihedral_groups:
      backbone_angles:
        selection: "resid 20:80"
        search_mode: protein_backbone
        periodic_encoding: True       # store sin and cos of each angle

plumed_environment:
  bin_path: plumed
```

### Main options

| Option | Default | Description |
|--------|---------|-------------|
| `plumed_settings.traj_stride` | `1` | Use one frame out of every `traj_stride`. |
| `plumed_settings.timeout` | `172800` | Maximum time for each PLUMED run, in seconds (48 h). |
| `plumed_settings.features` | — | Groups of features to compute (see above). You must define at least one group. |
| `...distance_groups.<name>.first_selection`, `second_selection` | `"not name H*"` | The two atom selections. A distance is computed for every pair of atoms. |
| `...distance_groups.<name>.first_stride`, `second_stride` | `1`, `5` | Use only one atom out of every `stride` in each selection, to reduce the number of distances. |
| `...distance_groups.<name>.skip_neigh_residues` | `False` | Skip distances between atoms in the same or in consecutive residues. |
| `...distance_groups.<name>.skip_bonded_atoms` | `True` | Skip distances between bonded atoms. |
| `...dihedral_groups.<name>.selection` | `"not name H*"` | Atoms in which dihedrals are searched. |
| `...dihedral_groups.<name>.search_mode` | `real` | Which dihedrals to find: `real`, `protein_backbone` or `virtual` (see above). |
| `...dihedral_groups.<name>.periodic_encoding` | `True` | Store each angle as two features, its sine and cosine, so that -180° and 180° get the same value. If `False`, store the angle itself (in radians). |
| `...distance_to_center_groups.<name>.selection` | `"not name H*"` | Atoms whose distance to the center is computed. |
| `...distance_to_center_groups.<name>.center_selection` | `"not name H*"` | Atoms whose geometric center is used. |
| `...coordinate_groups.<name>.selection`, `stride` | `"not name H*"`, `1` | Atoms whose x, y, z positions are used, keeping one atom out of every `stride`. |
| `plumed_environment.bin_path` | `plumed` | PLUMED executable. The conda environment already provides one, so `plumed` is usually enough. |
| `plumed_environment.kernel_path` | — | Path to the PLUMED kernel library, if your installation needs it. |
| `plumed_environment.env_commands` | `[]` | Shell commands to run before PLUMED, e.g. `module load PLUMED`. |

`...` stands for `plumed_settings.features`. Defaults are the values used when an option is left out of your file. See the full [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/compute_features/default_config.yml) for a complete example.

## Output

```text
compute_features/
├── deep_cartograph.log          # log of the run (command line only)
├── configuration.yml            # configuration actually used, including defaults
├── ref_topology.pdb             # copy of the reference topology used to name the features
├── common_features/
│   └── ref_topology.pdb         # copy of the reference topology used to match features between topologies
└── <trajectory name>/           # one folder per trajectory
    ├── colvars.dat              # the colvars file: one row per frame, one column per feature
    ├── plumed_input.dat         # PLUMED input used to compute the features
    ├── plumed_topology.pdb      # copy of the topology read by PLUMED
    └── fit_template.pdb         # reference used to align frames (only with coordinate features)
```

Each column of `colvars.dat` is named after the feature it holds, for example `dist-@CA_12-@CA_45` (distance between the C-alpha atoms of residues 12 and 45), `sin-@phi_30` and `cos-@phi_30` (sine and cosine of the phi angle of residue 30) or `coord-@CA_12.x`. The first column, `time`, is the frame number in the original trajectory.

## Recommendations

- **Keep the number of features under control.** The number of distances grows with the product of the two selections: 300 C-alpha atoms give about 45,000 pairs. Use the strides, `skip_neigh_residues`, or narrower selections (for example only the region that moves) to keep it to a few thousand at most. Too many features make the next steps slow and memory hungry.
- **Use strides on the trajectory for long simulations.** Frames that are very close in time carry almost the same information. `-traj_stride` reduces the size of the colvars files and the time of the next steps.
- **Prefer `periodic_encoding: True` for angles.** An angle jumps from 180° to -180° although the structure barely changes; its sine and cosine do not.
- **Be careful when mixing types of features.** Distances are in nm, angles in radians and their sine and cosine between -1 and 1. Mixing them is fine, but some filters in [filter_features](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html) compare values across features and only make sense when all features share the same units.
- **Prepare your trajectories first.** Remove water and ions and make the molecules whole before computing features. Files are smaller and the features are measured correctly.
- **Coordinates depend on the alignment.** With `coordinate_groups`, each frame is aligned on the backbone of its own topology. If you combine several systems, make sure their topologies are aligned to each other, otherwise the same coordinate means different things in each system.
- **Delete old results to recompute.** If a `colvars.dat` already exists in the output folder, that trajectory is skipped. Remove the output folder (or use a new one) after changing the features.
- **AMBER trajectories.** If PLUMED fails with an error in `LatticeReduction.cpp`, see the [FAQ](https://nbdsoftware.github.io/deep_cartograph/faq.html).

## Limitations

- **Topology format.** Topologies must be PDB files. Trajectories can be `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr` or `.crd`.
- **PLUMED is required.** The conda environment installs PLUMED 2.9.0. To use another installation, set `plumed_environment` in the configuration.
- **Unique atom names.** Features are named by atom name and residue number. If several atoms share both (for example two chains with the same residue numbers), only one feature per name is kept and the rest are dropped with a warning. Renumber the residues so that they are unique.
- **Hydrogens in dihedrals.** The `real` and `virtual` search modes only use heavy atoms.
