# Analyze geometry

Simple geometry analysis of trajectories: RMSD, RMSF and dRMSD.

## Description

`analyze_geometry` makes quick plots of standard structural measures for a set of MD
[trajectories](https://nbdsoftware.github.io/deep_cartograph/concepts.html#trajectories-and-topologies):

- **RMSD** of each frame, after fitting it onto a reference. By default the reference is the topology
  structure of each trajectory. You can also give one or more reference structures, for example
  experimental structures of the states you are interested in.
- **RMSF** of each residue along each trajectory.
- **dRMSD** (distance RMSD) of each frame: it compares all the distances between the selected atoms
  with the same distances in a reference, so no fitting is needed.

Use it as a sanity check before a longer analysis: to see whether the simulations are stable, which
regions move the most and how close each trajectory gets to known structures. It is the first step of
the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html),
and it does not affect the following steps.

## Usage

### Command line

```bash
conda activate deep_cartograph
analyze_geometry -h
```

```bash
analyze_geometry -conf config.yml -traj_data trajectories/ -top_data topologies/ -output geometry/
```

Compute the RMSD and dRMSD against two experimental structures:

```bash
analyze_geometry -conf config.yml -traj_data traj.xtc -top_data top.pdb \
                 -ref_top_data references/ -output geometry/
```

### Python API

```python
from deep_cartograph.tools.analyze_geometry import analyze_geometry
from deep_cartograph.modules.common import read_configuration

analyze_geometry(
    configuration=read_configuration("config.yml"),
    trajectories=["traj_1.xtc", "traj_2.xtc"],
    topologies=["traj_1.pdb", "traj_2.pdb"],
    ref_topologies=["open.pdb", "closed.pdb"],   # or None
    output_folder="geometry",
)
```

The function returns nothing. The plots and CSV files are written to `output_folder`.

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf` | required | Configuration file (`.yml`) with the analyses to run. See the Configuration section below. |
| `-traj_data` | required | Trajectory file, or folder with trajectories, to analyze. |
| `-top_data` | required | Topology (`.pdb`), or folder with topologies. With several topologies, each one must have the same file name as its trajectory. |
| `-ref_top_data` | — | Reference structure (`.pdb`), or folder with reference structures, for the RMSD and dRMSD. Without it, each trajectory is compared with its own topology. |
| `-output` | `analyze_geometry` | Output folder. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-v`, `--verbose` | off | Write detailed (debug) messages to the log. |

## Configuration

The tool reads a YAML configuration file: the default configuration is in [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/analyze_geometry/default_config.yml); it is validated against the schema in [`yaml_schemas/analyze_geometry.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/analyze_geometry.py).

Under `analysis`, list the analyses you want, grouped by type (`RMSD`, `RMSF`, `dRMSD`). Give each
analysis a name of your choice; you can define several analyses of the same type. For example:

```yaml
dt_per_frame: 40
analysis:
  RMSD:
    backbone_rmsd:
      title: 'Backbone RMSD'
      selection: 'protein and name CA'
      fit_selection: 'protein and name CA'
  RMSF:
    backbone_rmsf:
      title: 'Backbone RMSF'
      selection: 'protein and name CA'
      fit_selection: 'protein and name CA'
```

| Option | Default | Description |
|--------|---------|-------------|
| `dt_per_frame` | `1.0` | Time between saved frames, in ps. Used only for the time axis of the plots. |
| `run` | `True` | Set to `False` to skip this step (useful in the full workflow). |
| `analysis.<type>.<name>.title` | depends on type | Title of the plot. |
| `analysis.<type>.<name>.selection` | `protein and name CA` | Atoms to analyze, in [MDAnalysis selection syntax](https://userguide.mdanalysis.org/stable/selections.html). |
| `analysis.RMSD.<name>.fit_selection`, `analysis.RMSF.<name>.fit_selection` | `protein and name CA` | Atoms used to fit each frame onto the reference before computing the RMSD or RMSF. |
| `analysis.dRMSD.<name>.selection_stride` | `5` | Use only one of every *n* selected atoms, to keep the number of distances small. |

## Output

```text
geometry/
├── deep_cartograph.log                # log of the run
├── configuration.yml                  # configuration used, including default values
├── <name>_<type>.png                  # one plot per analysis, with one line per trajectory (and reference)
├── <trajectory>first_frame.csv        # RMSD against the topology: time (ns) and RMSD
├── <trajectory>_to_<reference>.csv    # RMSD or dRMSD against a reference: time (ns) and value
├── <trajectory>.csv                   # RMSF: residue and RMSF
└── dRMSD_temp_<trajectory>_to_<reference>/   # intermediate PLUMED files of each dRMSD
```

The CSV files are named after the trajectory and the reference only. If two analyses use the same
trajectory and reference (for example an RMSD and a dRMSD), only the CSV of the last one is kept.
The plots are always kept.

## Recommendations

**Set `dt_per_frame`**: set it to the time between the saved frames of your trajectories, so that the
time axis of the plots is correct. It does not change the computed values.

**Reference structures**: the references do not need the same residue numbering as your
trajectories, because residues are matched through their sequence. This makes it easy to compare with
experimental structures.

**Choosing selections**: CA atoms (`protein and name CA`) give a quick, robust overview. To follow a
specific region, such as a loop or a bound peptide, use it as `selection` and keep a stable part of
the protein as `fit_selection`.

## Limitations

- **Units.** RMSD and RMSF are in Å; dRMSD is in nm.
- **dRMSD needs PLUMED.** The dRMSD uses the `plumed` executable, which must be available in your
  `PATH`.
- **Topology format.** Topologies and references must be PDB files.
