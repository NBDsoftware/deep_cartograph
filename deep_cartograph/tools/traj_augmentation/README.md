# Trajectory augmentation

Augment trajectory samples by interpolating between existing frames.

## Description

`traj_augmentation` creates a new
[trajectory](https://nbdsoftware.github.io/deep_cartograph/concepts.html#trajectories-and-topologies)
with more frames by interpolating the atom positions between the existing frames. It can also add a
small random displacement (noise) to the new frames, to mimic thermal fluctuations.

Use it when you have only a few structures, for example a short path between two states from a
coarse-grained or morphing method, and you need more frames to train a
[collective variable](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables)
(a few numbers that summarize the state of the molecule). You can also use it to build extra
trajectories to validate a trained collective variable.

In the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html),
this step runs on the *seed* trajectories (`-seed_traj_data`), after
[`analyze_geometry`](https://nbdsoftware.github.io/deep_cartograph/tools/analyze_geometry.html). The
augmented trajectories are added to the training data and go on to
[`compute_features`](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html).

## Usage

### Command line

```bash
conda activate deep_cartograph
traj_augmentation -h
```

```bash
traj_augmentation -conf config.yml -traj_data path.dcd -top_data path.pdb -output augmented/
```

Create three noisy copies of each trajectory (set `noise_std` in the configuration):

```bash
traj_augmentation -conf config.yml -traj_data trajectories/ -top_data topologies/ -n 3 -output augmented/
```

### Python API

```python
from deep_cartograph.tools.traj_augmentation import traj_augmentation

configuration = {
    "num_frames": 500,
    "interpolation_method": "pchip",
    "noise_std": 0.1,
    "traj_format": "dcd",
}

new_trajectories, new_topologies = traj_augmentation(
    configuration=configuration,
    trajectory_data="path.dcd",
    topology_data="path.pdb",
    num_replicas=3,
    output_folder="augmented",
)
```

The function returns two lists: the paths to the new trajectories and to their topologies, in the
same order.

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | Configuration file (`.yml`). See the Configuration section below. |
| `-traj_data` | — | Trajectory files, or a folder with trajectories, to augment. Formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`. |
| `-top_data` | — | Topologies (`.pdb`), or a folder with topologies. A single topology is used for all trajectories; otherwise each topology must have the same file name as its trajectory. |
| `-output` | `traj_augmentation` | Output folder. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-n`, `-num_replicas` | `1` | Number of augmented trajectories to create from each input trajectory. The copies only differ if `noise_std` is set. |
| `-v`, `--verbose` | off | Write detailed (debug) messages to the log. |

## Configuration

The tool reads a YAML configuration file: it is validated against the schema in [`yaml_schemas/traj_augmentation.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/traj_augmentation.py).

Options you leave out take their default value, so an empty file is valid. In the full workflow, the
same options go under the `traj_augmentation` section.

| Option | Default | Description |
|--------|---------|-------------|
| `num_frames` | `1000` | Number of frames in the new trajectory. |
| `keep_original_frames` | `False` | If `True`, keep the original frames and add interpolated frames between them. If `False`, the new trajectory has only interpolated, evenly spaced frames. |
| `interpolation_method` | `pchip` | `pchip` or `akima`. `pchip` avoids overshooting when positions change abruptly; `akima` gives smoother paths. Set to `null` to skip the interpolation and keep the original frames (for example, to only add noise). |
| `noise_std` | `null` | Size of the random displacement added to every atom coordinate of the new frames, in Å (standard deviation of a Gaussian). `null` adds no noise. |
| `random_seed` | `42` | Seed for the noise, so that results are reproducible. Each replica uses the next seed. |
| `atom_selection` | `all` | Atoms to keep in the new trajectory, in [MDAnalysis selection syntax](https://userguide.mdanalysis.org/stable/selections.html). |
| `traj_format` | `xtc` | Format of the new trajectory: `xtc`, `dcd`, `nc` or `pdb`. |
| `prepare_trajectory` | `False` | If `True`, make the molecules whole and center them before interpolating. It is better to do this beforehand. |

## Output

```text
augmented/
├── deep_cartograph.log                                 # log of the run
├── configuration.yml                                   # configuration used, including default values
├── <trajectory>_augmented_<method>[_rep<i>].<format>   # new trajectory (the _rep<i> suffix only with -n > 1)
└── <trajectory>_augmented_<method>[_rep<i>].pdb        # matching topology, with only the selected atoms
```

## Recommendations

**Prepare the input first**: interpolation works on the raw atom positions. Molecules split across
the periodic box, or frames with different orientations, give unphysical intermediate structures.
Make the molecules whole and align the frames first, for example with
[`align_trajectories`](https://nbdsoftware.github.io/deep_cartograph/tools/align_trajectories.html).

**Interpolated frames are not MD**: the new frames are smooth guesses between the original ones, not
samples from a simulation. Use them to help train or validate a collective variable, not to
estimate populations or free energies.

**Choosing the noise**: small values (a fraction of an Å) mimic thermal motion without distorting the
structure. Start small and check the result visually.

**Changing settings**: if the output files already exist in the output folder, they are reused and
nothing is recomputed. Use a new output folder, or delete the old files, after changing the
configuration.

## Limitations

- **Memory.** Each trajectory is loaded fully into memory, which can be a problem for large systems
  with many frames. Use `atom_selection` to keep only the atoms you need.
- **Topology format.** Topologies must be PDB files.
