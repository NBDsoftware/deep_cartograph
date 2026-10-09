# Align trajectories

Align trajectories to a reference topology using MDAnalysis.

## Description

`align_trajectories` superimposes a set of protein
[trajectories](https://nbdsoftware.github.io/deep_cartograph/concepts.html#trajectories-and-topologies)
onto one reference structure. It fits every frame on the CA atoms of the residues that all the
systems share. These residues are found by aligning the sequences, so the tool also works when the
systems have different sequences or residue numbering, for example different constructs, mutants or
homologous proteins.

Use it to prepare your data before the rest of Deep Cartograph. It is useful before
[`traj_augmentation`](https://nbdsoftware.github.io/deep_cartograph/tools/traj_augmentation.html),
which interpolates atom positions between frames, and when you compute features from atom
coordinates, which depend on the orientation of the molecule. It also makes visual comparisons
easier. It is not part of the
[Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html),
so you run it separately and give its output to `deep_carto` or to the other tools.

## Usage

### Command line

```bash
conda activate deep_cartograph
align_trajectories -h
```

```bash
align_trajectories -traj_data trajectories/ -top_data topologies/ -ref_top reference.pdb -output aligned/
```

### Python API

```python
from deep_cartograph.tools.align_trajectories import align_trajectories

align_trajectories(
    trajectory_data=["traj_1.xtc", "traj_2.xtc"],
    topology_data=["traj_1.pdb", "traj_2.pdb"],
    ref_topology="reference.pdb",
    output_folder="aligned",
)
```

The function returns nothing. The aligned files are written to `output_folder`.

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-traj_data` | — | Trajectory files, or a folder with trajectories, to align. Formats: `.xtc`, `.dcd`, `.pdb`, `.xyz`, `.gro`, `.trr`, `.crd`. |
| `-top_data` | — | Topologies (`.pdb`), or a folder with topologies. A single topology is used for all trajectories; otherwise each topology must have the same file name as its trajectory. |
| `-ref_top` | first topology in `-top_data` | Reference structure (`.pdb`) to align to. |
| `-output` | `align_trajectories` | Output folder. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-v`, `--verbose` | off | Write detailed (debug) messages to the log. |

## Output

```text
aligned/
├── deep_cartograph.log    # log of the run
├── <trajectory>.<ext>     # aligned trajectory, with the same name and format as the input
└── <topology>.pdb         # first frame of the aligned trajectory, to use as its new topology
```

Use the aligned trajectories together with the new topologies: the original topologies are still in
the old orientation.

## Recommendations

**Choosing the reference**: pick a structure that contains the regions you care about, such as an
experimental structure of one of the states. Residues that are missing from any of the systems are
not used for the fit.

**Check the log**: it reports how many residues were used for the fit. A very low number means that
the sequences are too different, or that the topologies are missing parts of the protein.

## Limitations

- **Proteins only.** The fit uses protein CA atoms, so systems without protein residues cannot be
  aligned.
- **Topology format.** Topologies and the reference must be PDB files.
- **Whole-molecule fit.** The whole system is moved to fit the shared CA atoms. If the protein is
  split across the periodic box, make it whole first.
