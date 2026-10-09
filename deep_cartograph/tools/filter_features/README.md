# Filter features

Filter the features in a colvars file by bimodality, standard deviation or entropy, keeping the most informative subset.

## Description

This tool reads one or more [colvars files](https://nbdsoftware.github.io/deep_cartograph/concepts.html#colvars-files) (tables with the value of each feature in each frame) and keeps only the [features](https://nbdsoftware.github.io/deep_cartograph/concepts.html#features) that are likely to describe the important changes of your system. Features that barely move, or that always fluctuate around a single value, are discarded. The result is a text file with the names of the features that were kept.

Use it when you have computed many features (often thousands of distances or angles) and want a smaller, more informative set. Fewer features make the next steps faster and the resulting [collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables) (a few numbers that summarize the main changes of the system) easier to train and interpret.

In the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html), this tool runs after [compute_features](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html), which produces the colvars files, and before [train_colvars](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html), which uses the list of kept features.

## Usage

### Command line

```bash
conda activate deep_cartograph
filter_features -h
```

Filter the features of a colvars file with the settings in `config.yml`, and save a table with the value of each filter for each feature:

```bash
filter_features -conf config.yml -colvars compute_features/traj/colvars.dat -output filter_features -csv_summary
```

Also discard the features that do not change across a set of reference structures (waypoints, see [Filters](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html#filters)):

```bash
filter_features -conf config.yml -colvars compute_features/traj/colvars.dat \
    -waypoint_colvars waypoints/open/colvars.dat waypoints/closed/colvars.dat -output filter_features
```

### Python API

```python
from deep_cartograph.tools.filter_features import filter_features
from deep_cartograph.modules.common import read_configuration

configuration = read_configuration("config.yml")

features_path = filter_features(
    configuration=configuration,
    colvars_paths=["compute_features/rep1/colvars.dat", "compute_features/rep2/colvars.dat"],
    csv_summary=True,
    output_folder="filter_features",
)
# features_path: 'filter_features/filtered_features.txt'
```

When several colvars files are given, the filters are computed on all of them together.

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | YAML configuration file with the filter settings (see [Configuration](https://nbdsoftware.github.io/deep_cartograph/tools/filter_features.html#configuration)). |
| `-colvars` | required | Colvars file with the features to filter, as written by [compute_features](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html). |
| `-waypoint_colvars` | — | Colvars files with the feature values of a few reference structures along the change you want to study (e.g. the open and closed states). Features that do not change across them are discarded. |
| `-topologies` | — | Topology (`.pdb`) of the system in each colvars file. Only needed when the colvars files come from systems with different topologies (e.g. a wild type and a mutant), so that features can be matched between them. |
| `-waypoint_topologies` | — | Topology (`.pdb`) of each waypoint colvars file, used in the same way as `-topologies`. |
| `-ref_topology` | first of `-topologies` | Topology whose atom names and residue numbers are used to name the kept features. |
| `-output` | `filter_features` | Output folder. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-csv_summary` | off | Also save a table with the value of each filter for each feature, and whether it was kept. |
| `-v`, `--verbose` | off | Write more detailed messages to the log. |

## Configuration

The tool reads a YAML configuration file: the default configuration is in [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/filter_features/default_config.yml); it is validated against the schema in [`yaml_schemas/filter_features.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/filter_features.py).

When you run this tool on its own, `filter_settings` is at the top level of the file. In a configuration for the full [`deep_carto` workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html), the same content goes under a `filter_features:` section.

### Filters

The main decisions are which filters to use and how strict they are. A feature is kept only if it passes every active filter.

- **Multiple peaks** (`diptest_significance_level`): keeps features whose values group around two or more distinct values, i.e. features that take clearly different values in different states. It uses a statistical test (Hartigan's dip test). Lower values are stricter and keep fewer features.
- **Spread** (`std_quantile`): discards the features with the smallest fluctuations (standard deviation). A value of `0.2` removes the 20% of features that fluctuate the least. Higher values remove more.
- **Diversity of values** (`entropy_quantile`): discards the features whose values are concentrated in a narrow set, i.e. the least diverse ones (lowest entropy). A value of `0.2` removes the 20% least diverse. Higher values remove more.
- **Waypoints** (only with `-waypoint_colvars`): keeps features that change clearly between the reference structures: at least 2 Å for distances and 22.5° for angles.
- **Local contacts** (`local_distance_threshold`, only with `-waypoint_colvars`): keeps distances that are shorter than the threshold in at least one of the reference structures, i.e. atoms that come into contact at some point of the change.

Set a filter to `null` to turn it off. For the spread and diversity filters, `0` also turns the filter off but still reports their values in the summary table.

```yaml
filter_settings:
  diptest_significance_level: 0.05   # keep features with more than one peak
  std_quantile: 0.2                  # remove the 20% of features that fluctuate the least
  entropy_quantile: null             # off
  local_distance_threshold: null     # off (only used with waypoints)
```

| Option | Default | Description |
|--------|---------|-------------|
| `filter_settings.diptest_significance_level` | `0.05` | Strictness of the multiple-peaks filter. Lower keeps fewer features; `null` or `0` turns it off. |
| `filter_settings.std_quantile` | `0` (off) | Fraction of features with the smallest fluctuations to discard, between 0 and 1. |
| `filter_settings.entropy_quantile` | `0` (off) | Fraction of the least diverse features to discard, between 0 and 1. |
| `filter_settings.local_distance_threshold` | `null` (off) | Distance in Å below which two atoms are considered in contact. Only used with waypoints. |

See the full [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/filter_features/default_config.yml) for a complete example.

## Output

```text
filter_features/
├── deep_cartograph.log      # log of the run (command line only)
├── configuration.yml        # configuration actually used, including defaults
├── all_features.txt         # all the features found in the input colvars files, one per line
├── filtered_features.txt    # features that passed all the filters, one per line
└── filter_summary.csv       # value of each filter for each feature and whether it was kept (with -csv_summary)
```

`filtered_features.txt` is the list of features that [train_colvars](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html) will use.

## Recommendations

- **Start with the multiple-peaks filter alone.** It is the default and works well when your simulations visit several states. Look at how many features are kept, then add the other filters if you still have too many.
- **Use `-csv_summary` to tune thresholds.** The summary table shows the value of every filter for every feature, so you can see how many features each threshold would remove before choosing it.
- **Mind the units with the spread filter.** It compares fluctuations across features, so it only makes sense when all features share the same unit, for example only distances. Distances (nm) and sine/cosine of angles (between -1 and 1) are not comparable.
- **Short simulations that stay in one state.** The multiple-peaks filter may then discard almost everything. Relax it (higher `diptest_significance_level`) or turn it off and rely on the spread filter or on waypoints.
- **Getting waypoint colvars files.** Run [compute_features](https://nbdsoftware.github.io/deep_cartograph/tools/compute_features.html) on the PDB files of your reference structures, using each PDB as both trajectory and topology and the same feature configuration as for your trajectories.
- **Delete old results to filter again.** If `filtered_features.txt` already exists in the output folder, filtering is skipped. Remove it (or use a new output folder) after changing the settings.

## Limitations

- **One colvars file on the command line.** `-colvars` takes a single file. To filter several colvars files together, use the Python API.
- **Same features in all files.** Only features present in every colvars file are considered. Without `-topologies`, features are matched by name only.
