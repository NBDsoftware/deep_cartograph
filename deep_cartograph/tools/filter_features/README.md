# Filter features

Filter the features in a colvars file by bimodality, standard deviation or entropy, keeping the most informative subset.

## Description

<!-- TODO: what the tool does, when to use it, and how it fits into the Deep Cartograph workflow. -->

## Usage

### Command line

```bash
conda activate deep_cartograph
filter_features -h
```

<!-- TODO: minimal working example. -->

### Python API

```python
from deep_cartograph.tools.filter_features import filter_features
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

The tool reads a YAML configuration file: the default configuration is in [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/filter_features/default_config.yml); it is validated against the schema in [`yaml_schemas/filter_features.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/filter_features.py).

<!-- TODO: describe the main configuration sections. -->

## Output

<!-- TODO: files and folders written to the output folder. -->

## Recommendations

<!-- TODO -->

## Limitations

<!-- TODO -->
