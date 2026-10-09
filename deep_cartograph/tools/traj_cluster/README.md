# Trajectory clustering

Group trajectory frames into states based on their collective variable values, and extract a representative structure of each state.

## Description

After projecting your trajectories onto [collective variables](https://nbdsoftware.github.io/deep_cartograph/concepts.html#collective-variables) (CVs), frames that look alike end up close together, often inside the basins of the [free energy surface](https://nbdsoftware.github.io/deep_cartograph/concepts.html#free-energy-surface). This tool finds those groups automatically ([clustering](https://nbdsoftware.github.io/deep_cartograph/concepts.html#clustering)). It labels each frame with its cluster and saves one representative structure per cluster (the frame closest to the cluster center). You can also save all the frames of each cluster as a separate trajectory.

Use it to turn a CV map into a small set of states you can look at, compare, or use as starting points for new simulations. You can also give supplementary trajectories: their frames are not used to find the clusters, but each one is assigned to the nearest existing cluster.

In the [Deep Cartograph workflow](https://nbdsoftware.github.io/deep_cartograph/workflow/deep_carto.html), this tool comes last. Its input is the `projected_trajectory.csv` file written by [train_colvars](https://nbdsoftware.github.io/deep_cartograph/tools/train_colvars.html) or [traj_projection](https://nbdsoftware.github.io/deep_cartograph/tools/traj_projection.html).

## Usage

### Command line

```bash
conda activate deep_cartograph
traj_cluster -h
```

Cluster the frames of a trajectory projected onto a PCA model, and assign the frames of a second trajectory to the same clusters:

```bash
traj_cluster -conf config.yml \
             -cv_traj train_colvars/pca/traj_data/my_traj/projected_trajectory.csv \
             -trajectory my_traj.xtc -topology protein.pdb \
             -sup_cv_traj traj_projection/pca/new_replica/projected_trajectory.csv \
             -sup_trajectory new_replica.xtc -sup_topology protein.pdb \
             -frames_per_sample 10 -out traj_cluster
```

### Python API

```python
from deep_cartograph.tools.traj_cluster import traj_cluster

results = traj_cluster(
    configuration={"algorithm": "hierarchical", "search_interval": [3, 8]},
    cv_traj_paths=["train_colvars/pca/traj_data/my_traj/projected_trajectory.csv"],
    trajectories=["my_traj.xtc"],
    topologies=["protein.pdb"],
    frames_per_sample=10,
    output_folder="traj_cluster",
)

# One entry per trajectory: the CV values with a cluster label per frame
results["my_traj"]  # ['traj_cluster/my_traj/projected_trajectory.csv']
```

## Options

### Inputs

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-conf`, `-configuration` | required | YAML configuration file (see Configuration below). |
| `-cv_traj`, `-cv_trajectory` | required | CSV file with the CV values of each frame (`projected_trajectory.csv`). These frames define the clusters. |
| `-trajectory` | — | Trajectory file that the CV values come from. Needed to extract the structures of each cluster. |
| `-topology` | — | Topology of the trajectory. Required if `-trajectory` is given. |
| `-sup_cv_traj`, `-sup_cv_trajectory` | required | CSV file with the CV values of a supplementary trajectory. Each of its frames is assigned to the nearest cluster. |
| `-sup_trajectory` | — | Trajectory file of the supplementary CV values. Required if `-sup_cv_traj` is given; used to name its output folder. |
| `-sup_topology` | — | Topology of the supplementary trajectory. Required if `-sup_trajectory` is given. |
| `-out`, `-output` | — | Output folder. If not given, `traj_cluster` is used. If the folder exists, a new one with a number suffix is created. |

### Parameters

| Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Default | Description |
|------|---------|-------------|
| `-frames_per_sample` | — | Number of trajectory frames between two rows of the CV file (the stride used when computing the features). Used to extract the right frames. If not given, 1 is used. |
| `-v`, `--verbose` | off | Write more detailed messages to the log. |

## Configuration

The configuration file chooses the clustering method and what to save; write the options at the top level of the file. Any option you leave out takes the default shown below.

| Option | Default | Description |
|--------|---------|-------------|
| `run` | `true` | Set to `false` to skip the clustering. |
| `algorithm` | `hierarchical` | Clustering method: `hierarchical`, `kmeans` or `hdbscan` (see Recommendations below). |
| `search_interval` | `[3, 10]` | For `hierarchical` and `kmeans`: smallest and largest number of clusters to try. The tool keeps the number that gives the most compact, best separated clusters. |
| `linkage` | `complete` | For `hierarchical`: how the distance between two clusters is measured (`complete`, `average`, `single` or `ward`). |
| `min_cluster_size` | `5` | For `hdbscan`: smallest group of frames that counts as a cluster. Smaller groups are marked as noise. |
| `min_samples` | `3` | For `hdbscan`: higher values give fewer, denser clusters and more frames marked as noise. |
| `cluster_selection_epsilon` | `0` | For `hdbscan`: increase to merge clusters that are close together. |
| `cluster_selection_method` | `eom` | For `hdbscan`: `eom` gives a few large clusters, `leaf` gives many small, uniform ones. |
| `output_structures` | `centroids` | Structures to save: `centroids` (one PDB per cluster), `all` (also every frame of each cluster as a trajectory) or `null` (none). Needs `-trajectory` and `-topology`. |
| `figures.plot` | `true` | Plot the trajectory in the CV space, colored by cluster (2D CVs only). |

The example configuration is in [`default_config.yml`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/tools/traj_cluster/default_config.yml). All options and their defaults are in the schema [`yaml_schemas/traj_cluster.py`](https://github.com/NBDsoftware/deep_cartograph/blob/master/deep_cartograph/yaml_schemas/traj_cluster.py).

## Output

```text
traj_cluster/
├── configuration.yml                  # full configuration used, with all defaults filled in
├── deep_cartograph.log                # log of the run (command line only)
├── clusters_size.png                  # number of frames in each cluster
├── centroids/                         # needs -trajectory and -topology
│   ├── cluster_0.pdb                  # representative structure of each cluster
│   └── cluster_1.pdb
├── my_traj/                           # one folder per trajectory (named after the trajectory file)
│   ├── projected_trajectory.csv       # CV values plus cluster label, centroid flag and frame number
│   ├── trajectory_clustered.png       # trajectory in the CV space, colored by cluster (2D CVs only)
│   └── cluster_0.xtc                  # all frames of each cluster (only with output_structures: all)
└── sup_new_replica/                   # one folder per supplementary trajectory
    ├── projected_trajectory.csv       # CV values plus the assigned cluster
    └── trajectory_clustered.png
```

In `projected_trajectory.csv`, the `cluster` column gives the cluster of each frame. The `centroid` column marks the representative frame of each cluster, and `frame` is the frame number in the original trajectory. With `hdbscan`, frames that do not belong to any cluster get the label `-1` (noise).

## Recommendations

**Which method should I use?**

- **`hierarchical`** (default): builds clusters by merging nearby frames step by step. Reliable for most CV maps.
- **`kmeans`**: splits the frames into round groups of similar size. Fast, but may split a large basin or merge two small ones.
- **`hdbscan`**: finds dense regions of any shape and chooses the number of clusters by itself. Frames in sparse regions (for example transition regions) are left out as noise. A good choice when you don't know how many states to expect.

**How many clusters?**

- Look at the free energy surface plots from `train_colvars` first. The number of clear basins is a good guess for the number of clusters.
- For `hierarchical` and `kmeans`, set `search_interval` around that guess (for example `[3, 6]` if you see four basins). The tool tries every number in the range and picks the best. A narrow range gives more predictable results.
- For `hdbscan`, control the result with `min_cluster_size`. Set it to the smallest number of frames you would accept as a real state. Raise it if you get many tiny clusters; raise `cluster_selection_epsilon` to merge close ones.
- Check `clusters_size.png` and the clustered trajectory plot. Very small clusters are often noise or transition frames.

**Other tips**

- **Set `-frames_per_sample` correctly.** It must match the stride used in `compute_features`; otherwise the saved structures come from the wrong frames.
- **Cluster on few dimensions.** Clustering works best on 1 to 3 CV components.

## Limitations

- **Plots only for 2D CVs.** The clustered trajectory plots are made only when the CV has two components. The cluster labels are always saved.
- **Structures need the trajectory.** Without `-trajectory` and `-topology`, only the cluster labels are saved; no PDB or XTC files are written.
- **Large data sets are slow.** `hierarchical` clustering and the search over many cluster numbers can be slow for trajectories with many frames. Use fewer frames (a larger stride) if needed.
