import os
import tempfile

import pytest

from deep_cartograph.modules.common import check_data


def _touch(path):
    """Create an empty file at path."""
    with open(path, "w"):
        pass
    return path


def test_check_data_single_topology_expansion():
    """A single topology given for many trajectories is expanded to one per trajectory."""

    with tempfile.TemporaryDirectory() as tmp:
        traj_a = _touch(os.path.join(tmp, "traj_a.pdb"))
        traj_b = _touch(os.path.join(tmp, "traj_b.pdb"))
        shared_top = _touch(os.path.join(tmp, "shared_top.pdb"))

        trajs, tops = check_data([traj_a, traj_b], [shared_top])

        assert len(trajs) == 2
        assert tops == [shared_top, shared_top]


def test_check_data_is_idempotent():
    """Calling check_data on its own output (expanded single topology) must not fail."""

    with tempfile.TemporaryDirectory() as tmp:
        traj_a = _touch(os.path.join(tmp, "traj_a.pdb"))
        traj_b = _touch(os.path.join(tmp, "traj_b.pdb"))
        shared_top = _touch(os.path.join(tmp, "shared_top.pdb"))

        # First call expands the single topology.
        trajs, tops = check_data([traj_a, traj_b], [shared_top])

        # Second call over the already-expanded lists must return normally
        # (previously this hit the name-matching branch and called sys.exit(1)).
        trajs_2, tops_2 = check_data(trajs, tops)

        assert trajs_2 == trajs
        assert tops_2 == tops


def test_check_data_distinct_topologies_matched_by_name():
    """Distinct topologies are paired with trajectories by matching filename stems."""

    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = os.path.join(tmp, "traj")
        top_dir = os.path.join(tmp, "top")
        os.makedirs(traj_dir)
        os.makedirs(top_dir)

        traj_a = _touch(os.path.join(traj_dir, "sys_a.pdb"))
        traj_b = _touch(os.path.join(traj_dir, "sys_b.pdb"))
        top_a = _touch(os.path.join(top_dir, "sys_a.pdb"))
        top_b = _touch(os.path.join(top_dir, "sys_b.pdb"))

        trajs, tops = check_data([traj_a, traj_b], [top_a, top_b])

        assert len(trajs) == 2
        assert len(tops) == 2


def test_check_data_distinct_topologies_name_mismatch_exits():
    """Distinct topologies whose stems don't match the trajectories still error out."""

    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = os.path.join(tmp, "traj")
        top_dir = os.path.join(tmp, "top")
        os.makedirs(traj_dir)
        os.makedirs(top_dir)

        traj_a = _touch(os.path.join(traj_dir, "traj_a.pdb"))
        traj_b = _touch(os.path.join(traj_dir, "traj_b.pdb"))
        top_x = _touch(os.path.join(top_dir, "x.pdb"))
        top_y = _touch(os.path.join(top_dir, "y.pdb"))

        with pytest.raises(SystemExit):
            check_data([traj_a, traj_b], [top_x, top_y])
