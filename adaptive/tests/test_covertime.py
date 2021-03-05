import pytest
import numpy as np

import adaptive as ad


@pytest.fixture
def cov_traj_1():
    trajs = [[np.array([1, 2, 3])],
               [np.array([1, 2, 4])],
               [np.array([3, 2, 4])]]
    cov_traj = ad.CoverTrajectories()
    for traj in trajs:
        cov_traj.add(traj)
    return cov_traj


def test_states_per_epoch_1(cov_traj_1):
    ctime = ad.CoverTime(cov_traj_1)

    assert np.allclose()
