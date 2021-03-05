from typing import List

import pytest
import numpy as np

import adaptive as ad


@pytest.fixture
def cov_trajs():
    return ad.CoverTrajectories()


def trajs_are_identical(a: List[np.ndarray], b: List[np.ndarray]):
    assert len(a) == len(b), 'traj lists are not same size'
    result = np.all([np.allclose(a[i], b[i]) for i in range(len(a))])
    return result


def test_append_trajectory(cov_trajs):
    trajs = [np.array([1])]
    cov_trajs.add_trajectories(new_trajs=trajs)
    assert trajs_are_identical(cov_trajs.trajectories[0], trajs)


def test_multiple_tajectories(cov_trajs):
    multi_trajs = [[np.array([1])], [np.array([2])]]
    for traj in multi_trajs:
        cov_trajs.add_trajectories(traj)

    correct = True
    for i in range(len(multi_trajs)):
        added_traj = cov_trajs.trajectories[i]
        correct = correct & trajs_are_identical(added_traj, multi_trajs[i])
    assert correct


def test_coverage(cov_trajs):
    trajs = [np.array([1,2,3])]
    num_states = np.unique(trajs[0]).shape[0]
    cov_trajs.add_trajectories(trajs)
    assert cov_trajs.num_covered_states == num_states