"""
Unit and regression test for the adaptive package.
"""

# Import package, test suite, and other packages as needed
import adaptive as ad
import pytest
import sys
import numpy as np


@pytest.fixture
def create_dynamics_engine():
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    mm = ad.Dynamics(trans_mat=P)
    return mm


def test_dynamics_returns_good_arrays(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([0, 1])
    traj_lengths = np.array([10, 11])
    config = {'starting_states': starting_states,
              'traj_lengths': traj_lengths}
    trajs = mm.sample(config)
    traj_lengths_correct = np.allclose(traj_lengths, [x.shape[0] for x in trajs])
    start_states_correct = np.allclose(starting_states, [x[0] for x in trajs])
    assert traj_lengths_correct & start_states_correct


def test_dynamics_bad_start_states(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([3, 1])
    traj_lengths = np.array([10, 10])
    config = {'starting_states': starting_states,
              'traj_lengths': traj_lengths}
    with pytest.raises(ValueError):
        _ = mm.sample(config)


def test_dynamics_bad_lengths(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([0, 1])
    traj_lengths = np.array([10, -1])
    config = {'starting_states': starting_states,
              'traj_lengths': traj_lengths}
    with pytest.raises(ValueError):
        _ = mm.sample(config)


def test_dynamics_inconsistent_init(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([0, 1, 1])
    traj_lengths = np.array([10, 10])
    config = {'starting_states': starting_states,
              'traj_lengths': traj_lengths}
    with pytest.raises(ValueError):
        _ = mm.sample(config)







