"""
Unit and regression test for the adaptive package.
"""

# Import package, test suite, and other packages as needed
import numpy as np
import pytest

import adaptive as ad


@pytest.fixture
def create_dynamics_engine():
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    mm = ad.Dynamics(trans_mat=P)
    return mm


def test_dynamics_returns_good_arrays(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([0, 1])
    traj_lengths = np.array([10, 11])
    config = ad.SamplingConfig(starting_states=starting_states,
                            traj_lengths=traj_lengths)
    epoch = mm.sample(config)
    traj_lengths_correct = np.allclose(traj_lengths, [x.n_steps for x in epoch.walkers])
    start_states_correct = np.allclose(starting_states, [x.trajectory[0] for x in epoch.walkers])
    assert traj_lengths_correct & start_states_correct


def test_dynamics_bad_start_states(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([3, 1])
    traj_lengths = np.array([10, 10])
    config = ad.SamplingConfig(starting_states=starting_states,
                               traj_lengths=traj_lengths)
    with pytest.raises(ValueError):
        config.is_valid(mm.n_states)


def test_dynamics_bad_lengths(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([0, 1])
    traj_lengths = np.array([10, -1])
    config = ad.SamplingConfig(starting_states=starting_states,
                               traj_lengths=traj_lengths)
    with pytest.raises(ValueError):
        config.is_valid()


def test_dynamics_inconsistent_init(create_dynamics_engine):
    mm = create_dynamics_engine
    starting_states = np.array([0, 1, 1])
    traj_lengths = np.array([10, 10])
    config = ad.SamplingConfig(starting_states=starting_states,
                               traj_lengths=traj_lengths)
    with pytest.raises(ValueError):
        config.is_valid()







