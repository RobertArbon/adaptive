"""
Tests whether the various trajectory contain classes - Walkers, Epochs, CoverageRuns,
have:
 1. the correct number of items (walkers have steps, epochs have walkers, CoverageRuns
epochs.
 2. accumulate the correct states
"""
from typing import List

import pytest
import numpy as np

import adaptive as ad
from adaptive.containers import CoverageRun, Epoch, Walker

TRAJ = np.arange(3)[::-1]
N_WALKERS = 3


def trajs_are_identical(a: List[np.ndarray], b: List[np.ndarray]):
    assert len(a) == len(b), 'traj lists are not same size'
    result = np.all([np.allclose(a[i], b[i]) for i in range(len(a))])
    return result


@pytest.fixture
def walker_and_traj():
    traj = np.array(TRAJ)
    walker = Walker(traj)
    return walker, traj


@pytest.fixture
def epoch_and_walkers():
    walkers = [Walker(np.array(TRAJ)) for _ in range(N_WALKERS)]
    epoch = Epoch(walkers)
    return epoch, walkers


def test_walker_n_steps(walker_and_traj):
    walker, traj = walker_and_traj
    assert walker.n_steps == traj.shape[0]


def test_walker_states_visited(walker_and_traj):
    walker, traj = walker_and_traj
    walker = Walker(traj)
    assert np.allclose(walker.states_visited, np.sort(np.unique(traj)))


def test_walker_n_states_visited(walker_and_traj):
    walker, traj = walker_and_traj
    walker = Walker(traj)
    assert np.allclose(walker.n_states_visited, np.unique(traj).shape[0])


def test_walker_trajectory(walker_and_traj):
    walker, traj = walker_and_traj
    walker = Walker(traj)
    assert np.allclose(walker.trajectory, traj)


def test_epoch_n_steps(epoch_and_walkers):
    epoch, walkers = epoch_and_walkers
    # walkers are all identical
    assert epoch.n_steps == walkers[0].n_steps


def test_epoch_n_walkers(epoch_and_walkers):
    epoch, walkers = epoch_and_walkers
    assert epoch.n_walkers == len(walkers)


def test_epoch_n_walkers_zero():
    epoch = Epoch()
    assert epoch.n_walkers == 0


def test_epoch_has_walkers_zero():
    epoch = Epoch()
    with pytest.raises(ValueError):
        epoch.has_walkers()


def test_epoch_states_visited(epoch_and_walkers):
    epoch, walkers = epoch_and_walkers
    assert np.allclose(epoch.states_visited, walkers[0].states_visited)


def test_epoch_n_states_visited(epoch_and_walkers):
    epoch, walkers = epoch_and_walkers
    # walkers are all identical
    assert epoch.n_states_visited == walkers[0].n_states_visited


def test_cov_run_n_epochs():
    epoch = Epoch([Walker(np.array(TRAJ))])
    cov_run = CoverageRun()
    cov_run.add(epoch)
    assert cov_run.n_epochs == 1


def test_cov_run_states_visited():
    epoch = Epoch([Walker(np.array(TRAJ))])
    cov_run = CoverageRun()
    cov_run.add(epoch)
    assert np.allclose(cov_run.states_visited, np.sort(np.unique(TRAJ)))


def test_cov_run_n_states_visited():
    epoch = Epoch([Walker(np.array(TRAJ))])
    cov_run = CoverageRun()
    cov_run.add(epoch)
    assert np.allclose(cov_run.n_states_visited, np.unique(TRAJ).shape[0])


