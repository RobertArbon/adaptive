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


def trajs_are_identical(a: List[np.ndarray], b: List[np.ndarray]):
    assert len(a) == len(b), 'traj lists are not same size'
    result = np.all([np.allclose(a[i], b[i]) for i in range(len(a))])
    return result


@pytest.fixture
def cov_trajs():
    return CoverageRun()


@pytest.fixture
def walker_and_traj():
    traj = np.arange(3)[::-1]
    walker = Walker(traj)
    return walker, traj


@pytest.fixture
def epoch_and_walkers():
    walkers = [Walker(np.arange(3)) for _ in range(3)]
    epoch = Epoch(walkers)
    return epoch, walkers


def test_walker_steps(walker_and_traj):
    walker, traj = walker_and_traj
    assert walker.n_steps == traj.shape[0]


def test_walker_states(walker_and_traj):
    walker, traj = walker_and_traj
    walker = Walker(traj)
    assert np.allclose(walker.states_visited, np.sort(np.unique(traj)))


def test_epoch_walkers(epoch_and_walkers):
    epoch, walkers = epoch_and_walkers
    assert epoch.n_walkers == len(walkers)


def test_epoch_states(epoch_and_walkers):
    epoch, walkers = epoch_and_walkers
    # walkers are all identical
    assert np.allclose(epoch.states_visited, walkers[0].states_visited)


def test_cov_run_epoch(cov_trajs):
    traj = np.arange(3)
    epoch = Epoch([Walker(traj)])
    cov_trajs.add(epoch)
    assert cov_trajs.n_epochs == 1


def test_cov_run_states(cov_trajs):
    traj = np.arange(3)
    epoch = Epoch([Walker(traj)])
    cov_trajs.add(epoch)
    assert np.allclose(cov_trajs.states_visited, np.sort(traj))
