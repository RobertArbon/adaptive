import pytest
import numpy as np
from typing import NamedTuple, List

from adaptive.containers import CoverageRun, Walker, Epoch
from adaptive.statistics import cover_time


class FakeDatasetConfig(NamedTuple):
    n_states: int
    n_epochs: int
    n_walkers: int
    traj_length: int


class DatasetCoordinates(NamedTuple):
    epoch: int
    step: int
    walker: int


def create_random_training_set(config: FakeDatasetConfig) -> CoverageRun:
    cov_run = CoverageRun()
    state_array = np.arange(config.n_states-1)
    for i in range(config.n_epochs):
        epoch = Epoch()
        for j in range(config.n_walkers):
            random_walk = Walker(np.random.choice(state_array, replace=True, size=config.traj_length))
            epoch.add(random_walk)
        cov_run.add(epoch)
    return cov_run


def trajs_with_ctime(config: FakeDatasetConfig, coords:DatasetCoordinates) -> (CoverageRun, int):
    cov_run = create_random_training_set(config)
    epoch = cov_run.epochs[coords.epoch]
    walker = epoch.walkers[coords.walker]
    walker.trajectory[coords.step] = config.n_states - 1
    ctime = coords.epoch * config.traj_length + coords.step + 1
    return cov_run, ctime


def test_states_per_epoch_1():
    config = FakeDatasetConfig(
        n_states=4,
        n_walkers=2,
        n_epochs=3,
        traj_length=5
    )
    coords = DatasetCoordinates(
        epoch=2,
        walker=1,
        step=3
    )
    cov_run, ctime = trajs_with_ctime(config, coords)
    a = cover_time(cov_run, config.n_states)

    assert a == ctime