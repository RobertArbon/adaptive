import numpy as np
from typing import NamedTuple

from adaptive.containers import CoverageRun, Walker, Epoch
from adaptive.statistics import cover_time

np.random.seed(2039840)


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


def trajs_with_ctime(config: FakeDatasetConfig, coords: DatasetCoordinates) -> (CoverageRun, int):
    cov_run = create_random_training_set(config)
    epoch = cov_run.epochs[coords.epoch]
    walker = epoch.walkers[coords.walker]
    walker.trajectory[coords.step] = config.n_states - 1
    ctime = coords.epoch * config.traj_length + coords.step + 1
    return cov_run, ctime


def random_coordinates(config: FakeDatasetConfig) -> DatasetCoordinates:

    def rand_zero_to_n(n: int) -> int:
        return int(np.random.choice(np.arange(n)))

    coords = DatasetCoordinates(
        epoch=rand_zero_to_n(config.n_epochs),
        walker=rand_zero_to_n(config.n_walkers),
        step=rand_zero_to_n(config.traj_length)
    )
    return coords


# def test_states_per_epoch_1():
#     config = FakeDatasetConfig(
#         n_states=4,
#         n_walkers=2,
#         n_epochs=3,
#         traj_length=5
#     )
#     coords = DatasetCoordinates(
#         epoch=2,
#         walker=1,
#         step=3
#     )
#     cov_run, ctime = trajs_with_ctime(config, coords)
#     a = cover_time(cov_run, config.n_states)
#
#     assert a == ctime


def test_ctime_random_1():
    config = FakeDatasetConfig(
        n_states=4,
        n_walkers=2,
        n_epochs=3,
        traj_length=5
    )
    test = []
    n_tests = 100
    for _ in range(n_tests):
        coords = random_coordinates(config)
        cov_run, ctime = trajs_with_ctime(config, coords)
        a = cover_time(cov_run, config.n_states)
        is_equal = a == ctime
        # if not is_equal:
        #     print(cov_run)
        #     print(a, ctime)
        test.append(is_equal)
    assert np.sum(np.array(test)) == n_tests
