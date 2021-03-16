import numpy as np
from typing import NamedTuple

from adaptive.containers import CoverageRun, Walker, Epoch
from adaptive.statistics import *


def test_states_at_epoch():
    cov_run = CoverageRun()
    cov_run.add(Epoch([Walker(np.array([1, 2]))]))
    cov_run.add(Epoch([Walker(np.array([1, 3]))]))

    a = [set(), {1, 2}, {1, 2, 3}]
    b = cum_states_at_epoch_start(cov_run)
    assert np.all([x == y for x, y in zip(a, b)])


def test_states_at_epoch2():
    cov_run = CoverageRun()
    cov_run.add(Epoch([Walker(np.array([0, 1])),
                       Walker(np.array([0, 1]))]))
    cov_run.add(Epoch([Walker(np.array([0, 1])),
                       Walker(np.array([0, 2]))]))
    a = [set(), {0, 1}, {0, 1, 2}]
    b = cum_states_at_epoch_start(cov_run)
    assert np.all([x == y for x, y in zip(a, b)])


def test_steps_at_epoch():
    cov_run = CoverageRun()
    cov_run.add(Epoch([Walker(np.arange(3))]))
    cov_run.add(Epoch([Walker(np.arange(4))]))
    cov_run.add(Epoch([Walker(np.arange(5))]))
    est = cum_steps_at_epoch_start(cov_run)
    act = [0, 3, 7, 12]
    assert np.allclose(est, act)


def test_ctime_1():
    cov_run = CoverageRun()
    cov_run.add(Epoch([Walker(np.array([1, 2, 3]))]))
    cov_run.add(Epoch([Walker(np.array([0, 2, 3]))]))
    est = cover_time(cov_run, required_coverage=4)
    act = 4
    assert est == act


def test_ctime_2():
    cov_run = CoverageRun()
    cov_run.add(Epoch([Walker(np.array([1, 2, 3])),
                       Walker(np.array([1, 2, 3]))]))
    cov_run.add(Epoch([Walker(np.array([0, 2, 3])),
                       Walker(np.array([1, 2, 3]))]))
    est = cover_time(cov_run, required_coverage=4)
    act = 4
    assert est == act


def test_ctime_3():
    cov_run = CoverageRun()
    cov_run.add(Epoch([Walker(np.array([1, 2, 3])),
                       Walker(np.array([1, 2, 0]))]))
    cov_run.add(Epoch([Walker(np.array([0, 2, 3])),
                       Walker(np.array([1, 2, 3]))]))
    est = cover_time(cov_run, required_coverage=4)
    act = 3
    assert est == act


# np.random.seed(2039840)
#
#
# class FakeDatasetConfig(NamedTuple):
#     n_states: int
#     n_epochs: int
#     n_walkers: int
#     traj_length: int
#
#
# class DatasetCoordinates(NamedTuple):
#     epoch: int
#     step: int
#     walker: int
#
#
#
# def trajs_with_ctime(config: FakeDatasetConfig) -> (CoverageRun, int):
#     cov_run = CoverageRun()
#     traj = fake_total_trajectory(config)
#     cover_ix = cover_index(traj, config.n_states)
#     # print('\ncover_ix', cover_ix)
#     ctime = 0
#     for epoch_ix in range(config.n_epochs):
#         epoch = Epoch()
#         for walker_ix in range(config.n_walkers):
#             start_ix = (epoch_ix*config.n_walkers + walker_ix)*config.traj_length
#             stop_ix = start_ix + config.traj_length
#
#             if (cover_ix >= start_ix) and (cover_ix < stop_ix):
#                 ctime = epoch_ix*config.traj_length + (cover_ix - start_ix) + 1
#                 print('start_ix, ', start_ix, ' stop_ix ', stop_ix, ' cover_ix', cover_ix, ' ctime', ctime)
#             epoch.add(Walker(traj[start_ix:stop_ix]))
#         cov_run.add(epoch)
#     return cov_run, ctime
#
#
# def cover_index(traj: np.ndarray, required_states: int) -> int:
#     for i in range(traj.shape[0]):
#         if np.unique(traj[:i]).shape[0] == required_states:
#             return i - 1
#
#
# def fake_total_trajectory(config: FakeDatasetConfig) -> np.ndarray:
#     states = np.arange(config.n_states)
#     n_steps = config.n_epochs*config.n_walkers*config.traj_length
#     traj = np.empty(n_steps)
#     n = int(n_steps//states.shape[0])
#     m = int(n*states.shape[0])
#     traj[:m] = np.tile(states, (n))
#     traj[m:] = np.zeros(n_steps-m)
#     np.random.shuffle(traj)
#     return traj
