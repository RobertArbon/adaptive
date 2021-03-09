from typing import List, Callable, Optional, Set, Iterable

import pyemma as pm
import numpy as np

from adaptive.containers import Walker, Epoch, CoverageRun


class SamplingConfig:
    def __init__(self, starting_states: np.ndarray, traj_lengths:np.ndarray) -> None:
        self.starting_states = starting_states
        self.traj_lengths = traj_lengths

    def _lengths_states_consistent(self) -> None:
        if self.starting_states.shape[0]!= self.traj_lengths.shape[0]:
            raise ValueError(f'Dimension of starting_states ({len(self.starting_states)}) does not '
                             f'match that of traj_lengths ({len(self.traj_lengths)})')

    def _lengths_valid(self) -> None:
        if np.any(self.traj_lengths <= 0):
            raise ValueError(f'traj_lengths must be positive integers.')

    def _start_states_valid(self, n_states: int) -> None:
        if (np.min(self.starting_states) < 0) or (np.max(self.starting_states)> n_states-1):
            raise ValueError(f'starting states must be between 0 and n_states ({n_states}).')

    def is_valid(self, n_states: Optional[int] = None) -> True:
        self._lengths_valid()
        self._lengths_states_consistent()
        if n_states is not None:
            self._start_states_valid(n_states)
        return True


class Dynamics(object):

    def __init__(self, trans_mat: np.ndarray) -> None:
        self.model: pm.msm.MSM = pm.msm.MSM(trans_mat)
        self.n_states: int = trans_mat.shape[0]

    def sample(self, config: SamplingConfig) -> Epoch:
        if config.is_valid(self.n_states):
            starting_states = config.starting_states
            traj_lengths = config.traj_lengths
            trajs = [Walker(self.model.simulate(N=x, start=y)) for x, y in zip(traj_lengths, starting_states)]
            epoch = Epoch(trajs)
            return epoch


def cover_time(cov_run: CoverageRun, required_coverage: int) -> int:
    """ Calculates the number of steps to cover `num_states`

    Parameters
    ----------
    cov_run : CoverageRun
        The data from a single coverage run.
    required_coverage : int
        The required number of states to cover.

    Returns
    -------
    int :
        the cover time in number of steps.

    Raises
    ------
    ValueError:
        if the number of states is not covered.
    """

    cum_states = set()
    n_states_per_epoch = np.empty(cov_run.n_epochs)
    states_per_epoch = np.empty(cov_run.n_epochs)
    for i, epoch in enumerate(cov_run.epochs):
        states = set(epoch.states_visited)
        cum_states = states.union(cum_states)

        n_states_per_epoch[i] = len(cum_states)
        states_per_epoch[i] = states

    epoch_ix = np.min(np.where(n_states_per_epoch >= required_coverage)[0])

    epoch = cov_run.epochs[epoch_ix]

    walker_ixs = []
    for i, walker in enumerate(epoch.walkers):
        states = set(walker.states_visited)
        if len(states_per_epoch[epoch_ix - 1].union(states)) >= required_coverage:
            walker_ixs.append(i)



     # epoch_ix = find_first_to_cover(cov_run)
#     walker_ix = find_first_to_cover(cov_run.epochs[epoch_ix])
#     step_ix = find_first_to_cover(cov_run.epochs[epoch_ix].walkers[walker_ix])
#
# def cum_unique_states(epochs: List[Epoch]) -> np.ndarray:
#     cumulative_states = set()
#     for epoch in epochs:
#         states_visited = set(epoch.states_visited)
#         cumulative_states = cumulative_states.union(states_visited)


def single_matrix_cover(dynamics: Dynamics, policy: Callable, max_epochs: Optional[int]=int(1e3)) -> CoverageRun:
    trajectories = CoverageRun()
    for i in range(max_epochs):
        config = policy(trajectories)
        new_trajectories = dynamics.sample(config)
        trajectories.add(new_trajectories)
        if trajectories.num_covered_states == dynamics.n_states:
            return trajectories



# class CoverTimeStats:
#     """
#     Calculates cover time and associated statistics.
#     """
#     def __init__(self, coverage_run: CoverageRun) -> None:
#         self.trajectories = coverage_run
#         self.n_epochs: int = coverage_run.n_epochs
#         self._n_states_per_epoch: np.ndarray = np.empty(self.n_epochs)
#         self._n_steps_per_epoch: np.ndarray = np.empty(self.n_epochs)
#         self._walker_time_per_epoch: np.ndarray = np.empty(self.n_epochs)
#         self._calculate_per_epoch_stats()
#
#     @staticmethod
#     def _n_states(states: np.array) -> int:
#         return len(set(states))
#
#     @staticmethod
#     def _n_steps(states: np.array) -> int:
#         return len(states)
#
#     @staticmethod
#     def _max_walker_time(trajs: List[np.ndarray]) -> int:
#         return int(np.max([t.shape[0] for t in trajs]))
#
#     def _calculate_per_epoch_stats(self) -> None:
#         for i in range(self.n_epochs):
#             trajs = self.trajectories.trajectories[i]
#             all_states = np.concatenate(trajs)
#
#             self._walker_time_per_epoch[i] = self._max_walker_time(trajs)
#             self._n_states_per_epoch[i] = self._n_states(all_states)
#             self._n_steps_per_epoch[i] = self._n_steps(all_states)
#
#     @property
#     def n_states_per_epoch(self) -> np.ndarray:
#         return self._n_states_per_epoch
#
#     @property
#     def n_steps_per_epoch(self) -> np.ndarray:
#         return self._n_steps_per_epoch
#
#     @property
#     def walker_time_per_epoch(self) -> np.ndarray:
#         return self._walker_time_per_epoch
