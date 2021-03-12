from typing import Callable, Optional

import numpy as np

from adaptive.containers import Epoch, Walker, CoverageRun


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
    n_states_per_epoch = [] #np.empty(cov_run.n_epochs)
    states_per_epoch = [] #np.empty(cov_run.n_epochs)
    for i, epoch in enumerate(cov_run.epochs):
        states = set(epoch.states_visited)
        cum_states = states.union(cum_states)

        n_states_per_epoch.append(len(cum_states))
        states_per_epoch.append(states)

    n_states_per_epoch = np.array(n_states_per_epoch)
    epoch_ix = np.min(np.where(n_states_per_epoch >= required_coverage)[0])

    epoch = cov_run.epochs[epoch_ix]

    good_walkers = []
    for i, walker in enumerate(epoch.walkers):
        states = set(walker.states_visited)
        if len(states_per_epoch[epoch_ix - 1].union(states)) >= required_coverage:
            good_walkers.append(walker)

    good_steps = np.empty(len(good_walkers))
    for i, walker in enumerate(good_walkers):
        cum_states = states_per_epoch[epoch_ix-1]
        for step, state in enumerate(walker.trajectory):
            cum_states.add(state)
            if len(cum_states) == required_coverage:
                good_steps[i] = step
                break

    first_step = np.min(good_steps)
    n_steps_per_epoch = np.array([x.n_steps for x in cov_run.epochs])
    cum_steps_per_epoch = np.cumsum(n_steps_per_epoch)
    ctime = int(cum_steps_per_epoch[epoch_ix-1] + first_step) + 1
    return ctime







     # epoch = find_first_to_cover(cov_run)
#     walker = find_first_to_cover(cov_run.epochs[epoch])
#     step = find_first_to_cover(cov_run.epochs[epoch].walkers[walker])
#
# def cum_unique_states(epochs: List[Epoch]) -> np.ndarray:
#     cumulative_states = set()
#     for epoch in epochs:
#         states_visited = set(epoch.states_visited)
#         cumulative_states = cumulative_states.union(states_visited)



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
