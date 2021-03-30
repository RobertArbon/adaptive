from typing import Callable, Optional, List, Set

import numpy as np

from adaptive.containers import Epoch, Walker, CoverageRun


def cum_states_at_epoch_start(cov_run: CoverageRun) -> List[Set]:
    # Includes last epoch so array is n_epochs + 1 long
    cum_states = set()
    result = [set()]
    for i in range(cov_run.n_epochs):
        epoch = cov_run.epochs[i]
        cum_states = cum_states.union(set(epoch.states_visited))
        result.append(cum_states)
    return result


def cum_steps_at_epoch_start(cov_run: CoverageRun) -> np.ndarray:
    # Includes last epoch so array is n_epochs + 1 long
    result = [0]
    for i in range(cov_run.n_epochs):
        epoch = cov_run.epochs[i]
        result.append(epoch.n_steps)
    result = np.cumsum(np.array(result))
    return result


def cum_n_states_at_epoch_start(cov_run: CoverageRun) -> np.ndarray:
    cumulative_state_covered = cum_states_at_epoch_start(cov_run)
    result = np.array([len(x) for x in cumulative_state_covered])
    return result


def find_epoch_which_covers(cov_run: CoverageRun, required_coverage: int) -> (int, Epoch):
    cumulative_n_states_covered = cum_n_states_at_epoch_start(cov_run)
    epoch_ix = np.min(np.where(cumulative_n_states_covered >= required_coverage)[0]) - 1
    epoch = cov_run.epochs[epoch_ix]
    return epoch_ix, epoch


def cover_times(cov_runs: List[CoverageRun], required_coverage: int) -> np.ndarray:
    """ Convenience function. Runs cover_time over a list of coverage runs

    Parameters
    ----------
    cov_runs : List[CoverageRun]
        the output from run_experiment
    required_coverage : int
        number of states to be considered covered.

    Returns
    -------
    np.ndarray
        the cover times.
    """
    ctimes = np.empty(len(cov_runs))
    for i, cov_run in enumerate(cov_runs):
        ctimes[i] = cover_time(cov_run, required_coverage)
    return ctimes



def cover_time(cov_run: CoverageRun, required_coverage: int) -> int:
    """ Calculates the number of steps to cover `required_coverage` number of states.

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
    cumulative_state_covered = cum_states_at_epoch_start(cov_run)
    cumulative_steps = cum_steps_at_epoch_start(cov_run)

    epoch_ix, epoch = find_epoch_which_covers(cov_run, required_coverage)

    all_walkers = epoch.walkers_as_array
    covered_states = cumulative_state_covered[epoch_ix]
    for step in range(epoch.n_steps):
        covered_states = covered_states.union(set(all_walkers[step, :]))
        if len(covered_states) == required_coverage:
            cover_step = step
            break

    ctime = int(cumulative_steps[epoch_ix] + cover_step) + 1
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
