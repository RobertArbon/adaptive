from typing import List, Callable, Dict, Optional, Set

import pyemma as pm
import numpy as np


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

    def is_valid(self, n_states) -> True:
        self._lengths_valid()
        self._lengths_states_consistent()
        self._start_states_valid(n_states)
        return True


class Walker:
    def __init__(self, trajectory: np.ndarray) -> None:
        self.trajectory = trajectory

    @property
    def n_steps(self) -> int:
        return self.trajectory.shape[0]

    @property
    def n_states_visited(self) -> int:
        return len(self.states_visited)

    @property
    def states_visited(self) -> np.ndarray:
        return np.sort(np.unique(self.trajectory))


class Epoch:
    def __init__(self, walkers: List[Walker]) -> None:
        self.walkers = walkers

    @property
    def n_walkers(self) -> int:
        return len(self.walkers)

    @property
    def states_visited(self) -> np.ndarray:
        states = np.unique([x.states_visited for x in self.walkers])
        return np.sort(states)

    @property
    def n_states_visited(self) -> int:
        return self.states_visited.shape[0]


class Dynamics(object):

    def __init__(self, trans_mat: np.ndarray) -> None:
        self.model: pm.msm.MSM = pm.msm.MSM(trans_mat)
        self.n_states: int = trans_mat.shape[0]

    def sample(self, config: SamplingConfig) -> List[np.ndarray]:
        if config.is_valid(self.n_states):
            starting_states = config.starting_states
            traj_lengths = config.traj_lengths
            trajs = [self.model.simulate(N=x, start=y) for x, y in zip(traj_lengths, starting_states)]
            return trajs


class CoverageRun:
    """
    Container of trajectories used to calculate cover time
    """
    def __init__(self) -> None:
        self._epochs: List[Epoch] = []

    def add(self, new_epoch: Epoch) -> None:
        self._epochs.append(new_epoch)

    @property
    def states_visited(self) -> np.ndarray:
        states = np.concatenate([x.states_visited for x in self._epochs])
        return states

    @property
    def epochs(self) -> List[Epoch]:
        return self._epochs

    @property
    def n_epochs(self) -> int:
        return len(self._epochs)

    @property
    def n_states_visited(self) -> int:
        return len(self.states_visited)


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
