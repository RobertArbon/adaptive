from typing import List, Callable, Dict, Optional, Set

import pyemma as pm
import numpy as np


class Dynamics(object):

    def __init__(self, trans_mat: np.ndarray) -> None:
        self.model: pm.msm.MSM = pm.msm.MSM(trans_mat)
        self.n_states: int = trans_mat.shape[0]

    def sample(self, config: Dict[str, np.ndarray]) -> List[np.ndarray]:
        starting_states = config['starting_states']
        traj_lengths = config['traj_lengths']
        if starting_states.shape[0]!= traj_lengths.shape[0]:
            raise ValueError(f'Dimension of starting_states ({len(starting_states)}) does not '
                             f'match that of traj_lengths ({len(traj_lengths)})')
        if np.any(traj_lengths <= 0):
            raise ValueError(f'traj_lengths must be positive integers.')
        if (np.min(starting_states)<0) or (np.max(starting_states)>self.n_states-1):
            raise ValueError(f'starting states must be between 0 and n_states ({self.n_states}).')

        trajs = [self.model.simulate(N=x, start=y) for x, y in zip(traj_lengths, starting_states)]

        return trajs


class CoverTrajectories:
    """
    Container of trajectories used to calculate cover time
    """
    def __init__(self) -> None:
        self._trajectories: List[List[np.ndarray]] = []
        self._covered_states: Set[int] = set()

    def _update_state_coverage(self, new_trajs: List[np.ndarray]) -> None:
        unique_new_states = np.unique(np.concatenate(new_trajs))
        for x in unique_new_states:
            self._covered_states.add(x)

    def add(self, new_trajs: List[np.ndarray]) -> None:
        self._trajectories.append(new_trajs)
        self._update_state_coverage(new_trajs)

    @property
    def num_covered_states(self) -> int:
        return len(self._covered_states)

    @property
    def trajectories(self) -> List[List[np.array]]:
        return self._trajectories

    @property
    def num_epochs(self) -> int:
        return len(self._trajectories)


class CoverTime:
    """
    Calculates cover time and associated statistics.
    """
    def __init__(self, trajectories: CoverTrajectories) -> None:
        self.trajectories = trajectories
        self.n_epochs: int = trajectories.num_epochs
        self._n_states_per_epoch: np.ndarray = np.empty(self.n_epochs)
        self._n_steps_per_epoch: np.ndarray = np.empty(self.n_epochs)
        self._walker_time_per_epoch: np.ndarray = np.empty(self.n_epochs)
        self._calculate_per_epoch_stats()

    @staticmethod
    def _n_states(states: np.array) -> int:
        return len(set(states))

    @staticmethod
    def _n_steps(states: np.array) -> int:
        return len(states)

    @staticmethod
    def _max_walker_time(trajs: List[np.ndarray]) -> int:
        return int(np.max([t.shape[0] for t in trajs]))

    def _calculate_per_epoch_stats(self) -> None:
        for i in range(self.n_epochs):
            trajs = self.trajectories.trajectories[i]
            all_states = np.concatenate(trajs)

            self._walker_time_per_epoch[i] = self._max_walker_time(trajs)
            self._n_states_per_epoch[i] = self._n_states(all_states)
            self._n_steps_per_epoch[i] = self._n_steps(all_states)

    @property
    def n_states_per_epoch(self) -> np.ndarray:
        return self._n_states_per_epoch

    @property
    def n_steps_per_epoch(self) -> np.ndarray:
        return self._n_steps_per_epoch

    @property
    def walker_time_per_epoch(self) -> np.ndarray:
        return self._walker_time_per_epoch


def single_matrix_cover(dynamics: Dynamics, policy: Callable, max_epochs: Optional[int]=int(1e3)) -> CoverTrajectories:
    trajectories = CoverTrajectories()
    for i in range(max_epochs):
        config = policy(trajectories)
        new_trajectories = dynamics.sample(config)
        trajectories.add(new_trajectories)
        if trajectories.num_covered_states == dynamics.n_states:
            return trajectories
