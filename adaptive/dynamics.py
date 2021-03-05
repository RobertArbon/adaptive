from typing import List, Callable, Dict, Optional

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


class Trajectory:

    def __init__(self) -> None:
        self.trajectories = {}
        self.covered_states = set()

    def _append_trajectories(self, new_trajs: List[np.ndarray]) -> None:
        num_epochs = len(self.trajectories)
        self.trajectories[num_epochs + 1] = new_trajs

    def _update_state_coverage(self, new_trajs: List[np.ndarray]) -> None:
        unique_new_states = np.unique(np.concatenate(new_trajs))
        self.covered_states.add(unique_new_states)

    def add_trajectories(self, new_trajs: List[np.ndarray]) -> None:
        self._append_trajectories(new_trajs)
        self._update_state_coverage(new_trajs)

    @property
    def num_covered_states(self) -> int:
        return len(self.covered_states)


def single_matrix_cover(dynamics: Dynamics, policy: Callable, max_epochs: Optional[int]=int(1e3)) -> Trajectory:
    trajectories = Trajectory()
    for i in range(max_epochs):
        config = policy(trajectories)
        new_trajectories = dynamics.sample(config)
        trajectories.add_trajectories(new_trajectories)
        if trajectories.num_covered_states == dynamics.n_states:
            return trajectories
