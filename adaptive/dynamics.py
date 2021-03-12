from typing import  Optional

import pyemma as pm
import numpy as np

from adaptive.containers import Walker, Epoch


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

