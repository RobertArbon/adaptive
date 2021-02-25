import pyemma as pm
from typing import List
import numpy as np


class Dynamics(object):

    def __init__(self, trans_mat: np.ndarray) -> None:
        self.model: pm.msm.MSM = pm.msm.MSM(trans_mat)
        self.n_states: int = trans_mat.shape[0]

    def sample(self, starting_states: np.ndarray, traj_lengths: np.ndarray) -> List[np.ndarray]:

        if starting_states.shape[0]!= traj_lengths.shape[0]:
            raise ValueError(f'Dimension of starting_states ({len(starting_states)}) does not '
                             f'match that of traj_lengths ({len(traj_lengths)})')
        if np.any(traj_lengths <= 0):
            raise ValueError(f'traj_lengths must be positive integers.')
        if (np.min(starting_states)<0) or (np.max(starting_states)>self.n_states-1):
            raise ValueError(f'starting states must be between 0 and n_states ({self.n_states}).')

        trajs = [self.model.simulate(N=x, start=y) for x, y in zip(traj_lengths, starting_states)]

        return trajs


