import numpy as np
from typing import List


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
        states = np.sort(np.unique(states))
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
