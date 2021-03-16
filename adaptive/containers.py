from typing import List, Optional

import numpy as np


class Walker:
    def __init__(self, trajectory: np.ndarray) -> None:
        self._trajectory = trajectory

    @property
    def n_steps(self) -> int:
        return self.trajectory.shape[0]

    @property
    def n_states_visited(self) -> int:
        return len(self.states_visited)

    @property
    def states_visited(self) -> np.ndarray:
        return np.sort(np.unique(self.trajectory))

    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory

    def __str__(self):
        return np.array2string(self._trajectory)


class Epoch:
    def __init__(self, walkers: Optional[List[Walker]] = None) -> None:
        self._walkers = walkers

    @property
    def n_steps(self) -> int:
        if self.has_walkers():
            return int(np.max([x.n_steps for x in self._walkers]))

    @property
    def n_walkers(self) -> int:
        if self._walkers is not None:
            return len(self.walkers)
        else:
            return 0

    def has_walkers(self):
        if self._walkers is not None:
            return True
        else:
            raise ValueError('No walkers in this epoch')

    @property
    def states_visited(self) -> np.ndarray:
        if self.has_walkers():
            states = np.unique(np.concatenate([x.states_visited for x in self.walkers]))
            return np.sort(states)

    @property
    def n_states_visited(self) -> int:
        return self.states_visited.shape[0]

    def walkers_same_length(self):
        lengths = np.array([x.n_steps for x in self.walkers])
        return np.all(lengths == self.n_steps)

    @property
    def walkers_as_array(self) -> np.ndarray:
        if not self.walkers_same_length():
            raise NotImplementedError('Walkers need to be the same length.')
        array = np.concatenate([x.trajectory.reshape(-1, 1) for x in self.walkers], axis=1)
        return array

    @property
    def walkers(self) -> List[Walker]:
        if self.has_walkers():
            return self._walkers

    def add(self, walker: Walker) -> None:
        if self._walkers is None:
            self._walkers = [walker]
        else:
            self._walkers.append(walker)

    # @walkers.setter
    # def walkers(self, walkers: List[Walker]) -> None:
    #     if self.walkers is not None:
    #         self._walkers = walkers
    #     else:
    #         raise ValueError(f"walkers property already set")


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

    def __str__(self):
        msg = "\n"
        for epoch in self.epochs:
            msg += '='*80 + '\n'
            for walker in epoch.walkers:
                msg += f"{walker}\n"
        return msg
