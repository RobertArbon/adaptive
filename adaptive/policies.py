import numpy as np

from adaptive.containers import CoverageRun
from adaptive.dynamics import SamplingConfig


class SampledSystemStatistics:
    def __init__(self, cov_run: CoverageRun) -> None:
        states, counts = np.unique(cov_run.trajectories, return_counts=True)
        ix = np.argsort(states)
        self._states = states[ix].astype(int)
        self._counts = counts[ix].astype(int)

    @property
    def states(self) -> np.ndarray:
        return self._states

    @property
    def counts(self) -> np.ndarray:
        return self._counts

    @property
    def inv_count_probability(self) -> np.ndarray:
        inv_counts = 1.0/self.counts
        return inv_counts/np.sum(inv_counts)


def inverse_microcounts(cov_run: CoverageRun)  -> SamplingConfig:
    if cov_run.n_epochs == 0:
        raise ValueError('Policies require data')
    stats = SampledSystemStatistics(cov_run)
    n_walkers = cov_run.epochs[-1].n_walkers
    traj_length = cov_run.epochs[-1].n_steps

    starting_states = np.random.choice(stats.states, p=stats.inv_count_probability,
                                       replace=True, size=n_walkers)
    traj_lengths = np.repeat(traj_length, repeats=n_walkers)

    return SamplingConfig(starting_states=starting_states,
                          traj_lengths=traj_lengths)


def naive_walkers(cov_run: CoverageRun) -> SamplingConfig:
    if cov_run.n_epochs == 0:
        raise ValueError('Policies require data')
    starting_states = np.array([walker.trajectory[-1] for walker in cov_run.epochs[-1].walkers])
    traj_lengths = np.array([walker.trajectory.shape[0] for walker in cov_run.epochs[-1].walkers])
    return SamplingConfig(starting_states=starting_states,
                          traj_lengths=traj_lengths)
