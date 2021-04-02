from typing import List, Tuple, Optional, Union
import numpy as np
import pyemma as pm

from adaptive.containers import CoverageRun
from adaptive.dynamics import SamplingConfig


ZERO = 1E-8
METASTABLE_THRESHOLD = 1.5


class StationaryDistributions:
    def __init__(self, micro: np.ndarray, macro: np.ndarray) -> None:
        self._micro = micro
        self._macro = macro

    @property
    def micro(self) -> np.ndarray:
        return self._micro

    @property
    def macro(self) -> np.ndarray:
        return self._macro


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


def estimate_model(trajectories: List[np.ndarray], lag: Optional[int] = 1, **kwargs) -> pm.msm.MaximumLikelihoodMSM:
    return pm.msm.estimate_markov_model(trajectories, lag=lag, **kwargs)


def first_negative_index(array: np.ndarray) -> Union[int, None]:
    if np.all(array) > ZERO:
        return None
    else:
        return np.min(np.where(array <= ZERO)[0])


def get_good_eigenvalues(model: pm.msm.MaximumLikelihoodMSM) -> np.ndarray:
    eigenvalues = model.eigenvalues()
    first_negative_ix = first_negative_index(eigenvalues)
    if first_negative_ix is None:
        return eigenvalues
    else: 
        return eigenvalues[:first_negative_ix]


def determine_n_metastable(eigenvalues: np.ndarray) -> int:
    ratios = eigenvalues[:-1] / eigenvalues[1:]
    if eigenvalues.shape[0] <= 2:
        return 1
    elif np.max(ratios) < METASTABLE_THRESHOLD:
        return 1
    else:
        return np.argmax(ratios) + 1


def macro_stationary_distribution(model: pm.msm.MaximumLikelihoodMSM) -> np.ndarray:
    pi = np.empty(model.n_metastable)
    for i in range(model.n_metastable):
        pi[i] = np.sum(model.stationary_distribution[model.metastable_assignments == i])
    return pi


def stationary_ditributions(model: pm.msm.MaximumLikelihoodMSM, n_metastable: int) -> StationaryDistributions:
    model.pcca(n_metastable)
    macro_dist = macro_stationary_distribution(model)
    micro_dist = model.stationary_distribution
    dists = StationaryDistributions(micro_dist, macro_dist)
    return dists


def hierarchical_sample(distributions: StationaryDistributions, size=int) -> np.ndarray:
    sample = np.zeros(size, dtype=int)
    macro_states = distributions.macro.shape[0]
    micro_states = distributions.micro.shape[0]
    macro_ixs = np.random.choice(macro_states, p=distributions.macro, size=size)
    for i in range(size):
        sample[i] = np.random.choice()
    pass



def inverse_macrocount(cov_run: CoverageRun, **kwargs) -> SamplingConfig:
    # model = estimate_model(cov_run.trajectories, **kwargs)
    # eigenvalues = get_good_eigenvalues(model)
    # if eigenvalues is None:
    #     return inverse_microcounts(cov_run)
    # else:
    #     n_walkers = cov_run.epochs[-1].n_walkers
    #     traj_length = cov_run.epochs[-1].n_steps
    #     n_metastable = determine_n_metastable(eigenvalues)
    #     (macro_stat_dist, micro_stat_dist) = stationary_distributions(model, n_metastable)
    #     micro_ix = hierarchical_sample((macro_stat_dist, micro_stat_dist), size=n_walkers)
    pass



def inverse_microcounts(cov_run: CoverageRun) -> SamplingConfig:
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
