from typing import List, Tuple, Optional, Union

import numpy as np
import pyemma as pm

from adaptive.containers import CoverageRun
from adaptive.dynamics import SamplingConfig


ZERO = 1E-8
METASTABLE_THRESHOLD = 1.5


class Clustering:
    def __init__(self, model: pm.msm.MaximumLikelihoodMSM) -> None:
        self._model = model
        self._memberships = model.metastable_memberships
        self._distributions = model.metastable_distributions
        self._stationary_distribution = model.stationary_distribution
        self._macrostates = np.arange(self._memberships.shape[1])
        self._microstates = np.arange(self._memberships.shape[0])

    @property
    def micro_stationary_distribution(self) -> np.ndarray:
        return self._model.stationary_distribution

    @property
    def macro_stationary_distribution(self) -> np.ndarray:
        macro = np.dot(self._memberships, self._stationary_distribution)
        return macro

    def micro_memberships(self, microstate: int) -> np.ndarray:
        return self._memberships[microstate, :]

    def macro_distribution(self, macrostate: int) -> np.ndarray:
        return self._distributions[macrostate, :]

    def sample_macro(self) -> int:
        return np.random.choice(self._macrostates, p=self.macro_stationary_distribution)

    def sample_micro_from_macro(self, macro_state: int) -> int:
        return np.random.choice(self._microstates, p=self.macro_distribution(macro_state))


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


def cluster_model(model: pm.msm.MaximumLikelihoodMSM, n_metastable: int) -> Clustering:
    model.pcca(n_metastable)
    pcca = Clustering(model)
    return pcca


def hierarchical_sample(clusters: Clustering, size: int) -> np.ndarray:
    samples = np.zeros(size, dtype=int)
    for i in range(size):
        macro_state = clusters.sample_macro()
        samples[i] = clusters.sample_micro_from_macro(macro_state=macro_state)
    return samples


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
