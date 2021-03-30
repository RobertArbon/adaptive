import numpy as np

from adaptive.policies import *
from adaptive.containers import CoverageRun, Epoch, Walker

TRAJ = np.arange(3)
PROB_TOL = 1e-2

cov_run = CoverageRun()
cov_run.add(Epoch([Walker(np.array(TRAJ))]))
cov_run.add(Epoch([Walker(np.array(TRAJ))]))

start_states = np.repeat(0, 2)
traj_lengths = np.repeat(3, 2)
init_config = SamplingConfig(starting_states=start_states,
                             traj_lengths=traj_lengths)


def test_sampled_system_stats_states():
    stats = SampledSystemStatistics(cov_run)
    assert np.array_equal(stats.states, np.array(TRAJ))


def test_sampled_system_stats_counts():
    stats = SampledSystemStatistics(cov_run)
    assert np.array_equal(stats.counts, np.ones_like(TRAJ)*2)


def test_sampled_system_stats_inv_prob():
    stats = SampledSystemStatistics(cov_run)
    assert np.array_equal(stats.inv_count_probability, np.ones_like(TRAJ)/TRAJ.shape[0])


# def test_inverse_microcounts_init():
#     cov_run_empty = CoverageRun()
#     new_config = inverse_microcounts(cov_run_empty, init_config=init_config)
#
#     states_ok = np.allclose(new_config.starting_states, init_config.starting_states)
#     lengths_ok = np.allclose(new_config.traj_lengths, init_config.traj_lengths)
#     assert states_ok and lengths_ok


def test_inverse_microcounts_with_data_probs():
    new_configs = [inverse_microcounts(cov_run)
                   for _ in range(10000)]
    all_states = np.concatenate([x.starting_states for x in new_configs])
    states, counts = np.unique(all_states, return_counts=True)
    state_probs = counts/np.sum(counts)
    assert np.mean(np.abs(state_probs-np.ones_like(TRAJ)/TRAJ.shape[0])) < PROB_TOL


def test_inverse_microcounts_with_data_lengths():
    new_config = inverse_microcounts(cov_run)
    lengths = new_config.traj_lengths
    assert np.array_equal(lengths, [walkers.n_steps for walkers in cov_run.epochs[-1].walkers])


# def test_naive_walkers_init():
#     cov_run_empty = CoverageRun()
#     new_config = naive_walkers(cov_run_empty)
#     states_ok = np.allclose(new_config.starting_states, init_config.starting_states)
#     lengths_ok = np.allclose(new_config.traj_lengths, init_config.traj_lengths)
#     assert states_ok and lengths_ok


def test_naive_walkers():
    new_config = naive_walkers(cov_run)
    states_ok = np.allclose(new_config.starting_states, np.repeat(TRAJ[-1], cov_run.epochs[-1].n_walkers))
    lengths_ok = np.allclose(new_config.traj_lengths, np.repeat(TRAJ.shape[0], cov_run.epochs[-1].n_walkers))
    assert states_ok and lengths_ok


