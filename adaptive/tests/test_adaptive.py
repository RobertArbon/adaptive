import numpy as np

from adaptive.adaptive import *
from adaptive.dynamics import Dynamics
from adaptive.policies import naive_walkers
from adaptive.statistics import cover_times
from .utils import is_equivalent

t_01 = 0.4
MEAN_CTIME = 1 / t_01 + 1

experiment = Experiment(
    name='test',
    dynamics=Dynamics(trans_mat=np.array([[1 - t_01, t_01],
                                          [t_01, 1 - t_01]])),

    init_config=SamplingConfig(starting_states=np.array([0]),
                               traj_lengths=np.array([100])),
    policy=naive_walkers,
    n_runs=1000
)


def test_single_matrix_cover():
    # does it add data?
    cov_run = single_matrix_cover(experiment)
    assert cov_run.n_epochs > 0


# def test_run_experiment():
#     # does it add data?
#     results = run_experiment(experiment)
#     n_epochs = np.array([x.n_epochs for x in results])
#     assert np.all(n_epochs > 0)


def test_run_experiment_ctimes():
    results = run_experiment(experiment)
    ctimes = cover_times(results, required_coverage=2)
    assert is_equivalent(sample=ctimes, target=MEAN_CTIME,
                         window=1)


