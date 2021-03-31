from typing import Callable, Optional, NamedTuple, List
from multiprocessing import Pool, cpu_count
from rich.progress import Progress

import numpy as np

from adaptive.containers import CoverageRun
from adaptive.dynamics import Dynamics, SamplingConfig


class Experiment(NamedTuple):
    name: str
    dynamics: Dynamics
    init_config: SamplingConfig
    policy: Callable
    n_runs: int
    max_epochs: int = int(1e3)


# with Progress() as progress:
#     task = progress.add_task(f'[green] Running experiment {experiment.name}',
#                              total=experiment.n_runs)

def run_experiment(experiment: Experiment) -> List[CoverageRun]:
    n_workers = cpu_count() - 1
    args = [experiment]*experiment.n_runs
    with Pool(n_workers) as pool:
        results = list(pool.imap_unordered(single_matrix_cover, args))
    return results


def single_matrix_cover(experiment: Experiment) -> CoverageRun:
    cov_run = CoverageRun()
    epoch = experiment.dynamics.sample(experiment.init_config)
    cov_run.add(epoch)
    for i in range(experiment.max_epochs):
        config = experiment.policy(cov_run)
        epoch = experiment.dynamics.sample(config)
        cov_run.add(epoch)
        if cov_run.n_states_visited == experiment.dynamics.n_states:
            break
    return cov_run