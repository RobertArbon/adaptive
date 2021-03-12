from typing import Callable, Optional

import numpy as np

from adaptive.dynamics import Dynamics


def single_matrix_cover(dynamics: Dynamics, policy: Callable, max_epochs: Optional[int]=int(1e3)) -> CoverageRun:
    trajectories = CoverageRun()
    for i in range(max_epochs):
        config = policy(trajectories)
        new_trajectories = dynamics.sample(config)
        trajectories.add(new_trajectories)
        if trajectories.num_covered_states == dynamics.n_states:
            return trajectories

