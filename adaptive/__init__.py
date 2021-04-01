"""
adaptive
A package for prototyping adaptive sampling policies for molecular kinetics
"""

# Add imports here
from .dynamics import SamplingConfig, Dynamics
from .adaptive import Experiment, single_matrix_cover, run_experiment, run_trial
from .policies import inverse_microcounts, naive_walkers
from .statistics import cover_times

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
