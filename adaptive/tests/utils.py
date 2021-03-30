from typing import Optional

from scipy.stats import ttest_1samp
import numpy as np


def is_equivalent(sample: np.ndarray, target: float, window: float, alpha: Optional[float] = 0.05) -> bool:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5502906/
    lb = -window / 2
    ub = window / 2
    t_lb = ttest_1samp(sample, target + lb, alternative='greater')
    t_ub = ttest_1samp(sample, target + ub, alternative='less')
    pvalue = np.max([t_lb.pvalue, t_ub.pvalue])
    return pvalue < alpha

