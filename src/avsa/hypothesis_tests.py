"""Statistical hypothesis tests used in the AV safety analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    conclusion: str


def one_sample_t_test(
    sample: pd.Series,
    population_mean: float,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> TestResult:
    """One-sample t-test comparing `sample`'s mean against a known population mean.

    Used to compare AV reaction time against the human (non-AV) baseline.
    """
    sample = sample.dropna()
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    t_stat = (sample.mean() - population_mean) / se
    df = len(sample) - 1
    p_value = (1 - stats.t.cdf(abs(t_stat), df)) * (2 if two_tailed else 1)
    reject = bool(p_value < alpha)
    conclusion = (
        "Reject H0: the sample mean differs significantly from the population mean."
        if reject
        else "Fail to reject H0: no significant difference from the population mean."
    )
    return TestResult(float(t_stat), float(p_value), alpha, reject, conclusion)


def two_proportion_z_test(
    count_a: int, n_a: int, count_b: int, n_b: int, alpha: float = 0.05
) -> TestResult:
    """Two-sample z-test for a difference in disengagement counts between two conditions
    (e.g. cloudy vs. clear days), matching the original notebook's weather comparison.
    """
    p_a = count_a / n_a
    se = np.sqrt(p_a * (1 - p_a) / n_a + (count_b / n_b) * (1 - count_b / n_b) / n_b)
    z_stat = ((count_a / n_a) - (count_b / n_b)) / se
    p_value = (1 - stats.norm.cdf(abs(z_stat))) * 2
    reject = bool(p_value < alpha)
    conclusion = (
        "Reject H0: disengagement rates differ significantly between conditions."
        if reject
        else "Fail to reject H0: no significant difference between conditions."
    )
    return TestResult(float(z_stat), float(p_value), alpha, reject, conclusion)


def ks_two_sample_test(
    sample_a: pd.Series, sample_b: pd.Series, alpha: float = 0.05
) -> TestResult:
    """Kolmogorov-Smirnov test for whether two samples are drawn from the same distribution."""
    statistic, p_value = stats.ks_2samp(sample_a.dropna(), sample_b.dropna())
    reject = bool(p_value < alpha)
    conclusion = (
        "Reject H0: the two samples come from different distributions."
        if reject
        else "Fail to reject H0: no evidence the samples come from different distributions."
    )
    return TestResult(float(statistic), float(p_value), alpha, reject, conclusion)
