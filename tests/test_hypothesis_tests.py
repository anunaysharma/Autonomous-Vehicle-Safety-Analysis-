import pandas as pd

from avsa import hypothesis_tests


def test_one_sample_t_test_detects_shifted_mean():
    sample = pd.Series([5.0] * 30)
    result = hypothesis_tests.one_sample_t_test(sample, population_mean=1.0)
    assert result.reject_null is True


def test_one_sample_t_test_no_difference():
    sample = pd.Series([1.0, 1.1, 0.9, 1.0, 1.05, 0.95] * 5)
    result = hypothesis_tests.one_sample_t_test(sample, population_mean=1.0)
    assert result.reject_null is False


def test_two_proportion_z_test_runs():
    result = hypothesis_tests.two_proportion_z_test(835, 1024, 189, 1024)
    assert isinstance(result.p_value, float)


def test_ks_two_sample_test_identical_samples():
    s = pd.Series(range(100))
    result = hypothesis_tests.ks_two_sample_test(s, s)
    assert result.reject_null is False
