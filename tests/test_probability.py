import pytest

from avsa import data, probability


def test_bayes_flip_matches_manual_calculation():
    # P(A|B) = P(B|A) * P(A) / P(B)
    assert probability.bayes_flip(0.5, 0.2, 0.4) == pytest.approx(0.25)


def test_disengagement_rate_between_zero_and_one():
    _, df_miles = data.load_all()
    rate = probability.disengagement_rate(df_miles)
    assert 0 <= rate <= 1


def test_prob_event_given_weather_is_a_probability():
    df_diseng, df_miles = data.load_all()
    for weather in ("cloudy", "clear"):
        p = probability.prob_event_given_weather(df_diseng, df_miles, weather)
        assert 0 <= p <= 1
