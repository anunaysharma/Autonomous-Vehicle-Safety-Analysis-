"""Conditional-probability calculations (Bayes' theorem) on weather / trigger type.

These functions generalize the ad-hoc, copy-pasted arithmetic from the original
notebook (which repeated the same four-line calculation for every combination
of weather and trigger type) into two reusable, testable functions.
"""

import pandas as pd

# California DMV climate assumption used throughout the original analysis.
P_CLEAR_DAY = 0.72
P_CLOUDY_DAY = 1 - P_CLEAR_DAY


def disengagement_rate(
    df_miles: pd.DataFrame, column: str = "total_disengagements"
) -> float:
    """P(disengagement per mile) = total disengagements of `column` / total miles driven."""
    return df_miles[column].sum() / df_miles["miles_driven_autonomous"].sum()


def prob_weather_given_event(
    df_diseng: pd.DataFrame, weather: str, trigger_type: str | None = None
) -> float:
    """P(weather | event), estimated as the observed share of logged events with that weather."""
    subset = (
        df_diseng
        if trigger_type is None
        else df_diseng[df_diseng["TypeOfTrigger"] == trigger_type]
    )
    if len(subset) == 0:
        return float("nan")
    return (subset["Weather"] == weather).mean()


def bayes_flip(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """Bayes' theorem: given P(B|A), P(A) and P(B), return P(A|B)."""
    return (p_b_given_a * p_a) / p_b


def prob_event_given_weather(
    df_diseng: pd.DataFrame,
    df_miles: pd.DataFrame,
    weather: str,
    trigger_type: str | None = None,
) -> float:
    """P(disengagement (of `trigger_type`, if given) per mile | weather).

    Replaces the four near-duplicate blocks in the original notebook that computed
    P(DM|cloudy), P(DM|clear), P(ADM|cloudy) and P(ADM|clear) by hand.
    """
    p_weather = P_CLOUDY_DAY if weather == "cloudy" else P_CLEAR_DAY
    p_weather_given_event = prob_weather_given_event(df_diseng, weather, trigger_type)

    miles_column = (
        "total_disengagements" if trigger_type is None else "automatic_disengagements"
    )
    p_event = disengagement_rate(df_miles, miles_column)

    return bayes_flip(p_weather_given_event, p_event, p_weather)


def prob_accident_per_mile(
    df_diseng: pd.DataFrame,
    df_miles: pd.DataFrame,
    reaction_time_threshold: float,
    trigger_type: str = "automatic",
) -> float:
    """P(accident per mile) via the law of total probability over weather conditions.

    P(accident) = P(event | cloudy) * P(cloudy) + P(event | clear) * P(clear),
    where "event" here is a disengagement with reaction time above the given threshold.
    """

    def _conditional(weather: str) -> float:
        weather_trigger = df_diseng[
            (df_diseng["Weather"] == weather)
            & (df_diseng["TypeOfTrigger"] == trigger_type)
        ]
        if len(weather_trigger) == 0:
            return 0.0
        return (weather_trigger["ReactionTime"] > reaction_time_threshold).mean()

    return _conditional("cloudy") * P_CLOUDY_DAY + _conditional("clear") * P_CLEAR_DAY
