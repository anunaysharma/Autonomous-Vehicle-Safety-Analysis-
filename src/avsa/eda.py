"""Descriptive / exploratory summaries of the disengagement data."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSummary:
    total_disengagements: int
    unique_months: int
    unique_locations: list
    unique_causes: int
    missing_values: dict


def summarize(df_diseng: pd.DataFrame, df_miles: pd.DataFrame) -> DatasetSummary:
    """Reproduce the headline summary statistics used in the original analysis."""
    return DatasetSummary(
        total_disengagements=int(df_miles["total_disengagements"].sum()),
        unique_months=int(df_diseng["month_num"].nunique()),
        unique_locations=sorted(df_diseng["Location"].unique().tolist()),
        unique_causes=int(df_diseng["Cause"].nunique()),
        missing_values={
            "disengagements": df_diseng.isnull().sum().to_dict(),
            "total_miles": df_miles.isnull().sum().to_dict(),
        },
    )


def top_causes(df_diseng: pd.DataFrame, n: int = 2) -> pd.Series:
    """Return the n most frequent disengagement causes, most frequent first."""
    return df_diseng["Cause"].value_counts().head(n)


def disengagement_rate_by_month(df_miles: pd.DataFrame) -> pd.DataFrame:
    """Monthly disengagements-per-mile, sorted chronologically."""
    monthly = (
        df_miles.groupby("period")[["total_disengagements", "miles_driven_autonomous"]]
        .sum()
        .reset_index()
        .sort_values("period")
    )
    monthly["disengagements_per_mile"] = (
        monthly["total_disengagements"] / monthly["miles_driven_autonomous"]
    )
    return monthly
