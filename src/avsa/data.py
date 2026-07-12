"""Loading and cleaning for the CA DMV AV disengagement datasets."""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_MILES_COLUMN_MAP = {
    "miles driven in autonomous mode": "miles_driven_autonomous",
    "total number of disengagements": "total_disengagements",
    "number of automatic disengagements": "automatic_disengagements",
    "number of manual disengagements": "manual_disengagements",
}


def _parse_month(series: pd.Series) -> pd.DataFrame:
    """Split a 'YY-Mon' string column (e.g. '14-Sep') into year/month parts."""
    parts = series.str.split("-", expand=True)
    year = parts[0].astype(int) + 2000
    month = pd.to_datetime(parts[1], format="%b").dt.month
    period = pd.to_datetime(dict(year=year, month=month, day=1))
    return pd.DataFrame({"year": year, "month_num": month, "period": period})


def load_disengagements(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the per-event disengagement log (one row per disengagement)."""
    df = pd.read_csv(Path(data_dir) / "disengagements.csv")
    df = df.join(_parse_month(df["Month"]))
    return df


def load_total_miles(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the monthly miles-driven / disengagement-count summary table."""
    df = pd.read_csv(Path(data_dir) / "total_miles.csv")
    df = df.rename(columns=_MILES_COLUMN_MAP)
    df = df.join(_parse_month(df["Month"]))
    return df


def load_all(data_dir: Path = DEFAULT_DATA_DIR):
    """Convenience loader returning (disengagements, total_miles) dataframes."""
    return load_disengagements(data_dir), load_total_miles(data_dir)
