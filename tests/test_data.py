import pandas as pd

from avsa import data


def test_load_disengagements_parses_month():
    df = data.load_disengagements()
    assert "period" in df.columns
    assert "month_num" in df.columns
    assert df["month_num"].between(1, 12).all()
    assert len(df) > 0


def test_load_total_miles_renames_columns():
    df = data.load_total_miles()
    for col in [
        "miles_driven_autonomous",
        "total_disengagements",
        "automatic_disengagements",
        "manual_disengagements",
    ]:
        assert col in df.columns
    assert (df["miles_driven_autonomous"] >= 0).all()


def test_load_all_returns_both_frames():
    df_diseng, df_miles = data.load_all()
    assert isinstance(df_diseng, pd.DataFrame)
    assert isinstance(df_miles, pd.DataFrame)
