from avsa import data, eda


def test_summarize_matches_known_totals():
    df_diseng, df_miles = data.load_all()
    summary = eda.summarize(df_diseng, df_miles)
    assert summary.total_disengagements == df_miles["total_disengagements"].sum()
    assert summary.unique_causes == df_diseng["Cause"].nunique()
    assert len(summary.unique_locations) == df_diseng["Location"].nunique()


def test_top_causes_returns_requested_count():
    df_diseng, _ = data.load_all()
    top2 = eda.top_causes(df_diseng, n=2)
    assert len(top2) == 2
    assert top2.iloc[0] >= top2.iloc[1]


def test_disengagement_rate_by_month_is_nonnegative():
    _, df_miles = data.load_all()
    monthly = eda.disengagement_rate_by_month(df_miles)
    assert (monthly["disengagements_per_mile"] >= 0).all()
