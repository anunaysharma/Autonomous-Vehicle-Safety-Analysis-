"""End-to-end pipeline: reproduces the full analysis and writes figures to disk.

Run with: python -m avsa.pipeline
"""

from pathlib import Path

from avsa import classifier, data, eda, hypothesis_tests, probability, visualize

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
NON_AV_MEAN_REACTION_TIME = 1.09  # seconds; human baseline used in the original study


def run(figures_dir: Path = FIGURES_DIR) -> None:
    df_diseng, df_miles = data.load_all()

    # 1. Descriptive summary
    summary = eda.summarize(df_diseng, df_miles)
    print("=== Dataset summary ===")
    print(summary)

    causes = eda.top_causes(df_diseng, n=2)
    print("\nTop 2 causes of disengagement:")
    print(causes)

    monthly = eda.disengagement_rate_by_month(df_miles)
    visualize.plot_disengagement_trend(monthly, figures_dir)
    visualize.plot_cause_pie(df_diseng["Cause"].value_counts(), figures_dir)
    visualize.plot_missing_map(
        df_diseng,
        "Missing values - disengagements",
        figures_dir,
        "missing_disengagements.png",
    )
    visualize.plot_reaction_time_distribution(df_diseng["ReactionTime"], figures_dir)

    # 2. Hypothesis test: AV vs. human reaction time
    t_result = hypothesis_tests.one_sample_t_test(
        df_diseng["ReactionTime"], NON_AV_MEAN_REACTION_TIME
    )
    print("\n=== AV vs. human reaction time (t-test) ===")
    print(t_result)

    # 3. Conditional probabilities of disengagement by weather
    p_dm_cloudy = probability.prob_event_given_weather(df_diseng, df_miles, "cloudy")
    p_dm_clear = probability.prob_event_given_weather(df_diseng, df_miles, "clear")
    print(f"\nP(disengagement/mile | cloudy) = {p_dm_cloudy:.4f}")
    print(f"P(disengagement/mile | clear)  = {p_dm_clear:.4f}")

    # 4. Weather effect on disengagement counts (z-test)
    n_cloudy = int((df_diseng["Weather"] == "cloudy").sum())
    n_clear = int((df_diseng["Weather"] == "clear").sum())
    z_result = hypothesis_tests.two_proportion_z_test(
        n_cloudy, len(df_diseng), n_clear, len(df_diseng)
    )
    print("\n=== Cloudy vs. clear disengagement rate (z-test) ===")
    print(z_result)

    # 5. Naive Bayes classification of disengagement cause category
    df_classified = classifier.assign_class(df_diseng)
    X, y, encoders = classifier.encode_features(df_classified)
    result = classifier.train_and_evaluate(X, y)
    print(f"\n=== Naive Bayes classifier ===\nHold-out accuracy: {result.accuracy:.2%}")
    print(result.report)

    cv_accuracy = classifier.cross_validate(X, y)
    print(f"Average accuracy across 5 random splits: {cv_accuracy:.2f}%")

    class_labels = list(encoders["class"].classes_)
    visualize.plot_confusion_matrix(result.confusion, class_labels, figures_dir)

    print(f"\nFigures written to {figures_dir}")


if __name__ == "__main__":
    run()
