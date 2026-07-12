"""Plotting helpers. Each function saves a PNG to `output_dir` and returns its path."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def _save(fig, output_dir: Path, name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def plot_missing_map(df: pd.DataFrame, title: str, output_dir: Path, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    ax.set_title(title)
    return _save(fig, output_dir, name)


def plot_cause_pie(
    cause_counts: pd.Series, output_dir: Path, name: str = "causes_pie.png"
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(cause_counts, labels=cause_counts.index, autopct="%0.1f%%", shadow=True)
    ax.set_title("Causes of AV Disengagement")
    return _save(fig, output_dir, name)


def plot_disengagement_trend(
    monthly: pd.DataFrame, output_dir: Path, name: str = "disengagement_trend.png"
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly["period"], monthly["disengagements_per_mile"], marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Disengagements / Mile")
    ax.set_title("Disengagements per Mile Over Time")
    fig.autofmt_xdate()
    return _save(fig, output_dir, name)


def plot_reaction_time_distribution(
    reaction_times: pd.Series, output_dir: Path, name: str = "reaction_time_dist.png"
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(reaction_times.dropna(), kde=True, stat="density", ax=ax)
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Reaction Time")
    return _save(fig, output_dir, name)


def plot_confusion_matrix(
    cm: np.ndarray, labels: list, output_dir: Path, name: str = "confusion_matrix.png"
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    return _save(fig, output_dir, name)
