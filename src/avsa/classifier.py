"""Naive Bayes classifier that predicts the disengagement-cause category
(Controller / Perception System / Computer System) from Location, Weather
and TypeOfTrigger.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB

CAUSE_TO_CLASS = {
    "Incorrect behavior prediction of others": "Controller",
    "Recklessly behaving agent": "Controller",
    "Unwanted Driver Discomfort": "Controller",
    "Adverse road surface conditions": "Perception System",
    "Emergency Vehicle": "Perception System",
    "Position Estimation Failure": "Perception System",
    "Incorrect Traffic Light Detection": "Perception System",
    "System Tuning and Calibration": "Computer System",
    "Hardware Fault": "Computer System",
    "Software Froze": "Computer System",
}

FEATURE_COLUMNS = ["Location", "Weather", "TypeOfTrigger"]


def assign_class(df_diseng: pd.DataFrame) -> pd.DataFrame:
    """Map each disengagement's free-text `Cause` onto one of three system classes.

    Rows whose cause isn't in `CAUSE_TO_CLASS` are dropped (matches original analysis,
    which only classified causes it had explicitly triaged).
    """
    df = df_diseng.copy()
    df["class"] = df["Cause"].map(CAUSE_TO_CLASS)
    return df.dropna(subset=["class"])


def encode_features(df: pd.DataFrame):
    """Label-encode the categorical features and target into numeric arrays."""
    encoders = {
        col: preprocessing.LabelEncoder().fit(df[col])
        for col in FEATURE_COLUMNS + ["class"]
    }
    X = pd.DataFrame({col: encoders[col].transform(df[col]) for col in FEATURE_COLUMNS})
    y = encoders["class"].transform(df["class"])
    return X, y, encoders


@dataclass
class ClassifierResult:
    accuracy: float
    confusion: np.ndarray
    report: str
    class_labels: list = field(default_factory=list)


def train_and_evaluate(
    X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 60
) -> ClassifierResult:
    """Train a Gaussian Naive Bayes classifier on a single train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = GaussianNB().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = cm.diagonal().sum() / cm.sum()
    report = classification_report(y_test, y_pred)
    return ClassifierResult(accuracy=accuracy, confusion=cm, report=report)


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    random_states=(None, 101, 42, 1, 10),
    n_splits: int = 5,
) -> float:
    """Average held-out accuracy across several random train/test splits."""
    accuracies = []
    for state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=state
        )
        model = GaussianNB().fit(X_train, y_train)
        cm = confusion_matrix(y_test, model.predict(X_test))
        accuracies.append(cm.diagonal().sum() / cm.sum())
    return float(np.mean(accuracies)) * 100


def kfold_cross_val_score(
    X: pd.DataFrame, y: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """K-fold cross-validation accuracy scores using scikit-learn's KFold directly."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=60)
    return cross_val_score(GaussianNB(), X, y, cv=kfold, n_jobs=1)
