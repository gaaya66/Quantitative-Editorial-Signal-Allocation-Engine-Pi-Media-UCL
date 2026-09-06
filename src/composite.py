"""
composite.py
------------
Phase 4: construct the final editorial signal score.

The original project tested a composite of momentum and abnormal-activity
Z-score. The Z-score was subsequently excluded because the sparse
topic-level time series produced zero historical volatility in 1,885 of
3,136 topic-week observations.

The final engineered signal therefore uses momentum alone:

    Score(t,k) = Momentum(t,k)

This module keeps a separate `score` column so that downstream
backtesting and allocation code can operate on a consistent signal
interface.
"""

from __future__ import annotations

import pandas as pd


SIGNAL_FEATURES_PATH = "data/processed/signal_features.csv"
COMPOSITE_FEATURES_PATH = "data/processed/composite_features.csv"

EXPECTED_N_TOPICS = 16
EXPECTED_N_ROWS = 3136
EXPECTED_MOMENTUM_NANS = 176


def load_signal_features(path: str) -> pd.DataFrame:
    """Load the Phase 3 signal features table."""

    df = pd.read_csv(
        path,
        parse_dates=["week"],
    )

    return df


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """
    Return the final engineered signal score.

    Momentum is currently the sole primary signal, so the score is
    identical to momentum.
    """

    return df["momentum"].copy()


def build_composite_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the final `score` column while preserving the existing
    Phase 3 columns.
    """

    out = df.copy()

    out["score"] = compute_composite_score(out)

    return out


def validate_composite_features(
    df: pd.DataFrame,
) -> None:
    """Run structural checks on the final signal table."""

    n_topics = df["topic"].nunique()

    assert n_topics == EXPECTED_N_TOPICS, (
        f"Expected {EXPECTED_N_TOPICS} topics, "
        f"found {n_topics}."
    )

    n_rows = len(df)

    assert n_rows == EXPECTED_N_ROWS, (
        f"Expected {EXPECTED_N_ROWS} rows, "
        f"found {n_rows}."
    )

    assert not df.duplicated(
        subset=["week", "topic"]
    ).any(), (
        "Duplicate (week, topic) rows found."
    )

    for column in [
        "activity",
        "momentum",
        "score",
    ]:
        assert column in df.columns, (
            f"Expected column '{column}' was not found."
        )

    # Score should be missing exactly where momentum is missing.
    momentum_missing = df["momentum"].isna()
    score_missing = df["score"].isna()

    assert (
        momentum_missing == score_missing
    ).all(), (
        "score is NaN/non-NaN in a different pattern "
        "than momentum."
    )

    # Since score is currently defined as momentum, the values
    # should be identical.
    pd.testing.assert_series_equal(
        df["score"],
        df["momentum"],
        check_names=False,
    )

    actual_missing = int(df["score"].isna().sum())

    assert actual_missing == EXPECTED_MOMENTUM_NANS, (
        f"Expected {EXPECTED_MOMENTUM_NANS} NaN scores, "
        f"found {actual_missing}."
    )

    print("Structural validation checks passed:")
    print(f"  - {n_topics} topics")
    print(f"  - {n_rows} total rows")
    print("  - no duplicate (week, topic) pairs")
    print("  - score is NaN exactly where momentum is NaN")
    print("  - score equals momentum")


if __name__ == "__main__":

    signal_features = load_signal_features(
        SIGNAL_FEATURES_PATH
    )

    composite_features = build_composite_features(
        signal_features
    )

    validate_composite_features(
        composite_features
    )

    composite_features.to_csv(
        COMPOSITE_FEATURES_PATH,
        index=False,
    )

    n_rows = len(composite_features)
    n_topics = composite_features["topic"].nunique()
    n_missing_score = int(
        composite_features["score"].isna().sum()
    )
    n_valid_score = n_rows - n_missing_score

    print("\nSummary:")
    print(f"  Rows:                {n_rows}")
    print(f"  Topics:              {n_topics}")
    print(f"  Rows with score:     {n_valid_score}")
    print(f"  Rows with NaN score: {n_missing_score}")

    print(
        f"\nWrote {COMPOSITE_FEATURES_PATH}"
    )