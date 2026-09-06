"""
signals.py

Phase 3: construct a momentum signal from the weekly topic activity panel.

Momentum uses inclusive trailing windows:

    Momentum(t,k) =
        mean(activity over current + previous 3 weeks)
        / mean(activity over current + previous 11 weeks) - 1

All calculations are performed independently by topic.

The Z-score signal was tested during development but excluded from the
final signal because the sparse topic-level time series frequently produced
zero historical volatility, making the statistic unstable or undefined.
"""

from __future__ import annotations

import pandas as pd


# Signal parameters
SHORT_WINDOW = 4
LONG_WINDOW = 12
EPSILON = 1e-8

# File paths
WEEKLY_PANEL_PATH = "data/processed/weekly_topic_panel.csv"
SIGNAL_FEATURES_PATH = "data/processed/signal_features.csv"

# Structural expectations
EXPECTED_N_TOPICS = 16
EXPECTED_N_ROWS = 3136


def load_weekly_panel(path: str) -> pd.DataFrame:
    """Load and chronologically sort the weekly topic activity panel."""

    df = pd.read_csv(path, parse_dates=["week"])

    df = (
        df.sort_values(["topic", "week"])
        .reset_index(drop=True)
    )

    return df


def compute_momentum(
    df: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    epsilon: float = EPSILON,
) -> pd.Series:
    """
    Calculate trailing activity momentum for each topic.

    The short and long windows are inclusive of the current week.

    For example, with short_window=4:

        short-term mean =
        mean(activity[t-3], ..., activity[t])

    The first 11 weeks have no valid long-term window and therefore
    produce NaN momentum values.
    """

    grouped_activity = df.groupby("topic")["activity"]

    short_term_mean = grouped_activity.transform(
        lambda topic_activity: topic_activity.rolling(
            window=short_window,
            min_periods=short_window,
        ).mean()
    )

    long_term_mean = grouped_activity.transform(
        lambda topic_activity: topic_activity.rolling(
            window=long_window,
            min_periods=long_window,
        ).mean()
    )

    momentum = (
        short_term_mean
        / (long_term_mean + epsilon)
        - 1
    )

    return momentum


def build_signal_features(
    df: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Add the momentum signal to the weekly activity panel."""

    out = df.copy()

    out["momentum"] = compute_momentum(
        out,
        short_window=short_window,
        long_window=long_window,
        epsilon=epsilon,
    )

    return out


def validate_signal_features(
    df: pd.DataFrame,
    long_window: int = LONG_WINDOW,
) -> None:
    """Run structural checks on the signal dataset."""

    n_topics = df["topic"].nunique()
    n_rows = len(df)

    assert n_topics == EXPECTED_N_TOPICS, (
        f"Expected {EXPECTED_N_TOPICS} topics, found {n_topics}"
    )

    assert n_rows == EXPECTED_N_ROWS, (
        f"Expected {EXPECTED_N_ROWS} rows, found {n_rows}"
    )

    assert not df.duplicated(["week", "topic"]).any(), (
        "Duplicate (week, topic) pairs detected"
    )

    expected_momentum_nans = (
        n_topics * (long_window - 1)
    )

    actual_momentum_nans = df["momentum"].isna().sum()

    assert actual_momentum_nans == expected_momentum_nans, (
        f"Expected {expected_momentum_nans} momentum NaNs, "
        f"found {actual_momentum_nans}"
    )

    print("Structural validation checks passed:")
    print(f"  - {n_topics} topics")
    print(f"  - {n_rows} total rows")
    print("  - no duplicate (week, topic) pairs")
    print(
        f"  - momentum NaNs: {actual_momentum_nans} "
        f"(expected {expected_momentum_nans})"
    )


def validate_no_lookahead(
    df: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    epsilon: float = EPSILON,
) -> None:
    """
    Verify that future observations do not affect historical signals.

    The signal calculated using the full dataset should be identical
    to the signal calculated using only observations available up to
    a chosen historical cutoff.
    """

    all_weeks = sorted(df["week"].unique())

    cutoff_index = len(all_weeks) * 3 // 4
    cutoff_week = all_weeks[cutoff_index]

    full_result = build_signal_features(
        df,
        short_window=short_window,
        long_window=long_window,
        epsilon=epsilon,
    )

    truncated_df = df[df["week"] <= cutoff_week].copy()

    truncated_result = build_signal_features(
        truncated_df,
        short_window=short_window,
        long_window=long_window,
        epsilon=epsilon,
    )

    full_cutoff = full_result[
        full_result["week"] <= cutoff_week
    ][["week", "topic", "momentum"]].reset_index(drop=True)

    truncated_cutoff = truncated_result[
        ["week", "topic", "momentum"]
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        full_cutoff,
        truncated_cutoff,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )

    print(
        f"  - no-lookahead check passed at {cutoff_week.date()}"
    )


def main() -> None:
    """Build, validate, and save the momentum signal dataset."""

    df = load_weekly_panel(WEEKLY_PANEL_PATH)

    signal_features = build_signal_features(df)

    validate_signal_features(signal_features)

    validate_no_lookahead(signal_features)

    signal_features.to_csv(
        SIGNAL_FEATURES_PATH,
        index=False,
    )

    print("\nSummary:")
    print(f"  Rows: {len(signal_features)}")
    print(f"  Topics: {signal_features['topic'].nunique()}")
    print(f"  Weeks: {signal_features['week'].nunique()}")
    print(
        f"  First week: "
        f"{signal_features['week'].min().date()}"
    )
    print(
        f"  Last week: "
        f"{signal_features['week'].max().date()}"
    )
    print(
        f"  NaN momentum: "
        f"{signal_features['momentum'].isna().sum()}"
    )

    print(
        f"\nWrote data to {SIGNAL_FEATURES_PATH}"
    )


if __name__ == "__main__":
    main()