"""
backtest.py
-----------
Phase 5: point-in-time walk-forward backtest.

At each eligible week t, two ranking methods predict the top 5 topics
for the following week:

    1. momentum
       Ranks topics by their 4-week / 12-week activity momentum.

    2. baseline
       Ranks topics by their trailing 4-week mean activity.

The prediction is then compared with the actual top 5 topics in week
t+1 using Precision@5:

    Precision@5 =
        |predicted_top_5 ∩ actual_top_5| / 5

POINT-IN-TIME DISCIPLINE
------------------------
A prediction at week t uses only information available through week t.

Week t+1 activity is used only to evaluate the prediction after it has
been made. It never contributes to the prediction itself.

The Z-score signal was excluded from the final model because the sparse
topic-level time series frequently produced zero historical volatility.
The original Z-score experiment is therefore not part of the final
backtest.
"""

from __future__ import annotations

import pandas as pd


BASELINE_WINDOW = 4
TOP_N = 5

COMPOSITE_FEATURES_PATH = (
    "data/processed/composite_features.csv"
)

BACKTEST_RESULTS_PATH = (
    "data/processed/backtest_results.csv"
)

METHOD_COLUMNS = {
    "momentum": "momentum",
    "baseline": "baseline",
}


def load_composite_features(
    path: str,
) -> pd.DataFrame:
    """
    Load the Phase 4 feature table and sort chronologically
    within each topic.
    """

    df = pd.read_csv(
        path,
        parse_dates=["week"],
    )

    df = (
        df.sort_values(["topic", "week"])
        .reset_index(drop=True)
    )

    return df


def compute_baseline(
    df: pd.DataFrame,
    window: int = BASELINE_WINDOW,
) -> pd.Series:
    """
    Calculate the naive recent-activity baseline.

    baseline(t,k) =
        mean(activity over current + previous window-1 weeks)

    The current week is included, matching the short-term window
    convention used by the momentum signal.
    """

    grouped_activity = df.groupby("topic")["activity"]

    baseline = grouped_activity.transform(
        lambda topic_activity: topic_activity.rolling(
            window=window,
            min_periods=window,
        ).mean()
    )

    return baseline


def rank_top_n(
    week_data: pd.DataFrame,
    value_column: str,
    n: int = TOP_N,
) -> list[str]:
    """
    Rank topics within one week.

    Values are sorted descending. Ties are broken alphabetically
    by topic name so that results are fully deterministic.
    """

    ranked = week_data.sort_values(
        by=[value_column, "topic"],
        ascending=[False, True],
    )

    return ranked["topic"].head(n).tolist()


def precision_at_5(
    predicted_topics: list[str],
    actual_topics: list[str],
) -> float:
    """
    Calculate Precision@5.
    """

    overlap = len(
        set(predicted_topics) & set(actual_topics)
    )

    return overlap / TOP_N


def run_backtest(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the point-in-time walk-forward backtest.

    Each eligible week t produces one evaluation row comparing
    predictions made at t with actual activity observed at t+1.
    """

    df = df.copy()

    # Construct the naive baseline using only activity through
    # each current week.
    df["baseline"] = compute_baseline(df)

    all_weeks = sorted(df["week"].unique())

    week_to_data = {
        week: week_df
        for week, week_df in df.groupby("week")
    }

    results = []

    # The final week cannot be evaluated because there is no t+1.
    for i, current_week in enumerate(
        all_weeks[:-1]
    ):

        next_week = all_weeks[i + 1]

        current_week_data = week_to_data[
            current_week
        ]

        next_week_data = week_to_data[
            next_week
        ]

        # Only evaluate when every topic has a valid value for
        # every ranking method.
        required_columns = list(
            METHOD_COLUMNS.values()
        )

        if current_week_data[
            required_columns
        ].isna().any().any():
            continue

        # This is the outcome we are trying to predict.
        actual_top_5 = rank_top_n(
            next_week_data,
            "activity",
        )

        row = {
            "week": current_week,
            "next_week": next_week,
        }

        # Generate each prediction using information available
        # at current_week only.
        for (
            method_name,
            value_column,
        ) in METHOD_COLUMNS.items():

            predicted_top_5 = rank_top_n(
                current_week_data,
                value_column,
            )

            row[
                f"predicted_{method_name}"
            ] = ";".join(predicted_top_5)

            row[
                f"precision_{method_name}"
            ] = precision_at_5(
                predicted_top_5,
                actual_top_5,
            )

        row["actual_top_5"] = ";".join(
            actual_top_5
        )

        results.append(row)

    return pd.DataFrame(results)


def validate_backtest_results(
    results: pd.DataFrame,
) -> None:
    """
    Validate the finished backtest table.
    """

    assert len(results) > 0, (
        "No eligible evaluation weeks were found."
    )

    assert not results[
        "week"
    ].duplicated().any(), (
        "Duplicate evaluation weeks found."
    )

    for method_name in METHOD_COLUMNS:

        precision_column = (
            f"precision_{method_name}"
        )

        assert results[
            precision_column
        ].between(0, 1).all(), (
            f"{precision_column} contains values "
            "outside the valid [0, 1] range."
        )

    print(
        "Structural validation checks passed:"
    )

    print(
        f"  - {len(results)} evaluation periods, "
        "no duplicate weeks"
    )

    print(
        "  - all Precision@5 values lie in [0, 1]"
    )


if __name__ == "__main__":

    composite_features = (
        load_composite_features(
            COMPOSITE_FEATURES_PATH
        )
    )

    backtest_results = run_backtest(
        composite_features
    )

    validate_backtest_results(
        backtest_results
    )

    backtest_results.to_csv(
        BACKTEST_RESULTS_PATH,
        index=False,
    )

    n_periods = len(backtest_results)

    mean_momentum = (
        backtest_results[
            "precision_momentum"
        ].mean()
    )

    mean_baseline = (
        backtest_results[
            "precision_baseline"
        ].mean()
    )

    print("\nBacktest summary:")
    print(
        f"  Evaluation periods: {n_periods}"
    )
    print(
        f"  Momentum mean Precision@5: "
        f"{mean_momentum:.3f}"
    )
    print(
        f"  Baseline mean Precision@5: "
        f"{mean_baseline:.3f}"
    )

    print(
        f"\nWrote {BACKTEST_RESULTS_PATH}"
    )