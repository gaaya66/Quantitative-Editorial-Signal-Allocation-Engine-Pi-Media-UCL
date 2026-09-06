"""
robustness.py
-------------
Phase 6: robustness / parameter sensitivity analysis for the
momentum editorial signal.

Tests momentum across three window specifications:

    1. 4-week / 12-week
    2. 6-week / 18-week
    3. 8-week / 24-week

For each specification:

    Momentum(t,k) =
        mean(short-term activity)
        / mean(long-term activity) - 1

Each specification is evaluated using the same point-in-time
walk-forward methodology as src/backtest.py.

The benchmark is a fixed trailing 4-week mean activity baseline.

The purpose of this analysis is not to select the best specification
after observing the results. Instead, the full grid is reported to
assess whether the conclusion is sensitive to reasonable parameter
choices.
"""

from __future__ import annotations

import pandas as pd

from src.signals import compute_momentum
from src.backtest import (
    compute_baseline,
    rank_top_n,
    precision_at_5,
    BASELINE_WINDOW,
)


COMPOSITE_FEATURES_PATH = (
    "data/processed/composite_features.csv"
)

ROBUSTNESS_RESULTS_PATH = (
    "data/processed/robustness_results.csv"
)


# (short-term window, long-term window)
MOMENTUM_WINDOW_COMBINATIONS = [
    (4, 12),
    (6, 18),
    (8, 24),
]

EXPECTED_N_TOPICS = 16

EXPECTED_N_SPECIFICATIONS = len(
    MOMENTUM_WINDOW_COMBINATIONS
)


def load_features(path: str) -> pd.DataFrame:
    """Load the activity panel used for robustness testing."""

    df = pd.read_csv(
        path,
        parse_dates=["week"],
    )

    df = (
        df.sort_values(["topic", "week"])
        .reset_index(drop=True)
    )

    return df


def compute_mean_precision_at_5(
    df: pd.DataFrame,
    score_column: str,
) -> tuple[float, int]:
    """
    Run the point-in-time walk-forward backtest for one
    ranking column.

    Returns:

        (mean Precision@5, number of evaluation periods)
    """

    all_weeks = sorted(
        df["week"].unique()
    )

    week_to_data = {
        week: week_df
        for week, week_df in df.groupby("week")
    }

    precisions = []

    # Final week cannot be evaluated because there is no t+1.
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

        # Require a valid signal for every topic.
        if current_week_data[
            score_column
        ].isna().any():
            continue

        predicted_top_5 = rank_top_n(
            current_week_data,
            score_column,
        )

        actual_top_5 = rank_top_n(
            next_week_data,
            "activity",
        )

        precisions.append(
            precision_at_5(
                predicted_top_5,
                actual_top_5,
            )
        )

    if len(precisions) == 0:
        return float("nan"), 0

    return (
        sum(precisions) / len(precisions),
        len(precisions),
    )


def run_robustness_grid(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate every momentum window specification against
    one fixed baseline.
    """

    df = df.copy()

    # The benchmark remains the original 4-week trailing
    # activity baseline for every specification.
    df["baseline"] = compute_baseline(
        df,
        window=BASELINE_WINDOW,
    )

    (
        baseline_mean_precision,
        baseline_n_periods,
    ) = compute_mean_precision_at_5(
        df,
        "baseline",
    )

    results = []

    for (
        short_window,
        long_window,
    ) in MOMENTUM_WINDOW_COMBINATIONS:

        momentum = compute_momentum(
            df,
            short_window=short_window,
            long_window=long_window,
        )

        spec_df = df.copy()
        spec_df["momentum"] = momentum

        (
            mean_precision,
            n_periods,
        ) = compute_mean_precision_at_5(
            spec_df,
            "momentum",
        )

        lift_vs_baseline = (
            mean_precision
            - baseline_mean_precision
        ) / baseline_mean_precision

        results.append(
            {
                "short_window": short_window,
                "long_window": long_window,
                "n_periods": n_periods,
                "mean_precision": mean_precision,
                "baseline_mean_precision": (
                    baseline_mean_precision
                ),
                "baseline_n_periods": (
                    baseline_n_periods
                ),
                "lift_vs_baseline": (
                    lift_vs_baseline
                ),
            }
        )

    return pd.DataFrame(results)


def validate_robustness_results(
    results: pd.DataFrame,
    n_topics: int,
) -> None:
    """Run structural checks on the robustness results."""

    assert n_topics == EXPECTED_N_TOPICS, (
        f"Expected {EXPECTED_N_TOPICS} topics, "
        f"found {n_topics}."
    )

    assert len(results) == (
        EXPECTED_N_SPECIFICATIONS
    ), (
        f"Expected {EXPECTED_N_SPECIFICATIONS} "
        f"specifications, found {len(results)}."
    )

    duplicate_specs = results.duplicated(
        subset=[
            "short_window",
            "long_window",
        ]
    )

    assert not duplicate_specs.any(), (
        "Duplicate momentum window specifications found."
    )

    assert results[
        "mean_precision"
    ].between(0, 1).all(), (
        "mean_precision contains values "
        "outside [0, 1]."
    )

    baseline_values = results[
        "baseline_mean_precision"
    ].unique()

    assert len(baseline_values) == 1, (
        "Baseline performance should be identical "
        "across all momentum specifications."
    )

    print(
        "Structural validation checks passed:"
    )

    print(
        f"  - {n_topics} topics"
    )

    print(
        f"  - {len(results)} momentum specifications"
    )

    print(
        "  - no duplicate window specifications"
    )

    print(
        "  - all mean_precision values lie in [0, 1]"
    )

    print(
        "  - baseline performance is identical "
        "across specifications"
    )


def print_robustness_results(
    results: pd.DataFrame,
) -> None:
    """Print the full robustness grid."""

    baseline_mean = results[
        "baseline_mean_precision"
    ].iloc[0]

    baseline_n_periods = results[
        "baseline_n_periods"
    ].iloc[0]

    print(
        "\nRobustness analysis: "
        "momentum vs baseline"
    )

    print(
        f"Baseline: trailing "
        f"{BASELINE_WINDOW}-week mean activity"
    )

    print(
        f"Baseline mean Precision@5 = "
        f"{baseline_mean:.3f} "
        f"({baseline_n_periods} evaluation periods)\n"
    )

    header = (
        f"{'Short':>8}"
        f"{'Long':>8}"
        f"{'N':>8}"
        f"{'MeanP@5':>12}"
        f"{'Lift':>12}"
    )

    print(header)
    print("-" * len(header))

    for _, row in results.iterrows():

        lift_str = (
            f"{row['lift_vs_baseline']:+.1%}"
        )

        print(
            f"{int(row['short_window']):>8}"
            f"{int(row['long_window']):>8}"
            f"{int(row['n_periods']):>8}"
            f"{row['mean_precision']:>12.3f}"
            f"{lift_str:>12}"
        )

    print(
        "\nFull robustness grid reported as-is; "
        "no specification is selected as a new default."
    )


if __name__ == "__main__":

    features = load_features(
        COMPOSITE_FEATURES_PATH
    )

    results = run_robustness_grid(
        features
    )

    validate_robustness_results(
        results,
        features["topic"].nunique(),
    )

    results.to_csv(
        ROBUSTNESS_RESULTS_PATH,
        index=False,
    )

    print_robustness_results(
        results
    )

    print(
        f"\nWrote {ROBUSTNESS_RESULTS_PATH}"
    )