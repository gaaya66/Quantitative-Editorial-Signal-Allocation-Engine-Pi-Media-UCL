"""
evaluation.py
-------------
Phase 5b: evaluate the walk-forward backtest.

The final model compares two methods:

    1. Momentum
    2. Naive recent-activity baseline

For each method we report:

    - mean Precision@5
    - median Precision@5
    - standard deviation
    - number of evaluation periods
    - lift versus the baseline

The baseline is the reference method, so its lift is defined as 0%.
"""

from __future__ import annotations

import pandas as pd


BACKTEST_RESULTS_PATH = (
    "data/processed/backtest_results.csv"
)

EVALUATION_RESULTS_PATH = (
    "reports/evaluation_summary.csv"
)

METHOD_NAMES = [
    "momentum",
    "baseline",
]


def load_backtest_results(
    path: str,
) -> pd.DataFrame:
    """Load the completed walk-forward backtest."""

    return pd.read_csv(
        path,
        parse_dates=["week", "next_week"],
    )


def summarise_method(
    df: pd.DataFrame,
    method_name: str,
    baseline_mean: float,
) -> dict:
    """Calculate summary statistics for one method."""

    precision_column = (
        f"precision_{method_name}"
    )

    precision_values = df[
        precision_column
    ]

    mean_precision = precision_values.mean()

    median_precision = precision_values.median()

    std_precision = precision_values.std()

    n_periods = precision_values.count()

    if method_name == "baseline":
        lift_vs_baseline = 0.0
    else:
        lift_vs_baseline = (
            mean_precision / baseline_mean
        ) - 1

    return {
        "method": method_name,
        "mean_precision": mean_precision,
        "median_precision": median_precision,
        "std_precision": std_precision,
        "n_periods": n_periods,
        "lift_vs_baseline": lift_vs_baseline,
    }


def build_summary_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a summary table comparing momentum and baseline."""

    baseline_mean = df[
        "precision_baseline"
    ].mean()

    summaries = [
        summarise_method(
            df,
            method_name,
            baseline_mean,
        )
        for method_name in METHOD_NAMES
    ]

    return pd.DataFrame(summaries)


def validate_summary(
    summary: pd.DataFrame,
) -> None:
    """Validate the evaluation summary."""

    assert set(summary["method"]) == set(
        METHOD_NAMES
    )

    assert summary["n_periods"].gt(0).all()

    assert summary[
        "mean_precision"
    ].between(0, 1).all()

    assert summary[
        "median_precision"
    ].between(0, 1).all()

    assert summary[
        "std_precision"
    ].ge(0).all()

    print(
        "Evaluation validation passed."
    )


if __name__ == "__main__":

    backtest_results = load_backtest_results(
        BACKTEST_RESULTS_PATH
    )

    summary_table = build_summary_table(
        backtest_results
    )

    validate_summary(summary_table)

    print("\nEvaluation summary:")

    for _, row in summary_table.iterrows():

        print(
            f"  {row['method']}: "
            f"mean={row['mean_precision']:.3f}, "
            f"median={row['median_precision']:.3f}, "
            f"std={row['std_precision']:.3f}, "
            f"N={int(row['n_periods'])}, "
            f"lift={row['lift_vs_baseline']:.1%}"
        )

    summary_table.to_csv(
        EVALUATION_RESULTS_PATH,
        index=False,
    )

    print(
        f"\nWrote {EVALUATION_RESULTS_PATH}"
    )