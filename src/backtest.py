"""
backtest.py
-----------
Phase 5: point-in-time walk-forward backtest.

At each eligible week t, two methods (momentum, trailing-activity baseline)
predict the top 5 topics for week t+1; Precision@5 compares that with the
actual top 5 in t+1.

TIE-BREAKING
------------
Activity is sparse, so many topics tie (often at zero), both in the signal
and in the actual week-t+1 activity. Breaking ties alphabetically picks the
SAME topics on both sides and inflates Precision@5. Ties are therefore
broken randomly and Precision@5 is averaged over N_DRAWS draws. The actual
top 5 is drawn once per draw and shared by all methods, so methods are
compared against identical outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_WINDOW = 4
TOP_N = 5
N_DRAWS = 200
SEED = 0

COMPOSITE_FEATURES_PATH = "data/processed/composite_features.csv"
BACKTEST_RESULTS_PATH = "data/processed/backtest_results.csv"

METHOD_COLUMNS = {"momentum": "momentum", "baseline": "baseline"}


def load_composite_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    return df.sort_values(["topic", "week"]).reset_index(drop=True)


def compute_baseline(df: pd.DataFrame, window: int = BASELINE_WINDOW) -> pd.Series:
    """Trailing mean activity over the current + previous window-1 weeks."""
    return df.groupby("topic")["activity"].transform(
        lambda a: a.rolling(window=window, min_periods=window).mean()
    )


def rank_top_n(week_data: pd.DataFrame, value_column: str,
               rng: np.random.Generator, n: int = TOP_N) -> list[str]:
    """Top n topics by value (descending); ties broken randomly."""
    ranked = (
        week_data.assign(_tie=rng.random(len(week_data)))
        .sort_values([value_column, "_tie"], ascending=[False, False])
    )
    return ranked["topic"].head(n).tolist()


def precision_at_5(predicted_topics: list[str], actual_topics: list[str]) -> float:
    return len(set(predicted_topics) & set(actual_topics)) / TOP_N


def expected_precisions(current_week_data: pd.DataFrame,
                        next_week_data: pd.DataFrame,
                        value_columns: list[str],
                        rng: np.random.Generator,
                        draws: int = N_DRAWS) -> dict[str, float]:
    """Mean Precision@5 over random tie-break draws, for each value column."""
    totals = dict.fromkeys(value_columns, 0.0)
    for _ in range(draws):
        actual = rank_top_n(next_week_data, "activity", rng)
        for col in value_columns:
            predicted = rank_top_n(current_week_data, col, rng)
            totals[col] += precision_at_5(predicted, actual)
    return {col: total / draws for col, total in totals.items()}


def weekly_precision_table(df: pd.DataFrame, value_columns: list[str],
                           seed: int = SEED) -> pd.DataFrame:
    """
    One row per evaluation week with a precision_<column> for each column.
    A week is evaluated only if EVERY column is valid for every topic, so
    all methods are always scored on exactly the same weeks.
    """
    rng = np.random.default_rng(seed)
    all_weeks = sorted(df["week"].unique())
    week_to_data = {w: g for w, g in df.groupby("week")}

    rows = []
    for current_week, next_week in zip(all_weeks[:-1], all_weeks[1:]):
        current = week_to_data[current_week]
        if current[value_columns].isna().any().any():
            continue
        prec = expected_precisions(current, week_to_data[next_week], value_columns, rng)
        rows.append({
            "week": current_week,
            "next_week": next_week,
            **{f"precision_{c}": p for c, p in prec.items()},
        })
    return pd.DataFrame(rows)


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["baseline"] = compute_baseline(df)
    return weekly_precision_table(df, list(METHOD_COLUMNS.values()))


def validate_backtest_results(results: pd.DataFrame) -> None:
    assert len(results) > 0, "No eligible evaluation weeks were found."
    assert not results["week"].duplicated().any(), "Duplicate evaluation weeks found."
    for method_name in METHOD_COLUMNS:
        col = f"precision_{method_name}"
        assert results[col].between(0, 1).all(), f"{col} outside [0, 1]."
    print("Structural validation checks passed:")
    print(f"  - {len(results)} evaluation periods, no duplicate weeks")
    print("  - all Precision@5 values lie in [0, 1]")


if __name__ == "__main__":
    features = load_composite_features(COMPOSITE_FEATURES_PATH)
    results = run_backtest(features)
    validate_backtest_results(results)
    results.to_csv(BACKTEST_RESULTS_PATH, index=False)

    print("\nBacktest summary:")
    print(f"  Evaluation periods: {len(results)}")
    print(f"  Momentum mean Precision@5: {results['precision_momentum'].mean():.3f}")
    print(f"  Baseline mean Precision@5: {results['precision_baseline'].mean():.3f}")
    print(f"  Random level: {TOP_N / features['topic'].nunique():.3f}")
    print(f"\nWrote {BACKTEST_RESULTS_PATH}")
