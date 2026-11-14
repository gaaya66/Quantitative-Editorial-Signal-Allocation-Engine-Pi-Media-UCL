"""
robustness.py
-------------
Phase 6: sensitivity of the momentum result to window choice
(4/12, 6/18, 8/24) against a fixed trailing 4-week baseline.

All specifications and the baseline are scored on the SAME weeks (those
where the longest window is valid) with the same random tie-break draws,
so the comparison is like-for-like.
"""

from __future__ import annotations

import pandas as pd

from src.signals import compute_momentum
from src.backtest import (
    compute_baseline,
    weekly_precision_table,
    BASELINE_WINDOW,
    TOP_N,
)
from src.evaluation import block_bootstrap_ci

COMPOSITE_FEATURES_PATH = "data/processed/composite_features.csv"
ROBUSTNESS_RESULTS_PATH = "data/processed/robustness_results.csv"

MOMENTUM_WINDOW_COMBINATIONS = [(4, 12), (6, 18), (8, 24)]
EXPECTED_N_TOPICS = 16
EXPECTED_N_SPECIFICATIONS = len(MOMENTUM_WINDOW_COMBINATIONS)


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    return df.sort_values(["topic", "week"]).reset_index(drop=True)


def run_robustness_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["baseline"] = compute_baseline(df, window=BASELINE_WINDOW)

    spec_columns = {}
    for short_window, long_window in MOMENTUM_WINDOW_COMBINATIONS:
        col = f"momentum_{short_window}_{long_window}"
        df[col] = compute_momentum(df, short_window=short_window, long_window=long_window)
        spec_columns[(short_window, long_window)] = col

    # One weekly table for baseline + every spec: identical weeks, identical draws.
    weekly = weekly_precision_table(df, ["baseline", *spec_columns.values()])
    baseline = weekly["precision_baseline"]
    random_precision = TOP_N / df["topic"].nunique()

    results = []
    for (short_window, long_window), col in spec_columns.items():
        precision = weekly[f"precision_{col}"]
        diff = precision - baseline
        ci_low, ci_high = block_bootstrap_ci(diff)
        results.append({
            "short_window": short_window,
            "long_window": long_window,
            "n_periods": len(weekly),
            "mean_precision": precision.mean(),
            "baseline_mean_precision": baseline.mean(),
            "baseline_n_periods": len(weekly),
            "lift_vs_baseline": (precision.mean() - baseline.mean()) / baseline.mean(),
            "diff_vs_baseline": diff.mean(),
            "diff_ci_low": ci_low,
            "diff_ci_high": ci_high,
            "random_precision": random_precision,
        })
    return pd.DataFrame(results)


def validate_robustness_results(results: pd.DataFrame, n_topics: int) -> None:
    assert n_topics == EXPECTED_N_TOPICS, f"Expected {EXPECTED_N_TOPICS} topics, found {n_topics}."
    assert len(results) == EXPECTED_N_SPECIFICATIONS
    assert not results.duplicated(subset=["short_window", "long_window"]).any()
    assert results["mean_precision"].between(0, 1).all()
    assert results["baseline_mean_precision"].nunique() == 1
    assert results["n_periods"].nunique() == 1, "Specs must share the same evaluation weeks."
    print("Structural validation checks passed.")


def print_robustness_results(results: pd.DataFrame) -> None:
    base = results["baseline_mean_precision"].iloc[0]
    n = results["n_periods"].iloc[0]
    print("\nRobustness analysis: momentum vs baseline")
    print(f"Baseline: trailing {BASELINE_WINDOW}-week mean; P@5 = {base:.3f} "
          f"({n} common evaluation periods); random = {results['random_precision'].iloc[0]:.3f}\n")
    header = f"{'Short':>6}{'Long':>6}{'MeanP@5':>9}{'Lift':>9}{'Diff':>9}{'95% CI':>20}"
    print(header)
    print("-" * len(header))
    for _, r in results.iterrows():
        ci = f"[{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]"
        print(f"{int(r['short_window']):>6}{int(r['long_window']):>6}"
              f"{r['mean_precision']:>9.3f}{r['lift_vs_baseline']:>+9.1%}"
              f"{r['diff_vs_baseline']:>+9.3f}{ci:>20}")
    print("\nFull grid reported as-is; no specification selected as a new default.")


if __name__ == "__main__":
    features = load_features(COMPOSITE_FEATURES_PATH)
    results = run_robustness_grid(features)
    validate_robustness_results(results, features["topic"].nunique())
    results.to_csv(ROBUSTNESS_RESULTS_PATH, index=False)
    print_robustness_results(results)
    print(f"\nWrote {ROBUSTNESS_RESULTS_PATH}")
